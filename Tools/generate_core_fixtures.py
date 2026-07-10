#!/usr/bin/env python3
"""M3/M4 fixture generator: dumps reference per-module inputs/outputs (M3) and a full-stack
forward parity dump (M4) for the Swift fixture-diff tests (phase-2 §4).

Runs the pinned reference implementation (modeling_llada2_moe.py from
inclusionAI/LLaDA2.1-mini, fetched via huggingface_hub) on a toy config with
seeded random weights, and saves:

  <out>/weights.safetensors   full model state_dict (HF key layout)
  <out>/tensors.safetensors   per-module input/output pairs
  <out>/manifest.json         config + seeds + provenance

Usage (toy config, dev machine):
  python3 Tools/generate_core_fixtures.py

Studio BF16 real-weight run (M3 acceptance, second half):
  python3 Tools/generate_core_fixtures.py --config <hf_dir>/config.json \
      --weights <hf_dir> --dtype bfloat16 --out scratch/core_fixtures_bf16

Note (sourced, verified 2026-07-07): the reference `generate()` builds a 0/1-valued
4D block mask; transformers returns 4D masks as-is, so eager/SDPA attention *adds*
the 0/1 values to the scores (a soft bias, NOT -inf masking). Module-level fixtures
here therefore use a proper additive 0/-inf mask — mask *semantics* are an input to
the attention module, not part of its math. The quirk is recorded in the manifest
and in phase-2 §6; resolution is an M4/M5 concern.
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile

import torch
from safetensors.torch import save_file

REPO_ID = "inclusionAI/LLaDA2.1-mini"
REFERENCE_FILES = ["configuration_llada2_moe.py", "modeling_llada2_moe.py"]


def import_reference_package():
    """Fetch the reference .py files from the HF hub cache and import them as a package
    (the modeling file uses a relative import, so a real package dir is required)."""
    from huggingface_hub import hf_hub_download

    pkg_dir = tempfile.mkdtemp(prefix="llada2_ref_")
    pkg_name = "llada2_reference"
    dest = os.path.join(pkg_dir, pkg_name)
    os.makedirs(dest)
    open(os.path.join(dest, "__init__.py"), "w").close()
    for fname in REFERENCE_FILES:
        shutil.copy(hf_hub_download(REPO_ID, fname), os.path.join(dest, fname))
    sys.path.insert(0, pkg_dir)
    cfg_mod = importlib.import_module(f"{pkg_name}.configuration_llada2_moe")
    model_mod = importlib.import_module(f"{pkg_name}.modeling_llada2_moe")
    return cfg_mod, model_mod


def randomize_weights(model, seed):
    """Seeded random weights: norms near 1, expert bias small, everything else N(0, 0.05)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.state_dict().items():
            if "norm" in name.lower() and name.endswith(".weight"):
                new = 1.0 + (torch.rand(param.shape, generator=gen) - 0.5) * 0.5
            elif name.endswith("expert_bias"):
                new = (torch.rand(param.shape, generator=gen) - 0.5) * 0.1
            else:
                new = torch.randn(param.shape, generator=gen) * 0.05
            param.copy_(new)


def block_masks(total_len, block_len, dtype):
    """Reference-style tril expansion (0/1) and its strict additive (0/-inf) conversion."""
    nb = total_len // block_len
    tril = torch.tril(torch.ones(nb, nb))
    mask01 = tril.repeat_interleave(block_len, 0).repeat_interleave(block_len, 1)
    mask01 = mask01[None, None].to(dtype)
    additive = torch.where(mask01.bool(), 0.0, float("-inf")).to(dtype)
    return mask01, additive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scratch/dummy_hf/config.json")
    ap.add_argument("--weights", default=None,
                    help="Optional dir with real safetensors weights; random weights if omitted")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default="scratch/core_fixtures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-length", type=int, default=16)
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    cfg_mod, model_mod = import_reference_package()

    with open(args.config) as f:
        cfg_json = json.load(f)
    cfg_json.pop("architectures", None)
    # attn_implementation="eager": the pinned parity target (§1); SDPA agrees to tolerance.
    config = cfg_mod.LLaDA2MoeConfig(**cfg_json, attn_implementation="eager")

    torch.manual_seed(args.seed)
    model = model_mod.LLaDA2MoeModelLM(config)
    if args.weights:
        from safetensors.torch import load_file
        index = os.path.join(args.weights, "model.safetensors.index.json")
        if os.path.exists(index):
            with open(index) as f:
                files = sorted(set(json.load(f)["weight_map"].values()))
        else:
            files = [p for p in os.listdir(args.weights) if p.endswith(".safetensors")]
        state = {}
        for p in files:
            state.update(load_file(os.path.join(args.weights, p)))
        model.load_state_dict(state)
    else:
        randomize_weights(model, args.seed)
    model = model.to(dtype).eval()

    H = config.hidden_size
    n_heads, n_kv = config.num_attention_heads, config.num_key_value_heads
    head_dim = config.head_dim
    BLK = args.block_length
    L = 3 * BLK  # three blocks

    gen = torch.Generator().manual_seed(args.seed + 1)

    def randn(*shape, scale=1.0):
        return (torch.randn(*shape, generator=gen) * scale).to(dtype)

    tensors = {}
    layer_moe = model.model.layers[1]
    layer_dense = model.model.layers[0]

    with torch.no_grad():
        # 2.2 RMSNorm
        tensors["rmsnorm.input"] = randn(2, 8, H)
        tensors["rmsnorm.output"] = layer_moe.input_layernorm(tensors["rmsnorm.input"])

        # 2.3a Partial RoPE: cos/sin table + rotation application
        pos = torch.arange(L)[None]
        cos, sin = model.model.rotary_emb(torch.zeros(1, L, H, dtype=dtype), pos)
        tensors["rope.position_ids"] = pos.to(torch.int64)
        tensors["rope.cos"] = cos
        tensors["rope.sin"] = sin
        tensors["rope.q_in"] = randn(1, n_heads, L, head_dim)
        tensors["rope.k_in"] = randn(1, n_kv, L, head_dim)
        q_rot, k_rot = model_mod.apply_rotary_pos_emb(
            tensors["rope.q_in"], tensors["rope.k_in"], cos, sin)
        tensors["rope.q_out"] = q_rot
        tensors["rope.k_out"] = k_rot

        # Block-diffusion masks: reference 0/1 expansion + strict additive conversion
        mask01, mask_add = block_masks(L, BLK, dtype)
        tensors["mask.reference_01"] = mask01
        tensors["mask.additive"] = mask_add

        # 2.3b Attention module (strict additive mask; fused QKV + qk-norm + partial RoPE + GQA)
        tensors["attn.input"] = randn(1, L, H, scale=0.5)
        attn_out = layer_moe.attention(
            hidden_states=tensors["attn.input"],
            attention_mask=mask_add,
            position_embeddings=(cos, sin),
        )[0]
        tensors["attn.output"] = attn_out

        # 2.4 Dense FFN (layer 0)
        tensors["ffn.input"] = randn(2, 8, H, scale=0.5)
        tensors["ffn.output"] = layer_dense.mlp(tensors["ffn.input"])

        # 2.5a Router / gate
        tensors["gate.input"] = randn(1, 16, H, scale=0.5)
        topk_idx, topk_weight, router_logits = layer_moe.mlp.gate(tensors["gate.input"])
        tensors["gate.topk_idx"] = topk_idx.to(torch.int64)
        tensors["gate.topk_weight"] = topk_weight
        tensors["gate.logits"] = router_logits

        # 2.5b Full MoE block (router + experts via reference moe_infer + shared expert)
        tensors["moe.input"] = randn(1, 16, H, scale=0.5)
        moe_out, _aux = layer_moe.mlp(tensors["moe.input"])
        tensors["moe.output"] = moe_out

        # Decoder layers (block composition: norm -> attn -> residual -> norm -> mlp -> residual)
        tensors["layer0.input"] = randn(1, L, H, scale=0.5)
        l0_out = layer_dense(
            tensors["layer0.input"], attention_mask=mask_add, position_ids=pos,
            position_embeddings=(cos, sin))[0]
        tensors["layer0.output"] = l0_out
        l1_out = layer_moe(
            l0_out, attention_mask=mask_add, position_ids=pos,
            position_embeddings=(cos, sin))[0]
        tensors["layer1.output"] = l1_out

        # M4 full-forward parity fixture: drive the WHOLE stack (embeddings -> layers ->
        # norm -> lm_head) over a mixed prompt+mask sequence, under a STRICT 0/-inf block
        # mask — the corrected reference baseline (phase-2 §6 / deviation 8), NOT stock
        # generate()'s 0/1 soft-bias mask. The inner model unconditionally rebuilds its mask
        # via create_bidirectional_mask(), so we monkeypatch that symbol to pass our
        # hand-built strict mask through verbatim; this is the "patch the reference mask
        # construction" option the guide sanctions.
        prompt_len = BLK + BLK // 4  # 20: spans block 0 fully + block 1 partially (§1 tail-share)
        mask_token = config.vocab_size - 5  # arbitrary in-vocab id for the masked region
        fwd_ids = torch.randint(0, config.vocab_size, (1, L), generator=gen)
        fwd_ids[:, prompt_len:] = mask_token
        fwd_ids = fwd_ids.to(torch.int64)
        fwd_pos = torch.arange(L)[None].to(torch.int64)

        original_cbm = model_mod.create_bidirectional_mask
        model_mod.create_bidirectional_mask = (
            lambda config=None, inputs_embeds=None, attention_mask=None, **_kw:
            attention_mask.to(inputs_embeds.dtype)
        )
        try:
            fwd_logits = model(
                input_ids=fwd_ids,
                attention_mask=mask_add,
                position_ids=fwd_pos,
                use_cache=False,
            ).logits.float()
        finally:
            model_mod.create_bidirectional_mask = original_cbm

        tensors["forward.input_ids"] = fwd_ids
        tensors["forward.position_ids"] = fwd_pos
        tensors["forward.mask"] = mask_add.clone()    # strict 0/-inf, for cross-check
        tensors["forward.logits"] = fwd_logits        # [1, L, vocab] FP32
        tensors["forward.argmax"] = fwd_logits.argmax(-1).to(torch.int64)  # [1, L]

        # 2.1 Embeddings (includes pad id: no special-casing on lookup)
        ids = torch.randint(0, config.vocab_size, (1, 24), generator=gen)
        ids[0, 3] = config.pad_token_id
        tensors["embed.ids"] = ids.to(torch.int64)
        tensors["embed.output"] = model.model.word_embeddings(ids)

        # 2.6 Output head (logits cast to FP32)
        tensors["lmhead.input"] = randn(1, 8, H, scale=0.5)
        tensors["lmhead.logits"] = model.lm_head(tensors["lmhead.input"]).float()

    os.makedirs(args.out, exist_ok=True)
    weights = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(weights, os.path.join(args.out, "weights.safetensors"))
    save_file({k: v.contiguous() for k, v in tensors.items()},
              os.path.join(args.out, "tensors.safetensors"))

    import transformers
    manifest = {
        "reference": REPO_ID,
        "seed": args.seed,
        "dtype": args.dtype,
        "block_length": BLK,
        "seq_length": L,
        "config": cfg_json,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "weights": "random" if not args.weights else args.weights,
        "notes": [
            "Module fixtures use a strict additive (0/-inf) block mask.",
            "Reference generate() passes a 0/1 mask that transformers applies additively "
            "(soft bias, not causality) — open item, phase-2 §6.",
            "forward.* tensors are the M4 full-forward parity fixture: the whole stack driven "
            "under a strict 0/-inf mask via a monkeypatched create_bidirectional_mask "
            "(phase-2 §6 / deviation 8), NOT stock generate() numerics.",
        ],
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(weights)} weight tensors and {len(tensors)} fixture tensors to {args.out}")


if __name__ == "__main__":
    main()
