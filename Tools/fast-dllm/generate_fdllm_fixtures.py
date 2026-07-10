#!/usr/bin/env python3
"""Fast-DLLM core fixture generator.

Runs the pinned reference implementation (cached at models/fast-dllm-1-5B) on a toy config
with seeded random weights and dumps per-module inputs/outputs plus a full-forward logits
dump for the Swift fixture-diff tests:

  <out>/weights.safetensors   full model state_dict (HF key layout)
  <out>/tensors.safetensors   per-module input/output pairs + forward.* dump
  <out>/manifest.json         config + seed + provenance

Requires transformers >= 5.8 (the reference targets 5.8.1); run with the Sumi venv:
  scratch/sumi-venv/bin/python Tools/generate_sumi_fixtures.py

Studio BF16 real-weight run (deferred gate, sumi-plan.md §7):
  scratch/sumi-venv/bin/python Tools/generate_sumi_fixtures.py \
      --config <hf_dir>/config.json --weights <hf_dir> --dtype bfloat16 --out <dir>
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

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "sumi")
REFERENCE_FILES = ["configuration.py", "modeling.py"]

TOY_CONFIG = {
    "vocab_size": 200,
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 6,
    "num_key_value_heads": 1,  # GQA 12:2, exercises repeat_kv
    "head_dim": 16,
    "max_position_embeddings": 256,
    "rms_norm_eps": 1e-5,
    "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
    "bos_token_id": 196,
    "eos_token_id": 197,
    "pad_token_id": 198,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "add_qkv_bias": False,
    "mlp_bias": False,
}
SEQ_LEN = 24


def import_reference_package():
    """Import the cached reference .py files as a package (they use relative imports)."""
    pkg_dir = tempfile.mkdtemp(prefix="sumi_ref_")
    pkg_name = "sumi_reference"
    dest = os.path.join(pkg_dir, pkg_name)
    os.makedirs(dest)
    open(os.path.join(dest, "__init__.py"), "w").close()
    for fname in REFERENCE_FILES:
        shutil.copy(os.path.join(REFERENCE_DIR, fname), os.path.join(dest, fname))
    sys.path.insert(0, pkg_dir)
    cfg_mod = importlib.import_module(f"{pkg_name}.configuration_sumi")
    model_mod = importlib.import_module(f"{pkg_name}.modeling_sumi")
    gen_mod = importlib.import_module(f"{pkg_name}.generation_sumi")
    return cfg_mod, model_mod, gen_mod


def randomize_weights(model, seed):
    """Seeded random weights: norms near 1, everything else N(0, 0.05)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, param in model.state_dict().items():
            if "norm" in name.lower() and name.endswith(".weight"):
                new = 1.0 + (torch.rand(param.shape, generator=gen) - 0.5) * 0.5
            else:
                new = torch.randn(param.shape, generator=gen) * 0.05
            param.copy_(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="config.json path; toy config if omitted")
    ap.add_argument("--weights", default=None,
                    help="Optional dir with real safetensors weights; random weights if omitted")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default="scratch/sumi_core_fixtures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq-len", type=int, default=SEQ_LEN)
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    cfg_mod, model_mod, _ = import_reference_package()

    if args.config:
        with open(args.config) as f:
            cfg_json = json.load(f)
        for key in ("architectures", "auto_map", "model_type", "transformers_version", "dtype"):
            cfg_json.pop(key, None)
    else:
        cfg_json = dict(TOY_CONFIG)
    config = cfg_mod.SumiConfig(**cfg_json, attn_implementation="eager")

    torch.manual_seed(args.seed)
    model = model_mod.SumiForMaskGeneration(config)
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

    gen = torch.Generator().manual_seed(args.seed + 1)
    B, S = 1, args.seq_len
    H = config.hidden_size
    tensors = {}

    with torch.no_grad():
        # --- softmax_one oracle pair (includes an outlier row, |logit| ~ 40, no softcap) ---
        logits = torch.randn(2, 3, 8, 16, generator=gen) * 4.0
        logits[0, 0, 0, 0] = 40.0
        logits[1, 2, 7, 5] = -35.0
        tensors["softmax_one.input"] = logits
        tensors["softmax_one.output"] = model_mod.softmax_one(logits, dim=-1, dtype=torch.float32)

        # --- RoPE (full rotary) ---
        position_ids = torch.arange(S).unsqueeze(0)
        dummy = torch.zeros(B, S, H, dtype=dtype)
        cos, sin = model.model.rotary_emb(dummy, position_ids)
        tensors["rope.position_ids"] = position_ids.to(torch.int32)
        tensors["rope.cos"] = cos
        tensors["rope.sin"] = sin
        q_in = torch.randn(B, config.num_attention_heads, S, config.head_dim, generator=gen).to(dtype)
        k_in = torch.randn(B, config.num_key_value_heads, S, config.head_dim, generator=gen).to(dtype)
        q_out, k_out = model_mod.apply_rotary_pos_emb(q_in, k_in, cos, sin)
        tensors["rope.q_in"], tensors["rope.k_in"] = q_in, k_in
        tensors["rope.q_out"], tensors["rope.k_out"] = q_out, k_out

        # --- RMSNorm (eps from config) ---
        norm_in = torch.randn(B, S, H, generator=gen).to(dtype) * 2.0
        tensors["rmsnorm.input"] = norm_in
        tensors["rmsnorm.output"] = model.model.layers[0].input_layernorm(norm_in)

        # --- Attention module (layer 0), bidirectional, no mask ---
        attn_in = torch.randn(B, S, H, generator=gen).to(dtype)
        attn_out, _ = model.model.layers[0].self_attn(
            attn_in, position_embeddings=(cos, sin), attention_mask=None)
        tensors["attn.input"] = attn_in
        tensors["attn.output"] = attn_out

        # --- MLP (layer 0) ---
        mlp_in = torch.randn(B, S, H, generator=gen).to(dtype)
        tensors["mlp.input"] = mlp_in
        tensors["mlp.output"] = model.model.layers[0].mlp(mlp_in)

        # --- Decoder layer composition (both layers, chained) ---
        layer_in = torch.randn(B, S, H, generator=gen).to(dtype)
        tensors["layer0.input"] = layer_in
        layer0_out = model.model.layers[0](
            layer_in, attention_mask=None, position_embeddings=(cos, sin))
        tensors["layer0.output"] = layer0_out
        tensors["layer1.output"] = model.model.layers[1](
            layer0_out, attention_mask=None, position_embeddings=(cos, sin))

        # --- Embeddings + lm_head ---
        ids = torch.randint(0, config.vocab_size, (B, S), generator=gen)
        tensors["embed.ids"] = ids.to(torch.int32)
        tensors["embed.output"] = model.model.embed_tokens(ids)
        lmhead_in = torch.randn(B, S, H, generator=gen).to(dtype)
        tensors["lmhead.input"] = lmhead_in
        tensors["lmhead.logits"] = model.lm_head(lmhead_in)[..., : config.vocab_size].float()

        # --- Full forward (S1.3 gate): _compute_logits semantics ---
        # generation feeds an all-ones 2D mask, which _prepare_attention_mask expands to an
        # all-zero additive mask — assert it is numerically identical to attention_mask=None,
        # so the Swift engine can pass mask=nil.
        fwd_ids = torch.randint(0, config.vocab_size, (B, S), generator=gen)
        logits_none = model(input_ids=fwd_ids, attention_mask=None, use_cache=False).logits
        logits_ones = model(
            input_ids=fwd_ids, attention_mask=torch.ones(B, S, dtype=torch.long),
            use_cache=False).logits
        assert torch.equal(logits_none, logits_ones), "ones-mask must equal no-mask"
        tensors["forward.input_ids"] = fwd_ids.to(torch.int32)
        tensors["forward.logits"] = logits_none[..., : config.vocab_size].float()

        # --- Sampler step functions (S1.4 gate): deterministic parts, isolated from the model ---
        gen_pkg = sys.modules["sumi_reference.generation_sumi"]
        z = torch.randint(0, config.vocab_size, (B, S), generator=gen)
        step_logits = (torch.randn(B, S, config.vocab_size, generator=gen) * 3.0).float()
        noise_mask = torch.ones(B, S, dtype=torch.bool)
        noise_mask[:, :4] = False        # frozen prompt
        noise_mask[:, S - 2] = False     # frozen anchor mid-tail
        tensors["sampler.z"] = z.to(torch.int32)
        tensors["sampler.logits"] = step_logits
        tensors["sampler.noise_mask"] = noise_mask

        tensors["sampler.greedy_out"] = gen_pkg._greedy_step(z, step_logits, noise_mask).to(torch.int32)

        for k in (1, 3):
            z_out, pos = gen_pkg._adaptive_step(
                z, step_logits, noise_mask, tokens_per_step=k, temperature=0.0, generator=None)
            tensors[f"sampler.adaptive_out_k{k}"] = z_out.to(torch.int32)
            tensors[f"sampler.adaptive_pos_k{k}"] = pos.to(torch.int32)

        # Ancestral: gate the analytic posterior exactly (the multinomial draw is stochastic
        # and gated distributionally on the Swift side). Mirror _ancestral_step up to sampling.
        log_snr_t, log_snr_s = -1.5, 0.5
        x_hat = torch.softmax(step_logits, dim=-1)
        alpha_t = torch.sigmoid(torch.tensor(log_snr_t))
        alpha_s = torch.sigmoid(torch.tensor(log_snr_s))
        alpha_t_s = alpha_t / alpha_s.clamp(min=1e-12)
        u_t = (1.0 - alpha_t) / config.vocab_size
        u_s = (1.0 - alpha_s) / config.vocab_size
        u_t_s = (1.0 - alpha_t_s).clamp(min=0.0) / config.vocab_size
        q_s = alpha_s * x_hat + u_s
        one_hot_zt = torch.nn.functional.one_hot(z, num_classes=config.vocab_size).float()
        q_t_given_s = alpha_t_s * one_hot_zt + u_t_s
        x_hat_at_zt = x_hat.gather(-1, z.unsqueeze(-1)).squeeze(-1)
        q_t_at_zt = (alpha_t * x_hat_at_zt + u_t).clamp(min=1e-12)
        posterior = (q_s * q_t_given_s / q_t_at_zt.unsqueeze(-1)).clamp(min=0.0)
        tensors["sampler.ancestral_x_hat"] = x_hat
        tensors["sampler.ancestral_posterior"] = posterior

        # --- Log-SNR schedules (S1.4 gate) ---
        for kind in ("linear", "cosine"):
            for steps in (8, 128):
                tensors[f"schedule.{kind}_{steps}"] = gen_pkg._make_log_snr_schedule(
                    steps, kind, -9.0, 9.0, torch.device("cpu"))

    os.makedirs(args.out, exist_ok=True)
    state = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(state, os.path.join(args.out, "weights.safetensors"))
    save_file({k: v.contiguous() for k, v in tensors.items()},
              os.path.join(args.out, "tensors.safetensors"))
    manifest = {
        "model": "tohoku-nlp/sumi-7b (reference code, cached Tools/reference/sumi)",
        "config": cfg_json,
        "seed": args.seed,
        "seq_len": S,
        "dtype": args.dtype,
        "ancestral_log_snr_t": -1.5,
        "ancestral_log_snr_s": 0.5,
        "notes": [
            "attention is bidirectional, no mask (attention_mask=None == all-ones 2D mask, asserted)",
            "softmax_one fixture includes outlier logits (+40 / -35) for sink-underflow coverage",
            "forward.logits are truncated to vocab_size and upcast fp32 (_compute_logits semantics)",
        ],
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Wrote fixtures to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
