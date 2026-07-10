#!/usr/bin/env python3
"""M5 fixture generator: dumps reference `generate()` traces under the **corrected**
(strict 0/-inf) block mask for the Swift denoising-loop parity tests (phase-2 §4 M5(a)).

The reference `generate()` in modeling_llada2_moe.py bakes in a 0/1 soft-bias mask
(phase-2 §6 / deviation 8) that is NOT block-causal. NeoDiffusion ships `.strict`
semantics, so parity is measured against a corrected baseline, never stock `generate()`.
We therefore re-run the *exact* reference denoising algorithm here with two corrections
only:

  1. the attention mask is a strict 0/-inf block-causal mask (via a monkeypatched
     `create_bidirectional_mask`, the same technique `generate_core_fixtures.py` uses
     for the M4 full-forward dump), and
  2. the loop body is transcribed verbatim from `generate()` so we can instrument it
     (per-step Γ/Δ sets, per-block commits). The transcription is asserted line-faithful
     by construction — only the mask differs from the reference.

Emits, for a suite of prompts across S/Q modes and eos_early_stop on/off:

  <out>/weights.safetensors   full model state_dict (HF key layout) — self-contained
  <out>/traces.json           list of cases: prompt, params, per-block committed x,
                              final padded x, trimmed output, per-step Γ/Δ diagnostics
  <out>/manifest.json         config + seeds + provenance

Usage (toy config, dev machine — runs the reference on seeded random weights):
  python3 Tools/generate_loop_fixtures.py

Studio BF16 real-weight run (M5 acceptance, second half):
  python3 Tools/generate_loop_fixtures.py --config <hf_dir>/config.json \
      --weights <hf_dir> --dtype bfloat16 --out scratch/loop_fixtures_bf16
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
    """Seeded random weights: norms near 1, expert bias small, everything else N(0, 0.05).
    Identical convention to generate_core_fixtures.py so the two dumps agree at equal seed."""
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


def strict_additive_block_mask(total_len, block_len, dtype, device):
    """Strict 0/-inf block-causal mask [1, 1, total, total]: position i may attend j iff
    block(j) <= block(i). This is what NeoDiffusion's BlockDiffusionMask.build(.strict) emits."""
    nb = total_len // block_len
    tril = torch.tril(torch.ones(nb, nb, device=device))
    mask01 = tril.repeat_interleave(block_len, 0).repeat_interleave(block_len, 1)
    additive = torch.where(mask01.bool(), 0.0, float("-inf"))
    return additive[None, None].to(dtype)


def sample_argmax(logits):
    """temp-0 sampling: argmax token + its softmax probability (reference
    _sample_with_temperature_topk_topp at temperature=0)."""
    token = torch.argmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    token_prob = torch.gather(probs, -1, token.unsqueeze(-1)).squeeze(-1)
    return token, token_prob


def run_generate_strict(model, model_mod, input_ids, params, dtype, device, record_steps):
    """Verbatim transcription of reference `generate()` (temperature-0 path) under a
    STRICT 0/-inf block mask. Returns (trimmed_output, block_commits, final_x, per_block_steps,
    step_trace). Only the mask semantics differ from the reference — everything else is line-faithful.
    """
    block_length = params["block_length"]
    gen_length = params["gen_length"]
    eos_early_stop = params["eos_early_stop"]
    threshold = params["threshold"]
    editing_threshold = params["editing_threshold"]
    max_post_steps = params["max_post_steps"]
    num_to_transfer = params["num_to_transfer"]
    eos_id = params["eos_id"]
    mask_id = params["mask_id"]

    prompt_length = input_ids.shape[1]
    num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
    total_length = num_blocks * block_length

    strict_mask = strict_additive_block_mask(total_length, block_length, dtype, device)
    position_ids = torch.arange(total_length, device=device).unsqueeze(0)
    x = torch.full((1, total_length), mask_id, dtype=torch.long, device=device)
    x[:, :prompt_length] = input_ids.clone()

    prefill_blocks = prompt_length // block_length

    # Monkeypatch the inner model's mask builder to pass our strict 4D mask through verbatim
    # (the reference inner model unconditionally rebuilds a mask otherwise — see
    # generate_core_fixtures.py forward dump).
    original_cbm = model_mod.create_bidirectional_mask
    model_mod.create_bidirectional_mask = (
        lambda config=None, inputs_embeds=None, attention_mask=None, **_kw:
        attention_mask.to(inputs_embeds.dtype)
    )

    block_commits = []
    per_block_steps = []
    step_trace = []
    try:
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = strict_mask[:, :, :current_window_end, :current_window_end]
            cur_position_ids = position_ids[:, :current_window_end]
            block_start_pos = num_block * block_length

            post_steps = 0
            steps_taken = 0
            while True:
                old_block_tokens = cur_x[:, -block_length:].clone()
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if torch.any(active_block_mask) == False:
                    post_steps += 1
                if post_steps > max_post_steps:
                    break

                prompt_mask_in_block = torch.zeros(block_length, dtype=torch.bool, device=device)
                if block_start_pos < prompt_length:
                    prompt_end_in_block = min(prompt_length - block_start_pos, block_length)
                    prompt_mask_in_block[:prompt_end_in_block] = True

                logits = model(
                    input_ids=cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                    use_cache=False,
                ).logits
                active_logits = logits[:, -block_length:, :]
                x0, x0_p = sample_argmax(active_logits)

                mask_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                if active_block_mask.sum() > 0:
                    mask_confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                    high_conf_mask = (mask_confidence[0] > threshold) & active_block_mask[0]
                    num_high_confidence = high_conf_mask.sum().item()
                    if num_high_confidence >= num_to_transfer:
                        mask_transfer_index[0] = high_conf_mask
                    else:
                        num_available = active_block_mask.sum().item()
                        if num_available > 0:
                            _, idx = torch.topk(
                                mask_confidence[0], k=min(num_to_transfer, num_available))
                            mask_transfer_index[0, idx] = True

                editing_transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                non_mask_positions = ~active_block_mask
                non_prompt_positions = ~prompt_mask_in_block
                editable_positions = non_mask_positions & non_prompt_positions[None, :]
                editing_confidence = torch.where(editable_positions, x0_p, -torch.inf)
                high_conf_editing = (editing_confidence[0] > editing_threshold) & editable_positions[0]
                token_changed = x0[0] != old_block_tokens[0]
                editing_transfer_index[0] = high_conf_editing & token_changed
                final_transfer_index = mask_transfer_index | editing_transfer_index

                if record_steps:
                    step_trace.append({
                        "block": num_block,
                        "step": steps_taken,
                        "gamma": mask_transfer_index[0].nonzero(as_tuple=True)[0].tolist(),
                        "delta": editing_transfer_index[0].nonzero(as_tuple=True)[0].tolist(),
                        "x0": x0[0].tolist(),
                    })

                if final_transfer_index.any():
                    cur_x[:, -block_length:][final_transfer_index] = x0[final_transfer_index]

                steps_taken += 1
                if active_block_mask.sum() == 0 and not editing_transfer_index.any():
                    break

            x[:, :current_window_end] = cur_x
            block_commits.append(x[0, :current_window_end].tolist())
            per_block_steps.append(steps_taken)

            if eos_early_stop:
                generated_part = x[0, prompt_length:current_window_end]
                if (generated_part == mask_id).sum() == 0:
                    eos_positions = (generated_part == eos_id).nonzero(as_tuple=True)[0]
                    if len(eos_positions) > 0:
                        break
    finally:
        model_mod.create_bidirectional_mask = original_cbm

    # Trim (reference tail): first eos in the generated part, inclusive; else gen_length.
    generated_answer = x[:, : prompt_length + gen_length]
    eos_pos = (generated_answer[0][prompt_length:] == eos_id).nonzero(as_tuple=True)[0]
    first = eos_pos[0].item() if len(eos_pos) > 0 else gen_length
    output = generated_answer[0, prompt_length: prompt_length + first + 1].tolist()

    return {
        "output": output,
        "block_commits": block_commits,
        "final_x": x[0].tolist(),
        "per_block_steps": per_block_steps,
        "step_trace": step_trace,
        "prompt_length": prompt_length,
        "total_length": total_length,
        "num_blocks": num_blocks,
        "prefill_blocks": prefill_blocks,
    }


# S/Q served modes (model card): Q τ=0.7/0.5, S τ=0.5/0.0. max_post_steps=16, num_to_transfer=1.
MODE_PRESETS = {
    "q": {"threshold": 0.7, "editing_threshold": 0.5},
    "s": {"threshold": 0.5, "editing_threshold": 0.0},
}


def build_cases(vocab_size, block_length, eos_id, mask_id):
    """Prompt suite exercising: aligned vs tail-share prompts, prefill_blocks 0/1,
    both modes, eos_early_stop on/off. Prompts are seeded random token ids (< a safe
    id ceiling so we never accidentally emit mask_id/eos_id in the prompt)."""
    gen = torch.Generator().manual_seed(1234)
    safe_ceiling = min(vocab_size, min(eos_id, mask_id))
    prompt_lengths = [8, 16, 20, 33]  # short, aligned, tail-share, >2 blocks
    cases = []
    for plen in prompt_lengths:
        prompt = torch.randint(0, safe_ceiling, (1, plen), generator=gen).tolist()[0]
        for mode in ("q", "s"):
            for eos_stop in (False, True):
                cases.append({
                    "name": f"p{plen}_{mode}_{'eos' if eos_stop else 'noeos'}",
                    "mode": mode,
                    "prompt": prompt,
                    "params": {
                        "threshold": MODE_PRESETS[mode]["threshold"],
                        "editing_threshold": MODE_PRESETS[mode]["editing_threshold"],
                        "max_post_steps": 16,
                        "num_to_transfer": 1,
                        "eos_early_stop": eos_stop,
                        "temperature": 0.0,
                        "block_length": block_length,
                        "gen_length": 3 * block_length,
                        "mask_id": mask_id,
                        "eos_id": eos_id,
                    },
                })
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="scratch/dummy_hf/config.json")
    ap.add_argument("--weights", default=None,
                    help="Optional dir with real safetensors weights; random weights if omitted")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default="scratch/loop_fixtures")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--block-length", type=int, default=16)
    args = ap.parse_args()

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    device = torch.device("cpu")
    cfg_mod, model_mod = import_reference_package()

    with open(args.config) as f:
        cfg_json = json.load(f)
    cfg_json.pop("architectures", None)
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

    # Reference special ids default: mask_id=156895, eos_id=156892. Toy vocab is tiny, so
    # remap them into range while keeping eos != mask and both out of the prompt id range.
    vocab_size = config.vocab_size
    if vocab_size > 156895:
        mask_id, eos_id = 156895, 156892
    else:
        mask_id = vocab_size - 1
        eos_id = vocab_size - 2

    cases = build_cases(vocab_size, args.block_length, eos_id, mask_id)

    results = []
    with torch.no_grad():
        for i, case in enumerate(cases):
            input_ids = torch.tensor([case["prompt"]], dtype=torch.long, device=device)
            # Record per-step Γ/Δ for the first case of each mode only (keeps traces.json small).
            record = case["name"].endswith("_q_noeos") or case["name"].endswith("_s_noeos")
            trace = run_generate_strict(
                model, model_mod, input_ids, case["params"], dtype, device, record_steps=record)
            results.append({**case, **trace})
            print(f"[{i+1}/{len(cases)}] {case['name']}: "
                  f"blocks {trace['prefill_blocks']}..{trace['num_blocks']}, "
                  f"steps {trace['per_block_steps']}, out_len {len(trace['output'])}")

    os.makedirs(args.out, exist_ok=True)
    weights = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(weights, os.path.join(args.out, "weights.safetensors"))
    with open(os.path.join(args.out, "traces.json"), "w") as f:
        json.dump({"mask_id": mask_id, "eos_id": eos_id, "cases": results}, f)

    import transformers
    manifest = {
        "reference": REPO_ID,
        "seed": args.seed,
        "dtype": args.dtype,
        "block_length": args.block_length,
        "config": cfg_json,
        "mask_id": mask_id,
        "eos_id": eos_id,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "weights": "random" if not args.weights else args.weights,
        "notes": [
            "generate() traces under a STRICT 0/-inf block mask (monkeypatched "
            "create_bidirectional_mask), the corrected reference baseline — NOT stock "
            "generate() 0/1 soft-bias numerics (phase-2 §6 / deviation 8).",
            "Loop body transcribed verbatim from reference generate(); only the mask differs.",
        ],
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {len(weights)} weight tensors and {len(results)} generate traces to {args.out}")


if __name__ == "__main__":
    main()
