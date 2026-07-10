#!/usr/bin/env python3
"""Sumi loop-trace fixture generator (sumi-plan.md §3 S1.5).

Transcribes reference `SumiGenerationMixin.generate` end-to-end on the toy model for a case
matrix of deterministic samplers (greedy, adaptive at temperature 0) and dumps, per case:

  case{i}.canvas_init   [1, T]  canvas after prompt+randint concat, BEFORE anchor injection
  case{i}.step{s}       [1, T]  canvas after denoising step s
  case{i}.canvas_final  [1, T]  final untrimmed canvas
  case{i}.sequences     [L]     EOS-trimmed output (prompt + generation up to first EOS)

torch.randint cannot be reproduced across RNGs, so the Swift engine consumes `canvas_init`
via its `initialCanvas` override; everything downstream is deterministic and gated
token-for-token. The stochastic ancestral sampler is gated at step level (posterior) and
distributionally — see SumiSamplerTests.

Run with the Sumi venv:
  scratch/sumi-venv/bin/python Tools/generate_sumi_loop_fixtures.py
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file

# Reuse the reference-import and toy-config machinery from the core fixture generator.
from generate_sumi_fixtures import TOY_CONFIG, import_reference_package, randomize_weights

CANVAS = 32
STEPS = 6

# name, prompt_len, sampler, tokens_per_step, anchor_eosbos, denoise_end, max_new_tokens
CASES = [
    ("greedy_anchored",        3, "greedy",   1, True,  None, 20),
    ("greedy_unanchored",      3, "greedy",   1, False, None, 20),
    ("greedy_bos_denoise_end", 1, "greedy",   1, True,  20,   16),
    ("greedy_budget_clamp",    8, "greedy",   1, True,  None, 100),
    ("adaptive_k1",            5, "adaptive", 1, True,  None, 20),
    ("adaptive_k3",            5, "adaptive", 3, True,  None, 20),
    ("adaptive_unanchored",    4, "adaptive", 2, False, 24,   20),
    ("greedy_long_prompt",    12, "greedy",   1, True,  None, 10),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="scratch/sumi_loop_fixtures")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg_mod, model_mod, _ = import_reference_package()
    config = cfg_mod.SumiConfig(**TOY_CONFIG, attn_implementation="eager")

    torch.manual_seed(args.seed)
    model = model_mod.SumiForMaskGeneration(config).eval()
    randomize_weights(model, args.seed)

    prompt_gen = torch.Generator().manual_seed(args.seed + 100)
    tensors = {}
    manifest_cases = []

    for idx, (name, plen, sampler, tps, anchor, dend, mnt) in enumerate(CASES):
        prompt = torch.randint(0, config.vocab_size, (1, plen), generator=prompt_gen)
        case_seed = args.seed + 1000 + idx

        # Reconstruct canvas_init: randint is the FIRST consumption of the seeded generator
        # inside generate(), so a fresh generator with the same seed reproduces it exactly.
        g = torch.Generator().manual_seed(case_seed)
        completion = torch.randint(
            0, config.vocab_size, (1, CANVAS - plen), dtype=torch.long, generator=g)
        canvas_init = torch.cat([prompt, completion], dim=-1)
        tensors[f"case{idx}.canvas_init"] = canvas_init.to(torch.int32)

        steps_seen = []

        def callback(info, steps_seen=steps_seen):
            steps_seen.append(info["z"].clone())

        out = model.generate(
            input_ids=prompt,
            max_new_tokens=mnt,
            canvas_length=CANVAS,
            num_denoising_steps=STEPS,
            sampler=sampler,
            temperature=0.0,
            tokens_per_step=tps,
            anchor_eosbos=anchor,
            denoise_end=[dend] if dend is not None else None,
            trim_at_eos=True,
            seed=case_seed,
            progress_callback=callback,
        )

        assert len(steps_seen) == STEPS
        for s, z in enumerate(steps_seen):
            tensors[f"case{idx}.step{s}"] = z.to(torch.int32)
        tensors[f"case{idx}.canvas_final"] = out.canvas.to(torch.int32)
        tensors[f"case{idx}.sequences"] = out.sequences[0].to(torch.int32)

        manifest_cases.append({
            "name": name,
            "prompt_ids": prompt[0].tolist(),
            "sampler": sampler,
            "tokens_per_step": tps,
            "anchor_eosbos": anchor,
            "denoise_end": dend,
            "max_new_tokens": mnt,
        })

    os.makedirs(args.out, exist_ok=True)
    save_file({k: v.contiguous() for k, v in tensors.items()},
              os.path.join(args.out, "tensors.safetensors"))
    save_file({k: v.contiguous() for k, v in model.state_dict().items()},
              os.path.join(args.out, "weights.safetensors"))
    manifest = {
        "config": dict(TOY_CONFIG),
        "canvas_length": CANVAS,
        "num_denoising_steps": STEPS,
        "seed": args.seed,
        "cases": manifest_cases,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"Wrote {len(CASES)} loop-trace cases to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
