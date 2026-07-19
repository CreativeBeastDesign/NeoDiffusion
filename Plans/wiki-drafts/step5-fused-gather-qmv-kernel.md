# Step 5 — Fused MoE gather-QMV kernel (documented negative)

**Status**: CLOSED 2026-07-19, **rejected for serving** on the pre-registered wall-clock TPS
gate; the kernel itself is landed, correct, tested, and default-off. Campaign record:
`Plans/step5-kernel-logbook.md`. Host: Mac Studio M2 Ultra (Mac14,14), mlx-swift 0.31.6 /
MLX core 0.31.1.

## What was built

A from-scratch inline-MSL reimplementation of MLX's `affine_gather_qmv_fast` (4-bit affine,
group 64) for the LLaDA2.1-mini MoE routed-expert SwiGLU, delivered as `MLXFast.metalKernel`
inline strings (route (b) — no mlx-swift fork), integrated at `SwitchGLU` level behind the
default-off `NEODIFFUSION_FUSED_QMV` flag with shape-eligibility guards and fallback to stock.
Final form: 2 dispatches instead of 3 (gate+up in one kernel), compile-time dims via header
`#define`s, 8 output rows per simdgroup, 8 values per thread (`qmv_rows<IN_DIM, ROWS, VPT>`).

## Results ladder — where the win appeared and where it vanished

| level | fused/stock | verdict |
|---|---|---|
| SwitchGLU microbench, Mini shapes | 0.79–0.83 | 17–21% faster |
| SwitchGLU microbench, Flash shapes | 0.90–0.91 | faster, ranking holds |
| **end-to-end ms/forward** (real model, 3 suites) | **1.0006** | **wash** |
| end-to-end TPS gate (≥15–20% required) | ≈0% | **FAIL → reject** |

Peak memory +0.37 GB on the fused arm (transient f32 casts of fp16 scales/biases).

## Key findings (each measured, provenance in the logbook)

1. **The Step 5a occupancy hypothesis inverted under iteration.** The profile said
   register-pressure-limited occupancy (100 regs, 32.8%). But the winning variants *raised*
   registers to 126 and *lowered* occupancy to 26.5% and were faster anyway; the TG-staging
   variant that actually targeted registers was slower. What paid at microbench level: fewer
   dispatches, compile-time dims (ALU limiter 70→56, Integer 57→43), wider per-thread ILP.
2. **The microbench win did not survive the busy queue.** In the real forward the latency the
   fused kernel removes is already hidden by queued work — the same non-additivity
   `gather_qmm_handoff.md` §11 found for the router. This bounds the ~7.5 ms
   "not-moving-bytes" sizing: it is a latency/scheduling phenomenon the pipeline absorbs, not
   recoverable ALU waste. **Microbench deltas on this codebase are upper bounds, not
   predictions.**
3. **Threshold decoding is trajectory-sensitive to ~1e-5-class numeric perturbation.** The
   kernel's FP-reorder noise (~2e-5 relative, far under 4-bit quantization error) deterministically
   changed decoding trajectories on 7/12 bench prompts (different step counts and token counts;
   coherent text both arms; post-steps/block *lower* on fused). Consequence: **any
   numerics-perturbing kernel change needs an end-to-end trajectory check, and naive TPS A/Bs
   are confounded unless trajectory-matched or measured as ms/forward.**
4. **x is cache-resident** (~256 KB/layer vs the 128 MiB weight stream) — "share/stage x"
   levers were aimed at a non-cost. Weight-stream traffic is the only bandwidth that matters
   in this kernel family.
5. **Methodology**: in-situ MoE-block captures remain the only working profiling route; trace
   bundles flakily finalize as 40-file torsos — verify ~84 files / ~1.2 GB and retry. A
   once-per-process activation echo (`[fused-qmv] active`) is required engagement proof for
   any flag-gated arm (env vars alone prove nothing).

## What stays in the tree

`Packages/DiffusionCore/Sources/MoEGatherQMVRunner.swift` (kernel + runner, default-off),
the SwitchGLU branch, 9 unit tests + 2 env-gated microbenches (Mini/Flash), the capture
harness improvements, and the traces under `scratch/captures/fused_v{0,2,3}*` (host-local).
Branch `kernel`, commits `9219fc2..b86e817`.

## Lessons for the wiki

- Pre-registered mechanisms can be refuted while their speed target is met at one level and
  lost at another — gates must name the *level* (microbench vs ms/forward vs TPS) or they
  don't bind.
- The three-level results ladder (op microbench → block → end-to-end) caught a sign error that
  any two-level version would have shipped.
- Negative results with this much instrumentation are cheap insurance: the next person who
  proposes "just rewrite the MoE kernel" starts from measured ground.
