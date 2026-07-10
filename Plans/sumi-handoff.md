# Sumi Handoff — the porting-and-optimising recipe for the next model

**Audience**: future Claude Code (or anyone) porting the next diffusion-LM into
NeoDiffusion on limited local hardware. This distils *process*; the Sumi-specific evidence
lives in [`sumi-logbook.md`](./sumi-logbook.md), the executed plan in
[`sumi-plan.md`](./sumi-plan.md). Where a step says "we learned", the logbook has the
receipts.

The recipe took Sumi from zero to parity-green + 10–16× end-to-end speedup in two
dev-machine days. Follow the phases in order; the ordering is load-bearing.

---

## Phase 0 — Pin the reference before writing any code

1. **Cache the reference verbatim** under `Tools/reference/<model>/`: `config.json`,
   `configuration_*.py`, `modeling_*.py`, `generation_*.py`, `generation_config.json`,
   `tokenizer.json` + `tokenizer_config.json`, `model.safetensors.index.json`. The
   reference *is* the parity target — HF pages change; your cache doesn't.
2. **Check its transformers requirement vs the system install.** If they diverge, make an
   isolated venv with `python3 -m venv --system-site-packages` (reuses the multi-GB torch)
   and `pip install -U transformers` inside it. Never upgrade the global install that
   serves earlier models' fixture scripts.
3. **Prove the reference runs at toy scale**: instantiate its config at tiny dims (hidden
   ~64, 2 layers, vocab ~200), seeded random weights, run `generate()` end-to-end. If this
   works, the whole local-parity strategy works, because everything downstream gates
   against dumps from exactly this setup.
4. **Extract the model-guide facts and take the design decisions NOW** (they are cheap
   before code): parity oracle for anything stochastic, API edge policies, numerics
   formulation choices, tokenizer identity. Record them in the plan; every later parity
   surprise gets triaged against this list.
5. **Read the reference line-by-line for load-bearing quirks** and write each into the plan
   as a checklist item: logits truncation/upcast, mask construction (LLaDA's 0/1-bias bug
   and Sumi's all-ones≡no-mask both hid here), special-token anchors, dead parameters,
   per-step CPU syncs you must not copy, absent-key config defaults living in the Python
   class rather than the JSON.

## Phase 1 — Toy-config parity ladder (all local, all exact)

Build one fixture dumper (`Tools/generate_<model>_fixtures.py`, patterned on
`generate_sumi_fixtures.py`): toy config, seeded weights, dumps per-module in/out pairs +
full-forward logits + `manifest.json` (config + seed + provenance notes). Swift tests are
**config-driven from the manifest** with an env-var fixture-dir override, so the identical
tests later consume real-weight BF16 fixtures on the Studio.

Climb strictly in this order — each rung's failures then have one suspect:

1. **Config** (absent-key semantics vs the Python class defaults) + **tokenizer**
   (100-case encode/decode parity; test *both* encode variants — post-processors that
   prepend BOS are found by test, not by docs).
2. **The model's novel primitive** (Sumi: off-by-one softmax) as a *twin*: verbatim
   reference formulation kept as oracle + the hot-path formulation, gated ≡ to 1e-6
   including adversarial inputs (outlier logits, etc.).
3. **Modules** (attention, MLP, norms, rope) vs fixtures at 1e-5 fp32.
4. **Full forward**: top-1 100% + max |Δprob| (expect float-epsilon at toy FP32; leave the
   bound at 1e-2 for BF16-real headroom).
5. **Samplers**: split every sampler into its deterministic core (gate exactly) and its
   stochastic draw (gate distributionally, or by algebraic equivalence under a *shared*
   RNG draw — that trick turns "stochastic" rewrites into exactly-testable ones).
6. **End-to-end loop**: dump per-step traces from the reference for a case matrix spanning
   every API flag (anchors on/off, window bounds, budget clamp, minimal prompt). Ship the
   reference's initial canvas in the fixture and add an `initialCanvas:`-style override —
   cross-RNG init is the one thing you cannot reproduce, so remove it from the equation
   and gate the rest token-for-token.

Reuse aggressively: check whether existing modules (RMSNorm, RoPE, SwiGLU) are already
parameterised for the new model before writing variants. Sumi needed zero refactor —
verify, don't assume, in either direction.

## Phase 2 — Real weights

1. Verify the converted artefact **bit-perfect** against the original in python-MLX
   (`mx.quantize(orig) ≡ artefact bytes`, dequant within expected noise). Ten minutes here
   buys the right to say "not a conversion bug" during every later investigation — we
   cashed that cheque repeatedly.
2. Expect format quirks: safetensors `U8 [out, in/2]` vs MLX's `U32 [out, in·bits/32]`
   packed layout → `view(dtype:)` reinterpret, validated by the step-1 ground truth.
3. Real-weight tests are **env-gated opt-ins** (minutes of wall-clock); the default suite
   stays seconds.
4. First generations WILL look wrong. Before concluding anything, write down every
   variable that differs from the reference's happy path (canvas length, sampler, steps,
   prompt style) and change **one at a time**. Our "canvas-locked" false conclusion came
   from a probe + a confounded comparison; the de-confounded single-axis experiment
   overturned it in one run. Full worked example: logbook §2.
5. Distrust proxy metrics twice as much as generations: a reconstruction/self-agreement
   probe can be off-distribution for the model even when generation is fine. Validate any
   probe against a known-good operating point before using it to map a parameter space.

## Phase 3 — Baseline, then levers

1. **Freeze a bench baseline before optimising anything.** Extend `diffusion-bench` with a
   named-arm suite for the model: JSONL per arm-run, per-step stats (mean/std/p95/first),
   peak memory, cross-run variance gate (<5%), `--cooldown` between runs. Laptop thermals
   will fail the gate on >10-minute arms no matter what — compare within-run steady
   medians there and note that desktops must re-check.
2. **Every optimisation is a named arm vs the frozen baseline.** No arm, no claim.
3. **Measure cost attribution, never FLOP-derive it.** FLOP intuition failed twice on Sumi
   (attention "1%" was 15% of wall-clock; "compute-saturated" hid 0.8 s/step of dispatch
   overhead). Micro-bench the component in isolation at real shapes, or `xctrace` it.
4. **Check what the framework already ships before building anything.** MLX had grown
   native SDPA `sinks:` support that equalled our best fusion *and* the planned Metal
   kernel, for one parameter. Grep the checked-out framework source for the primitive you
   are about to hand-build (`.build/checkouts/mlx-swift/...`).
5. Lever order that paid, most→least: **step count** (early exit / fewer scheduled steps),
   **tokens-per-step**, **sampler tensor traffic** (structural removal — compute
   confidences from max/LSE/gather instead of materialising softmax; prefer removing
   `[S,V]` intermediates over downcasting them, which blurs decision margins),
   **`asyncEval` pipelining**, **head/embedding quantization** (post-load
   `MLXNN.quantize` filter — no artefact change needed). Kernel fusion paid **zero** —
   see the Metal verdict (plan §6) before considering it for the next model, and require
   the phase-1 §13.1 trigger to be *measured* open first.
6. **Metric design for quality gates**: condition every "X% changed" metric on whether the
   change can matter — and condition on the **decision margin** (top-1/top-2 gap), not on a
   confidence threshold: a p_max gate pools razor-margin flips (benign) with wide-margin
   ones (damage) and misses ambiguity below its cutoffs. Calibrate against the *measured*
   perturbation scale ("largest margin that flipped" vs p99 margin-perturbation — they
   matched at 0.089 for Sumi's head quant, the signature of benign noise), and test the
   downstream decision (top-k selection, committed tokens) directly on real mid-generation
   inputs. This lesson took two iterations to learn (93% pooled false alarm, then a
   p_max gate with the same flaw one level down) — logbook §3.
7. Record negative results in the plan with the same prominence as wins (projection fusion
   dead, greedy+scoped-window degenerate, exact-stability exit never fires). They are the
   next port's shortcuts.

## Phase 4 — Model-behaviour adaptations (SchED-style example)

Paper techniques rarely drop in unchanged. The early-exit adaptation took three measured
iterations — expect the same shape: (1) the naive signal (whole-window, exact stability)
never fires; (2) diagnose *from arm data* which positions/regions block it; (3) scope the
signal (content window) and relax it (fractional stability). Ship the knobs
(`stableSteps/minSteps/stableFraction`) rather than the tuned constants — the next model
will need different values. Keep every behavioural deviation from the reference **off by
default** and listed in the plan's deviations table, so parity gates keep meaning something.

## Operational checklist (each cost us real time once)

- [ ] SwiftPM release builds: copy `mlx-swift_Cmlx.bundle` next to the executable, or the
      binary dies on "Failed to load the default metallib".
- [ ] `setvbuf(stdout, nil, _IOLBF, 0)` in any long-running CLI whose output you'll tail.
- [ ] Background bench runs: line-buffered logs + a Monitor with failure patterns in the
      grep, not just success lines.
- [ ] Big-model forwards on a small machine: python-MLX with per-layer `mx.load`/release
      streams a 15.5 GB BF16 file through 16 GB RAM fine — useful as an unquantized oracle.
- [ ] `swift test` env-gates: `NEODIFFUSION_<MODEL>_REAL=1` for real weights,
      `..._ABBENCH=1` for micro-benches, fixture-dir overrides for Studio reruns.
- [ ] Keep a per-campaign logbook as you go (this repo: `sumi-logbook.md`) — write the
      method and the *why* at the moment of testing; reconstructing it later loses the
      failed branches, and the failed branches are half the value.

## What transfers vs what is Sumi-specific

**Transfers as-is**: the whole parity-ladder method; fixture dumper pattern; twin-oracle
pattern for novel primitives; shared-draw equivalence testing; bench-arm discipline;
bit-perfect conversion verification; the operational checklist; the metric-design lessons.

**Sumi-specific (re-derive per model)**: canvas/window semantics and their quality
envelope; which sampler is the quality anchor; the early-exit signal scoping and threshold;
head-quantization safety (Sumi's head tolerated 4-bit with zero confident flips — a model
with tighter confidence margins may not); every number in the logbook.
