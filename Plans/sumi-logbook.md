# Sumi-7B Port & Optimisation Logbook

**Period**: 2026-07-08 → 2026-07-09 (single dev-machine campaign, MacBook Pro M1 16 GB)
**Purpose**: evidentiary record of what was built, what was tested, *how* it was tested and
why that way, every measurement, and every negative result. Written to serve as (a) proof
of the results claimed in `sumi-plan.md` and (b) a worked reference for repeating the
process on another model (the distilled recipe lives in `sumi-handoff.md`).
**Provenance**: every number below was measured on this machine; raw arm-level records are
in `scratch/sumi_bench.jsonl` and the `scratch/sumi_*.log` files. Test names are cited so
each claim can be re-run.

---

## 0. Method in one paragraph

The campaign follows the phase-2 house discipline: **pin the reference algorithm, gate
every layer of the port against reference-dumped fixtures before touching the next layer,
never optimise before a frozen baseline, A/B every optimisation against that baseline, and
record negative results with the same care as wins.** Correctness always had an *oracle*
(the verbatim reference implementation, or a verbatim-formulation twin kept in the code);
performance always had an *arm* (a named bench configuration with JSONL output and a
cross-run variance gate). Anything stochastic was gated distributionally or by algebraic
equivalence under a shared RNG draw, never by eyeball alone; anything eyeballed is labelled
as such.

## 1. Timeline of work

### Day 1 (2026-07-08) — Port: S0 groundwork → S1 toy parity → S2 real weights

| # | What | How tested | Why this way |
|---|---|---|---|
| 1 | Cached reference (`Tools/reference/sumi/`: config, modeling, generation, tokenizer files) | files present; toy-config `generate()` runs end-to-end in an isolated venv | the reference *is* the parity target; pin it before writing code. Venv (`scratch/sumi-venv`, transformers 5.13) because the reference needs ≥5.8 and the system 5.2.0 serves the LLaDA fixtures — isolation avoids breaking the older pipeline |
| 2 | Five design decisions taken (parity oracle, anchors, cache flag, softmax formulation, tokenizer) | n/a — decisions | cheap before code, expensive after; each is recorded in `sumi-plan.md` §2 |
| 3 | `SumiConfig` + `SumiTokenizer` | `SumiConfigTests` (real config.json values + Python-default-table semantics), `SumiTokenizerTests` (100-case encode/decode parity vs HF, both encode variants) | config absent-key semantics are a known LLaDA gotcha (defaults live in the Python class, not the JSON); the tokenizer suite caught that the `TemplateProcessing` post-processor prepends BOS — found by test, not by reading docs |
| 4 | Off-by-one softmax (`softmaxOne` multiplicative + `softmaxOneSink` verbatim oracle) and `OffByOneAttention` | `SumiCoreFixtureTests`: formulation A ≡ B ≤ 1e-6 fp32 incl. outlier logits ±(35–40); module output vs reference fixture 1e-5 | two-formulation twinning means the hot path is always checkable against the literal reference construction; the outlier-logit case targets the sink-underflow risk (softcap-free model) |
| 5 | Model stack (`SumiDecoderLayer`, `SumiModel`) | `SumiForwardParityTests`: toy FP32 full-forward **top-1 24/24, max \|Δprob\| 6.5e-9** | toy-config seeded-random-weight fixtures (dumped by `Tools/generate_sumi_fixtures.py`) run on the 16 GB machine where 15.8 GB BF16 cannot; the same test is config-driven so Studio BF16 fixtures reuse it verbatim |
| 6 | Sampler family (`LogSNRSchedule`, `SumiNoiseMask`, `UniformStateSampler`) | `SumiSamplerTests`: greedy + adaptive step functions token-exact vs reference; ancestral analytic posterior ≤ 1e-6; Gumbel-max draw gated **distributionally** (mode-mass band over 32 keys); schedule ≤ 5e-5 | deterministic parts gate exactly; `torch.multinomial` is not reproducible cross-RNG, so the stochastic draw is gated on distribution, and the deterministic posterior is split out precisely so *it* can gate exactly. The 5e-5 schedule tolerance is documented in-test: torch fp32 `linspace` ulps × logit slope at the ±9 clamp — matching torch's fp32 op chain got within 1 ulp-effect; bit-mimicking `linspace` was judged not worth it |
| 7 | `SumiEngine` end-to-end loop | `SumiLoopParityTests`: **8/8 cases token-for-token** — per-step canvases, final canvas, EOS trim; case matrix spans samplers × anchors × `denoise_end` × budget clamp × bos-only prompt | canvas init RNG is not portable, so fixtures ship the reference's initial canvas (`initialCanvas:` override) and everything downstream is deterministic — turning an end-to-end stochastic system into an exactly-gateable one |
| 8 | 4-bit artefact loader (`SumiDiffusionModel`) | conversion verified **bit-perfect**: artefact bytes ≡ `mx.quantize(BF16 original)`; U8→U32 packed-weight reinterpret validated by that same ground-truth diff | when a converted artefact underperforms, "conversion bug" must be excludable by evidence, not assumption — this check later saved days (see §2, canvas-lock investigation) |
| 9 | Real-weight smoke | `SumiRealWeightSmokeTests` (opt-in env gate): loads in ~3 s, generates; quality validated next day | opt-in gating keeps the default suite fast; every real-weight test in the campaign is env-gated |

### Day 2 (2026-07-09) — Investigation, optimisation, bench

| # | What | Result | Method note |
|---|---|---|---|
| 10 | **Canvas-length investigation** | initial probe said "locked ≥1536"; de-confounded generation experiment **overturned it: 1024 works, 512 marginal** | see §2 — the campaign's main methodology lesson |
| 11 | Quality validation | ancestral 64 steps t=0.7: *"The capital of Japan is Tokyo."* — correct, fluent | one-prompt eyeball, labelled as such; task-level claims deferred to a quality harness |
| 12 | Fused off-by-one attention, round 1: SDPA × sigmoid(row-LSE) | algebraic identity, parity-neutral (9.3e-9), **time-neutral** (85 vs 91 ms/layer), memory −40% | derivation: `softmax_one = softmax × sigmoid(LSE)`; kept as oracle when round 2 landed |
| 13 | Fused attention, round 2: **MLX native sinks** | `sinks:` param on fused SDPA (upstream, for gpt-oss); Sumi = one zero-logit sink/head → **22.7 ms/layer ≡ plain SDPA, 0.03 GB** (from 91 ms / 1.18 GB eager) | checking what the framework already ships beat writing a kernel; existing parity tests re-gated the swap for free |
| 14 | Ancestral sampler reduction | Gumbel-max over the posterior reduces to a two-candidate comparison (per-position constants can't change an argmax): **34.1 → 27.0 s/step** at 1536 | gated by *exact equivalence* under a shared uniform draw (`testAncestralReducedFormEquivalence`, 16 keys, 0 mismatches) — algebraic rewrites of samplers can be gated exactly even though sampling is stochastic |
| 15 | Lever experiments (`SumiLeverExperiments`) | canvas 512/1024/1536 × k∈{1,4} × revision budget — see summary table | one axis per arm, everything else held fixed; per-step canvas readbacks deliberately allowed (diagnostic suite, not the sync-budgeted engine) |
| 16 | Bench (`diffusion-bench sumi`) + frozen baseline | 3 arms × 3 runs, cross-run variance gate < 5%; recipe arm froze at **155 s (0.2% dev)** | JSONL per arm-run so later A/Bs diff against records, not memory; `--cooldown` added after thermal outliers; `setvbuf` line-buffering after an hour of invisible output |
| 17 | S3.2 A/B: LSE-confidence sampler | conf from max/LSE/gather, softmax never materialised: **−3.5%, peak −0.38 GB**, token-exact vs fixtures | chosen over fp16 logits (which would blur exactly the top-k margins the sampler ranks by); structural removal beats precision downgrade |
| 18 | S3.2 A/B: `asyncEval` pipelining | **−9% steady state** (136–138 s); run-0 artefact = kernel compile racing the dispatch queue | falsified the earlier "GPU is compute-saturated" reading — ~0.8 s/step was dispatch overhead. FLOP-based intuition failed twice this campaign (see §3) |
| 19 | Projection-fusion micro-bench | separate ≡ fused within 0.2% at QKV and gate+up shapes → **do not build** | 2-minute micro-bench with asserted numerical equivalence answered a would-be day of loader surgery; negative result kept in-tree as evidence |
| 20 | LM-head 4-bit eval | **8.3 s/step (133 s, best arm), peak −0.55 GB, 0 confident-position flips**; naive all-positions flip rate 93% was a broken metric (flat random-tail rows flip freely) | first eval *failed* and the failure was metric design, not the head — see §3; refined gate: flips where F16 p_max ≥ {0.3, 0.5, 0.7} |
| 21 | SchED-style early exit (S4.2) | three iterations: whole-canvas signal never fires → content-window never fires at exact stability → **fractional (0.9) fires: ancestral 26/64 steps, 230 s vs 594–659 s (2.6×), correct output** | each non-firing round was diagnosed from arm data (which positions churn), not guessed; greedy + `denoise_end` found degenerate and recorded as known-bad |

## 2. The canvas-length investigation (worked example of the falsification loop)

Recorded in full because it is the campaign's best example of the method catching its own
error, and the template for investigating "the model behaves badly" on any future port.

1. **Observation**: first real-weight generations were babble; a clean-text reconstruction
   probe (clean 35-token prefix + random tail, measure argmax self-agreement) scored 2–8/35
   at canvas ≤1024 but 25–32/35 at ≥1536.
2. **Exclusion of port/quantization bugs** — three independent implementations produced
   identical behaviour: the Swift 4-bit path, a from-scratch python-MLX forward (F16 *and*
   FP32 activations), and the **unquantized BF16 model** streamed layer-by-layer through a
   15.5 GB file on a 16 GB machine. Plus the bit-perfect conversion check (§1.8). Conclusion
   at the time: model property, "canvas-length-locked ≥1536".
3. **Challenge (André)**: training reportedly used canvases 1024–4096, and the
   generation-level evidence was **confounded** — the small-canvas babble runs had also used
   a different sampler and step count.
4. **De-confounded experiment**: same sampler, steps, k, budget, prompt; canvas as the only
   axis (`SumiLeverExperiments.testCanvasAxis`). Result: **canvas 1024 produces clean,
   correct output**; 512 answers correctly with degraded continuation.
5. **Resolution**: the probe was the artefact — a mostly-*clean* canvas is itself
   off-distribution for the model's global noise-level inference, unrelated to the
   mostly-noise generation regime. The probe was demoted to a weight-unpacking sanity check
   (≥1536 only) and the retraction recorded in the plan.

**Lessons encoded**: (a) cross-implementation agreement excludes bugs but cannot validate a
*proxy metric*; (b) never change two variables between the observation and the conclusion;
(c) write the retraction into the same document that carried the claim.

## 3. Metric-design failures (both caught, both documented)

1. **FLOP-based cost attribution failed twice.** "Attention is <1% of the step" (FLOPs)
   → actually ~15% of wall-clock (unfused elementwise costs); "the GPU is compute-saturated
   at ~25–50% of peak" → ~0.8 s/step of it was dispatch overhead recoverable by `asyncEval`.
   Rule adopted: cost claims come from measured arms or traces (`xctrace`), never from
   FLOP arithmetic alone.
2. **Aggregate metrics over mixed populations lie.** The 93% LM-head flip rate pooled
   near-flat random-tail rows (where argmax flips are free) with confident rows (where they
   matter). Conditioning on confidence (F16 p_max ≥ τ) showed **zero** meaningful flips.
   Rule adopted: any "X% changed" metric must be conditioned on whether the change can
   matter downstream.
3. **The replacement gate had the same flaw one level down (caught by André).** The
   p_max-thresholded gate pooled razor-margin positions (top-1/top-2 ≈ 0.40/0.35, flips
   expected and benign) with wide-margin ones (0.40/0.02, flips = damage), and was blind to
   ambiguity below its thresholds — exactly the population adaptive top-k ranks over. The
   margin-conditioned replacement (`testHeadQuantizationMarginAnalysis`, run on 4 096
   positions from 4 **real mid-generation canvases**, not the synthetic probe) measured:
   flips 34%/30%/3-of-39 in the [0,0.01)/[0.01,0.05)/[0.05,0.10) margin bins, **zero in
   every bin ≥ 0.10**; **largest margin that flipped 0.0890 vs p99 margin-perturbation
   0.0894** — flips confined exactly to below the measured noise scale, the calibrated
   signature of benign quantization. Downstream: adaptive k=4 committed tokens agree at
   all jointly-selected positions; selection *sets* differed on 2/4 canvases (near-tie
   reordering at the top-k cut → different-but-plausible trajectories, consistent with the
   arm texts). Rules adopted: condition flip metrics on **decision margin**, calibrate
   against the measured perturbation scale, and test the downstream selection directly;
   caveat: mid-margin bins are small-n (22–39) — the perturbation-scale argument, measured
   on all positions, carries the wide-margin claim.

## 4. Results summary table

All M1 16 GB, 4-bit weights, canvas 1024, 64-token budget unless noted. "Frozen baseline" =
`diffusion-bench` 3-run arms with cooldown.

### Correctness (gates, all green — 59 tests, 9 env-gated opt-ins)

| Layer | Gate | Result |
|---|---|---|
| Config / tokenizer | value + 100-case parity | ✅ exact |
| Off-by-one softmax | A ≡ B ≤ 1e-6 incl. outliers | ✅ |
| Attention module | vs reference fixture | ✅ 1e-5 |
| Full forward (toy FP32) | top-1 + Δprob | ✅ 24/24, 6.5e-9 |
| Samplers (deterministic) | token-exact vs reference | ✅ |
| Ancestral posterior / reduced form | 1e-6 / exact under shared draw | ✅ |
| End-to-end loop | 8 trace cases token-for-token | ✅ 8/8 |
| Conversion | bytes ≡ `mx.quantize(BF16)` | ✅ bit-perfect |
| BF16 real-weight parity | — | ⏸ Studio-deferred (§7 of plan) |

### Performance & verdicts

| Item | Before | After | Verdict |
|---|---|---|---|
| Attention (per layer, S=1536) | eager 91 ms / 1.18 GB | **native sinks 22.7 ms / 0.03 GB** | ✅ adopted (≡ SDPA floor) |
| Ancestral sampler (per step, S=1536) | +15.6 s | **+8.5 s** (reduced form) | ✅ adopted (exact-equiv gated) |
| Canvas operating point | 1536 (wrong) | **1024** (512 for smoke) | ✅ revised via de-confounded exp |
| k (tokens/step, adaptive) | 1 | **4** (16 steps for 64 tokens) | ✅ adopted; paper Fig-4 confirmed |
| Revision budget | 32+ steps | **coverage-exact** (`freezeCommitted`) | ✅ extra passes churn, never help |
| Frozen recipe baseline | — | **155 s / 0.41 tok/s (0.2% var)** | reference point |
| LSE-confidence sampler | 155 s | **149–153 s, −0.38 GB** | ✅ adopted (token-exact) |
| `asyncEval` pipelining | 149–153 s | **136–138 s (−9%)** | ✅ adopted for serving; needs warmup run in bench |
| Projection fusion (QKV, gate+up) | — | Δ ≤ 0.2% | ❌ dead — do not build |
| LM-head 4-bit | 9.3 s/step, 6.30 GB | **8.3 s/step, 5.75 GB; margin-gated: 0 flips at margin ≥ 0.10, largest flipped margin 0.089 ≈ p99 noise 0.089** | ✅ adopt-candidate (margin-calibrated, §3.3; task-level run pending — Q4 trajectories tie-break-diverge, so identity is not on offer) |
| Early exit (ancestral, frac 0.9, R=3) | 594–659 s | **230 s (26/64 steps), correct output** | ✅ adopted for ancestral; greedy+window known-bad |
| **Net (recipe arm)** | 19.4 s/step, 0.03–0.33 tok/s (day 1) | **8.3 s/step, ~0.48 tok/s**; ancestral 0.28 tok/s | ~10–16× end-to-end on ancestral-quality output |

Studio extrapolation (~5.5× FLOPS, **inferred**): ~1.5–2.5 tok/s. Verify on return.

## 5. Metal / mega-kernel evaluation

**Question**: would a hand-written Metal kernel (per metal-shader-guide) or a whole-step
mega-kernel (phase-1 §13.1 escape hatch) provide meaningful benefit for Sumi?

**Verdict: no. Estimated total ceiling for all remaining kernel work: ~4–8% of step time,
most of it already claimed by `asyncEval` and claimable by MLX `compile()`.** Grounds, each
measured this campaign:

1. **The step is quantized-GEMM-bound.** ~14.6 TFLOP/step at canvas 1024 through MLX's
   `quantized_matmul`; the fusion micro-bench showed those GEMMs have **zero recoverable
   launch overhead** (fused ≡ separate within 0.2%). A kernel cannot reduce FLOPs, and
   MLX's qmm is already at hardware-efficiency territory — beating it is an MLX-upstream
   problem, not an app-kernel problem.
2. **Attention is already at the fused-SDPA floor.** Native sinks made the off-by-one
   attention *exactly* as fast as plain SDPA (22.7 vs 22.5 ms/layer). The originally
   planned S4.1 flash-attention-with-sink kernel would at best tie this: **0% available**.
3. **Dispatch/idle overhead is ~0.8 s/step and `asyncEval` already recovers it** (−9%
   measured). A mega-kernel's main structural benefit — removing inter-kernel gaps — is
   therefore mostly spent. MLX `compile()` (elementwise fusion, step-invariant shapes)
   is the remaining cheap claimant for what's left.
4. **What remains for hand-Metal**: the elementwise islands — RMSNorm×2/layer (fp32
   internals, ~8 MB/pass), SwiGLU epilogue (~75 MB/layer), residuals, and the sampler's
   `[S, V]` log/Gumbel chain (~0.3 s). Generous bandwidth arithmetic bounds all of it at
   ~0.3–0.7 s/step ≈ **4–8%**, against days of work and a new correctness surface — and
   `compile()` may take half of it for one line.
5. **Phase-1 §13.1 trigger check**: "loop overhead >10–15% of step time after Phase-3
   fusion" — with pipelining on, measured loop overhead is below the trigger. **The
   escape hatch stays closed.**

What *would* move the needle instead, in measured-impact order: step count (early exit —
2.6× already), tokens-per-step (4×), hardware (Studio ~5.5×), MLX-upstream qmm
improvements, and batching for *throughput* (not latency) if serving ever needs it.

## 6. Operational quirks worth remembering (all hit this campaign)

1. SwiftPM omits mlx-swift's metallib bundle from **release** executable dirs → `cp` the
   `mlx-swift_Cmlx.bundle` next to the binary (documented in bench README).
2. Swift `print` to a redirected file is block-buffered → an hour of invisible bench output;
   `setvbuf(stdout, nil, _IOLBF, 0)` at process start.
3. Converted artefacts may store packed quantized weights as safetensors `U8 [out, in/2]`
   while MLX wants `U32 [out, in·bits/32]` — same bytes, `view(dtype:)` reinterpret;
   *verify against a ground-truth dequantize*, don't assume nibble order.
4. Laptop thermal/system interference makes >10-minute bench arms fail 5% cross-run
   variance regardless of cooldowns; compare within-run steady medians there, and re-check
   gates on desktop hardware.
5. The reference implementation's transformers version may exceed the system install —
   isolate fixtures in a venv (`--system-site-packages` reuses the big torch install).
6. `MLXNN.quantize(model:filter:)` quantizes in-memory weights — head/embedding quant
   experiments need no artefact changes (post-load, one call).
7. Check the framework's current API surface before writing kernels: MLX had grown native
   SDPA `sinks:` support (for gpt-oss) that made our two earlier attention fusions and the
   planned Metal kernel obsolete in one parameter.
