# NeoDiffusion — Sumi-7B Port Plan (away-from-Studio track)

**Status**: **S0–S4 COMPLETE (2026-07-09)** — port parity-green, real-weight inference
validated, baseline frozen, optimisation levers measured and landed. Only the [STUDIO]
column (§7) remains. **Full evidentiary record + results summary + Metal verdict:
[`sumi-logbook.md`](./sumi-logbook.md). Process recipe for the next model:
[`sumi-handoff.md`](./sumi-handoff.md).**
**Headline (M1 16 GB, 4-bit, canvas 1024, 64-token budget)**: recipe-adaptive-k4 at
**8.3 s/step / ~0.48 tok/s** (from 19.4 s/step on day 1); ancestral-with-early-exit
**230 s vs 594–659 s baseline (2.6×)** with correct output. 59 tests green.
**Written**: 2026-07-08, post-M5, while the Studio is unreachable

## Status (2026-07-08): Phase S0 + S1 complete — 46/46 tests green (26 LLaDA + 20 Sumi)

**Decisions (André, 2026-07-08)**: (1) parity budget on greedy + adaptive, ancestral gated
distributionally; (2) in-prompt anchors rejected at API level; (3) cache disabled during
denoise; (4) off-by-one softmax per metal-shader-guide §2.4 — multiplicative form on the hot
path, additive-sink form as oracle; (5) tokenizer confirmed from `tokenizer.json`: standard
HF `tokenizers` BPE (cl100k-style regex, OLMo-family added tokens), loads directly via
swift-transformers — no tiktoken port.

**S0.2 refactor: skipped** (per André). `LLaDA2RMSNorm` (ε param), `PartialRotaryEmbedding`
(`partialRotaryFactor: 1.0` = full RoPE), and `LLaDA2MLP` (bias-free SwiGLU) are reused
as-is. Only two shared-code touches: a behavioural no-op full-rotary fast path in
`PartialRotaryEmbedding.apply` (avoids a zero-width passthrough slice), and `MLXRandom`
added to DiffusionGeneration's deps.

**Environment**: reference needs transformers ≥ 5.8 (imports `type_validators.interval`
etc.); system install is 5.2.0, so fixtures run in an isolated venv reusing system torch:
`scratch/sumi-venv` (transformers 5.13). All fixture commands below use
`scratch/sumi-venv/bin/python`.

**What landed** (all gates [LOCAL], toy config FP32, seed 0):

| Step | Files | Gate result |
|---|---|---|
| S1.1 config | `Packages/DiffusionModel/Sources/SumiConfig.swift`, `SumiTokenizer.swift`; tests `SumiConfigTests`, `SumiTokenizerTests`; `Tools/generate_sumi_tokenizer_fixtures.py` | real config.json decodes with all documented values; 100-case tokenizer parity (encode raw + with-specials, decode). Found: the tokenizer's `TemplateProcessing` post-processor **prepends BOS** on default encode — prompts to `generate` carry it |
| S1.2 attention | `Packages/DiffusionCore/Sources/SoftmaxOne.swift`, `OffByOneAttention.swift`; tests in `SumiCoreFixtureTests`; `Tools/generate_sumi_fixtures.py` | formulation A ≡ B ≤ 1e-6 fp32 incl. outlier logits (+40/−35); attention vs reference 1e-5; hot path vs sink oracle ≤ 1e-6 |
| S1.3 stack | `Packages/DiffusionCore/Sources/SumiDecoderLayer.swift`, `Packages/DiffusionModel/Sources/SumiModel.swift`; test `SumiForwardParityTests` | full-forward top-1 **24/24**, max \|Δprob\| **6.5e-9**, max \|Δlogit\| 7.2e-7 (float-epsilon, M4-quality) |
| S1.4 samplers | `Packages/DiffusionGeneration/Sources/LogSNRSchedule.swift`, `SumiNoiseMask.swift`, `UniformStateSampler.swift`; test `SumiSamplerTests` | greedy + adaptive (k=1,3) steps token-exact; ancestral posterior ≤ 1e-6; Gumbel-max draw gated distributionally; schedule ≤ 5e-5 (residual = torch fp32 `linspace` rounding × logit slope at clamp edges — recorded in test comment) |
| S1.5 engine | `Packages/DiffusionGeneration/Sources/SumiEngine.swift`; test `SumiLoopParityTests`; `Tools/generate_sumi_loop_fixtures.py` | **8/8 cases token-for-token** (per-step canvases + final canvas + EOS trim; greedy/adaptive × anchors on/off × denoise_end × budget clamp × bos-only prompt); ancestral end-to-end smoke green |

**Deviations from the reference introduced in S1** (parity-suspect list, §5-style):
1. Ancestral sampling via Gumbel-max over the analytic posterior instead of
   `torch.multinomial` — distributionally equivalent, not RNG-identical (decision 1).
2. In-prompt anchors rejected (`precondition`) instead of silently skipped (decision 2).
3. Multiplicative off-by-one softmax on the hot path; sink form kept as oracle (decision 4).
4. `mask: nil` in generation instead of the reference's all-ones 2D mask — proven identical
   (asserted in the fixture dumper: ones-mask expands to an all-zero additive mask).
5. Canvas init RNG is MLX's, not torch's — parity fixtures ship the reference's initial
   canvas via `SumiEngine.generate(initialCanvas:)`.

**Fixture regeneration**:
```
scratch/sumi-venv/bin/python Tools/generate_sumi_tokenizer_fixtures.py
scratch/sumi-venv/bin/python Tools/generate_sumi_fixtures.py
cd Tools && ../scratch/sumi-venv/bin/python generate_sumi_loop_fixtures.py --out ../scratch/sumi_loop_fixtures
```
The `NEODIFFUSION_SUMI_FIXTURE_DIR` / `NEODIFFUSION_SUMI_LOOP_FIXTURE_DIR` overrides point
the same tests at Studio BF16 real-weight fixtures later (§7).
**Scope**: current state (M0–M5 green on toy-config FP32, LLaDA2.1-mini only) → basic Sumi inference on the M1 dev machine → optimised fused kernels.
**Companion documents**: [`sumi-implementation-guide.md`](./sumi-implementation-guide.md) (the model dossier this plan executes — read §1–§4 first), [`lmoe-implementation-guide.md`](./lmoe-implementation-guide.md) (§3.2's refactor sequencing is borrowed here), [`handoff-post-M5.md`](./handoff-post-M5.md) (current engine state), [`phase-2-implementation-guide.md`](./phase-2-implementation-guide.md) (the fixture/gate discipline this plan copies).

Provenance discipline applies: claims below are **sourced** (from the guides / repo), **inferred**, or **speculative**, flagged inline.

---

## 0. Why Sumi, why now — the viability argument

The Studio is unreachable; the open LLaDA2.1-mini items (M4/M5 BF16 real-weight parity, M6 Studio baseline) are blocked on it. The question is what makes progress on the M1 16 GB alone.

- **Sumi is the only candidate that runs on the M1 with real weights**: ~5.4 GB at 4-bit, ~6.8 GB peak with activations (**sourced**, sumi guide §8). LMOE fits at 4-bit too (~5.7 GB) but its parity anchor and its ExactPrefixCache design question (LMOE guide §3.4) both need work that is better done with the Studio and the paper in hand.
- **The toy-config fixture technique transfers.** `modeling_sumi.py`/`generation_sumi.py` are config-driven, so a toy-size PyTorch reference with seeded random weights runs on the M1 — the same pattern that made M0–M5 green locally (**inferred** from the reference code being pure-config `PreTrainedModel`; verify in S1.2 when the first fixture dumper runs).
- **What stays Studio-deferred**: real-weight BF16 greedy parity (15.8 GB BF16 cannot load on the M1 — **sourced**, sumi guide §8.1), the 4-bit-vs-BF16 Levenshtein gate, and the pre-existing LLaDA M4/M5 BF16 items. These queue up cleanly for the office return (§7).
- **Deviation from the sumi guide's ordering**: the guide recommends Sumi *after* LMOE lands (§TL;DR). This plan starts Sumi first because LMOE offers no away-from-Studio advantage. To avoid paying the multi-model refactor twice, Phase S0 pulls forward exactly the model-agnostic refactor prefix from the LMOE guide §3.2 (the steps that keep all existing tests green), and defers the MoE-side abstractions (`SoftmaxTopKRouter`, shared-expert-optional `SparseMoEBlock`) until LMOE actually starts. Flagged per working rules: this plan *is* the update to the Plans docs for that deviation.

**Path correction (repo reality)**: the sumi guide's file paths (`Sources/NeoDiffusion/Configs/...`) do not match this repo. Actual layout is `Packages/{DiffusionCore,DiffusionModel,DiffusionGeneration,DiffusionKernels}/Sources/` + `Tests/{DiffusionCoreTests,DiffusionGenerationTests,DiffusionKernelsTests}` + `Tools/*.py` fixture dumpers. All paths below use the real layout. Hexagonal mapping (phase-1 §4) is preserved: model math in DiffusionCore/DiffusionModel, sampling loop in DiffusionGeneration, Metal in DiffusionKernels.

---

## 1. Gate philosophy: two tracks, one discipline

Every milestone keeps the phase-2 discipline (fixtures dumped once by a Python script, Swift tests diff against safetensors, no PyTorch at test time), but each acceptance gate is tagged:

- **[LOCAL]** — runnable on the M1 now. Toy-config random-weight fixtures, 4-bit real-weight smoke runs, kernel benchmarks.
- **[STUDIO]** — needs the Studio (real-weight BF16). These are *recorded as open*, not skipped; a milestone with only its [STUDIO] gate open may proceed to the next milestone (same concession already made for LLaDA M4/M5).

The existing 26/26 LLaDA tests are a standing regression gate for **every** step below. Any step that turns one red is reverted or fixed before proceeding.

---

## 2. Phase S0 — Groundwork (no Sumi model code)

### S0.1 Fetch and cache the reference artefacts [LOCAL]

Cache under `Tools/reference/sumi/` (same pattern as the cached LLaDA `.py` noted in `handoff-post-M5.md`):

- `config.json`, `generation_config.json`, `modeling_sumi.py`, `generation_sumi.py`, `configuration_sumi.py` from `tohoku-nlp/sumi-7b`.
- **`tokenizer_config.json` + `tokenizer.json`** — this resolves the guide's #1 uncertainty flag (cl100k-family or custom fork; **speculative** until fetched). Do this before any tokenizer code.
- `model.safetensors.index.json` — weight-name verification without downloading weights.

**Accept**: files cached and skimmed; the five §3.4 decisions below answered by André (they are cheap now, expensive later):

1. **Ancestral parity oracle** — recommend option (a): greedy-only exact parity; ancestral tested as mode-of-32-seeds (guide §3.4.1).
2. **In-prompt frozen anchors** — recommend rejecting at API level (guide §3.4.2).
3. **Cache during denoise** — recommend `disableCacheDuringDenoise = true` default for uniform-state samplers; prompt-slice caching is a Phase-3-style opt, not baseline (guide §3.4.3, §1.1).
4. **Off-by-one softmax formulation** — recommend multiplicative form on the hot path, additive-sink form kept as the validation oracle (mirrors the SDPA-vs-`attendReference` pattern from M3) (guide §3.4.4).
5. **Tokenizer** — decided by the fetched `tokenizer_config.json`, not by recommendation.

### S0.2 Pull-forward refactor (LMOE guide §3.2, steps 1–5 adapted) [LOCAL]

Each step ships alone; the 26/26 suite stays green after each. Files per the *actual* package layout:

1. Generalise RMSNorm ε if anything is hardcoded (Sumi: `rms_norm_eps` from config; LLaDA2.1-mini: 1e-6). Likely a no-op — verify, don't assume.
2. Parameterise the rotary embedding on (`rotaryDim`, `theta`) so `rotaryDim == headDim` expresses Sumi's full RoPE at θ=500 000. Defaults must reproduce LLaDA2.1-mini bit-for-bit.
3. Extract an `Attention` protocol in `Packages/DiffusionCore`; current `LLaDA2Attention` conforms (rename optional, defer churn).
4. Introduce a family-config protocol in `Packages/DiffusionModel`; `LLaDA2MoeConfig` conforms.
5. Extract a `SamplingPolicy` protocol in `Packages/DiffusionGeneration`; the current `DiffusionEngine.generate/generateCached` loop becomes `DraftAndEditPolicy` behind it, engine becomes a thin driver.

**Skip for now** (LMOE-specific, not needed by dense Sumi): router split, `SparseMoEBlock` shared-expert-optional field, `LLaDAMoEConfig`.

**Accept**: `swift test` 26/26 after each step; no behavioural change anywhere.

**Honest sizing note**: step 5 is the same "if it's too big, defer" call the LMOE guide makes (§3.2). Fallback: land 1–4, build Sumi's sampler family as free-standing types, and do the `SamplingPolicy` extraction when the third policy (LMOE) forces it. Sumi's samplers share almost no structure with draft-and-edit anyway (**sourced** — guide §2.5), so the abstraction pressure is genuinely lower than for LMOE. Recommend attempting step 5 timeboxed; don't die on it.

---

## 3. Phase S1 — Basic inference (toy-config parity, all [LOCAL])

Maps to the guide's `sumi-M1'`–`M5'` with gates re-scoped to the two-track scheme. New fixture dumper: `Tools/generate_sumi_fixtures.py`, following the `generate_core_fixtures.py` / `generate_loop_fixtures.py` pattern (toy config, seeded, safetensors + `manifest.json`, config-driven Swift tests with a `NEODIFFUSION_SUMI_FIXTURE_DIR` override so the *same* tests later run the Studio BF16 fixtures).

### S1.1 = sumi-M1' — Config + tokenizer

- `Packages/DiffusionModel/Sources/SumiConfig.swift`: fields per guide §3.3, full Python-default table encoded (same absent-key discipline as gotcha 2 — read `configuration_sumi.py`, not just `config.json`).
- Tokenizer wiring per the S0.1 finding (swift-transformers; cl100k-family expected).
- **Accept [LOCAL]**: config round-trips on the real `config.json` (hidden 4096, 36 layers, 32/8 heads, head_dim 128, θ=500 000, vocab 100 278, bos/eos/pad 100256/100257/100277, `tie_word_embeddings=false`); tokenizer encode/decode parity on a 100-case suite vs HF (reuse `generate_tokenizer_fixtures.py` machinery).

### S1.2 = sumi-M2' — Off-by-one softmax + attention

- `softmaxOne` (additive-sink, oracle) and `softmaxOneMultiplicative` (hot path) in `Packages/DiffusionCore` — guide §4.1 sketches.
- `OffByOneAttention`: split QKV, no biases, no qk-norm, full RoPE, GQA 32/8. **MLX fused SDPA cannot be used** (`_supports_sdpa=False` is load-bearing — the sink changes the normaliser; **sourced** guide §TL;DR-1). Manual attention path; the fused Metal kernel is Phase S4.
- Fixture: toy-config `SumiAttention` intermediates from `modeling_sumi.py`.
- **Accept [LOCAL]**: (a) formulation A ≡ B on random logits to ≤1e-6 fp32, including an outlier-logit case (|x|≥30, guide §10 risk — Sumi has no softcap); (b) attention output matches the reference fixture to 1e-5 fp32.

### S1.3 — Model stack + full-forward parity (analog of LLaDA M3+M4)

- `SumiMLP` (SwiGLU, reuse existing if shaped right), `SumiDecoderLayer` (pre-norm, no qk-norm), `SumiModel` in `Packages/DiffusionModel`, registered with the engine/config dispatch.
- Honour the two reference quirks: logits truncated to `vocab_size` then upcast fp32 before softmax (guide §2.5.7); no attention mask, `use_cache=False` (guide §2.5.6).
- Fixture: `generate_sumi_fixtures.py --toy` dumps per-module intermediates + full-forward logits over a `[prompt | random canvas]` sequence.
- **Accept [LOCAL]**: per-module fixture diffs at the M3 tolerances; full-forward toy FP32 top-1 100% + |Δprob| within the M4-style bound. **[STUDIO] deferred**: same test, BF16 real weights.

### S1.4 — Sampler family

Order: greedy → adaptive → ancestral (deterministic first, so parity failures have one suspect at a time).

- New types in `Packages/DiffusionGeneration`: `LogSNRSchedule` (linear + cosine), `NoiseMask`/`FrozenAnchor` (prompt-frozen + pinned `[EOS,BOS]` anchor + `denoise_end` — first-class, per guide §2.5.2; in-prompt anchors rejected per decision S0.1-2), `GreedyUniformSampler`, `AdaptiveUniformSampler`, `AncestralUniformSampler` (Gumbel-max form from guide §4.4, not `mx.random.categorical` — avoids materialising the fp32 posterior).
- Fixture: per-step traces from `generation_sumi.py` on the toy model — greedy full trace; adaptive per-step selected positions + committed ids; ancestral final outputs over 32 seeds.
- **Accept [LOCAL]**: greedy per-step canvas token-for-token; adaptive selected-position sets match up to ties (permutation-tolerant, guide §3.2 `sumi-M5'`); ancestral mode-of-32-seeds matches reference mode-of-32 (per decision S0.1-1); `LogSNRSchedule` values match `_make_log_snr_schedule` to 1e-6.

### S1.5 — End-to-end generate (the "basic inference" gate)

- `generate(model:request:)` per guide §4.5: canvas init (uniform randint), noise mask, fixed-step loop, EOS trim. Loop-control note: Sumi's loop is a **fixed step count with no break condition** (**sourced** §2.3) — the M5 K-step speculation/rollback machinery is unnecessary here; there are zero mandatory per-step readbacks until SchED (S4) adds an early-exit test. Record the sync-point count anyway.
- **Accept [LOCAL]**: toy-config end-to-end greedy generation token-for-token vs reference on ≥8 prompts × anchors on/off × `denoise_end` set/unset; anchor appears at `prompt_len + max_new_tokens` in every output. **[STUDIO] deferred**: the guide's `sumi-M3'` real-weight greedy gate (8 prompts, BF16, token-for-token).

**End of Phase S1 = "basic inference" achieved**: a correct-by-toy-parity Sumi engine, one `[STUDIO]` column of deferred re-runs.

---

## 4. Phase S2 — Real weights on the M1 (4-bit) [LOCAL]

### Status (2026-07-08): S2.1 done by André; S2.2 loader + smoke landed; two findings

**S2.1**: André downloaded BF16 (`models/sumi-7b`, 15.5 GB single-file) and converted to
4-bit (`models/sumi-7b-4bit`, 5.55 GB; affine g64, attention+MLP quantized, embed/lm_head/
norms F16; config.json carries the `quantization` block). **Conversion verified bit-perfect**
(**sourced**, measured): the artefact's packed bytes are identical to `mx.quantize(orig,
group_size=64, bits=4)` of the BF16 tensor, and dequantize to within expected 4-bit noise.
Note the artefact stores packed weights as safetensors `U8 [out, in/2]`; MLX's
`QuantizedLinear` wants `U32 [out, in·bits/32]` — `SumiDiffusionModel.sanitize` reinterprets
the bytes (little-endian, low-nibble-first; verified by the ground-truth comparison).

**Loader**: `Packages/DiffusionModel/Sources/SumiDiffusionModel.swift` (quantize-on-detect +
U8→U32 sanitize + load), quantization metadata decoded by `SumiConfig.Quantization`.
Smoke test: `SumiRealWeightSmokeTests` (opt-in via `NEODIFFUSION_SUMI_REAL=1` — minutes of
wall-clock, so not in the default suite).

**Finding 1 (REVISED 2026-07-09) — the "canvas-length lock" was a probe artefact; canvas
1024 works.** The original evidence: clean-text reconstruction agreement (35-token clean
prefix, random tail) of 3/35 @128, 2/35 @512, 8/35 @1024, 32/35 @1536, 25/35 @2048 —
identical across the Swift 4-bit path, an independent python-MLX forward, and unquantized
BF16 (so at least not a port/quantization bug). But that evidence was **confounded at the
generation level** (the canvas-128 babble used a different sampler and step count), and
André flagged it — training reportedly used canvases 1024–4096. The de-confounded
generation experiment (`SumiLeverExperiments.testCanvasAxis`, adaptive k=4/16 steps, all
else identical) settles it:
- **canvas 1024: clean, correct output** — the guide §8.4's "1024 for M1 iteration" advice
  was right after all; the reconstruction probe was the wrong proxy (a mostly-clean canvas
  is itself off-distribution for the model's global noise-level inference, unrelated to the
  mostly-noise generation regime).
- **canvas 512: primary answer still correct**, continuation region visibly degraded —
  marginal, usable for quick iteration where only the first answer matters.
- The reconstruction probe (`testCleanReconstructionInCanvas`) is **demoted to a
  weight-unpacking sanity check at canvas ≥1536 only**; do not use it to reason about
  usable canvas lengths. Negative result recorded per house rules.

**Finding 2 — uncertainty flag #8 resolved pessimistically; the bottleneck is GEMMs, not
attention (measured 2026-07-09).** At canvas 1536 a greedy step costs ~18.5 s on the M1 —
and it is **GEMM-bound**: ~22 TFLOP of quantized matmuls per step (7B params × 1536 tokens)
against a few-TFLOPS GPU; attention is <0.2% of the FLOPs. Verified directly: swapping eager
attention for the fused path (below) moved step time by only ~5%. Effective throughput
≈ 0.3 tok/s greedy / 0.03 tok/s ancestral-64-steps over a 48–64-token budget — far below
the guide's speculative 2–5 tok/s. Consequences for the optimisation phases:
- **S4.1 (Metal off-by-one attention kernel): RESOLVED — closed by MLX native sinks
  (2026-07-09).** MLX's fused SDPA supports attention sinks natively (`sinks:` parameter,
  added upstream for gpt-oss-style models); Sumi's `softmax_one` is exactly one zero-logit
  sink per head, so `attendFast` is now a **single fused kernel call**. A/B micro-bench
  (`SumiAttentionABBench`, per layer, S=1536): eager 91 ms / 1.18 GB → sigmoid(LSE)
  two-pass 85 ms / 0.75 GB → **native sinks 22.7 ms / 0.03 GB** (≡ plain SDPA — the sink is
  free). Parity gates pass *through* the sinks path (full-forward 9.3e-9, loop 8/8), which
  also validates MLX's sink semantics against the verbatim reference oracle. The
  sigmoid(LSE) two-pass (`attendFastLSE` + `rowLogSumExp`) is retained as fallback/oracle.
  André's review observations that led here are on record: (a) the fast path fires on
  literal `nil` mask only — routing contract documented, a zero-additive mask silently hits
  the eager path; (b) the LSE's full-length `repeat_kv` was replaced by a 5-D broadcast
  matmul (no K copy); (c) QKᵀ-twice made the LSE variant time-neutral vs eager — measured,
  and mooted by the sinks path. Note the earlier "<1% of step" claim was FLOP-based and
  wrong on wall-clock: attention was ~15% of step time via unfused-elementwise costs;
  native sinks recover ~1.1 s/step at canvas 1024. **Measured on real weights**: ancestral
  step at canvas 1024 now 9.0 s (previously adaptive — a lighter sampler — measured
  9.2–10.3 s there). Adaptive k=4/16 steps at 1024 now extrapolates to ~0.47 tok/s on M1.
- **Ancestral sampler overhead was ~15 s/step on top of the forward** (naive posterior
  materialises 5+ `[1536, 100278]` FP32 tensors — guide §4.4's warning realized). Landed:
  reduced two-candidate Gumbel-max form in `UniformStateSampler.ancestralStep` (per-position
  constants drop from the argmax; spike-vs-best comparison; exact-equivalence-gated against
  the full posterior in `testAncestralReducedFormEquivalence`). **Measured: ancestral step
  34.1 → 27.0 s** (sampler overhead ~15.6 → ~8.5 s; remainder is `log`/Gumbel elementwise
  traffic over `[1536, 100278]` FP32 — further shaving is an S3 bench item).
- **Remaining real levers**: tokens-per-step and step count (measured below), canvas 1024
  as the operating point (Finding 1 revised), and Studio hardware (~5–8× raw FLOPS).

**Lever experiments (measured 2026-07-09, `SumiLeverExperiments`, adaptive temp 0, one QA
prompt, 64-token budget, M1)**:

| Arm | avg step | tok/s | Quality (eyeball) |
|---|---|---|---|
| canvas 512, k=4, 16 steps | 4.9 s | **0.82** | answer correct; continuation degraded |
| canvas 1024, k=4, 16 steps | 10.3 s | 0.39–0.44 | answer correct; continuation clean-ish |
| canvas 1536, k=4, 16 steps | 16.3 s | 0.24 | answer correct; continuation mixed |
| canvas 1024, k=1, 64 steps | 9.2 s | 0.11 | best coherence (varied, factually correct QA) |
| canvas 1024, k=4, freeze-committed, 16 steps | 9.9 s | 0.39 | comparable to revise-16 |
| canvas 1024, k=4, revisions, 32 steps | 9.8 s | 0.20 | extra budget churned, did not improve |

- **k=4 is the paper's Figure-4 lever confirmed on our stack**: 4× fewer steps, primary
  answer intact, small glitch increase in the continuation — consistent with the paper
  (holds on HumanEval/MBPP, degrades GSM8K, collapse at k≥8).
- **Revision budget**: post-coverage steps kept committing k token-changes every step with
  no convergence (our `adaptiveStep` always commits k — it has no change-threshold), text
  kept churning (round trips 3/64), and quality did not improve — supporting the paper's
  "don't over-denoise". `freezeCommitted` (engine option, off by default, a documented
  deviation) gives exact coverage at `steps = window/k` and matched quality.
- **Practical M1 recipe**: canvas 1024, adaptive k=4, 16 steps ≈ **0.4 tok/s**; canvas 512
  for quick smoke ≈ 0.8 tok/s. Extrapolated Studio: ~2–4.5 tok/s (**inferred** from the
  ~5.5× FLOPS ratio; verify at §7).
- Single-prompt eyeball quality only — task-level claims need the S3 bench suite.

**S2.2 quality (measured 2026-07-09, 4-bit, canvas 1536)**: ancestral sampler, 64 steps,
temp 0.7, prompt "Question: What is the capital of Japan?\nAnswer:" → *"The capital of
Japan is Tokyo."* — correct and fluent, followed by typical packed-document continuation
(the base-model anchor behaviour the README describes). Clean-prefix reconstruction 18/34
at canvas 1536 (probe uses a deterministic LCG tail; agreement is tail-sensitive — the
python probe with a uniform tail scored 32/35). Greedy at 8 steps produces pad/backslash
babble — the "original naive sampler" needs far more steps; not a bug. Load 3.2 s, peak
GPU 6.8–7.3 GB. Smoke tests are opt-in: `NEODIFFUSION_SUMI_REAL=1` (+
`NEODIFFUSION_SUMI_STEPS` to shorten the quality run).

### S2.1 Download + streaming conversion

- Download the BF16 shards (~16 GB disk — check free space; the M1 also holds the LLaDA 4-bit artefact).
- Extend `Tools/convert_weights.py` for Sumi: 4-bit group-64 everything except embeddings + `lm_head` + norms at 16-bit (guide §8.2 keep-list; simpler than LLaDA's — no router, no experts).
- **Risk (the one real S2 risk)**: converting a 15.8 GB checkpoint on a 16 GB machine. **Inferred**: MLX's mmap-backed lazy loading + per-layer quantise/eval keeps peak RSS at ~one-layer granularity (~400 MB/layer), so it fits; **verify on first run**. Fallback: two-pass per-shard conversion (load shard → quantise → write → release), which is a mechanical change to the script.

**Accept**: artefact ≤~5.5 GB; zero unmatched keys in either direction vs `model.safetensors.index.json` (M1-style gate); loads on the M1.

### S2.2 Real-weight smoke + qualitative check

- Run all three samplers on real prompts, canvas 1024 (M1 default per guide §8.4 — attention scores scale with canvas², and the score tensor, not the weights, is the peak-memory item), 64 steps.
- **Accept [LOCAL]**: coherent text from all three samplers; anchors respected; peak memory recorded (expect ≲7 GB); no NaNs at BF16 activations. This is a *smoke* gate, not parity — 4-bit is never token-parity-comparable (house rule, gotcha 5).
- **[STUDIO] deferred**: the guide's `sumi-M6'` Levenshtein-≤3 gate needs BF16 reference completions; generate them on the Studio and check the 4-bit outputs then. (**Speculative** fallback if impatient: PyTorch CPU + mmap greedy on 4 short prompts overnight on the M1 — swap-bound, may not terminate usefully; strictly optional.)

---

## 5. Phase S3 — Bench + MLX-level optimisation [LOCAL]

Freeze a *local* baseline before touching kernels — same "no optimisation before a frozen baseline" rule as Phase 2/3.

### S3.1 status (2026-07-09): bench implemented; baseline round 1 measured

`Tools/diffusion-bench` Sumi mode landed (see its README for usage + the M6 metric
mapping): baseline suite (recipe-adaptive-k4 / quality-adaptive-k1 / reference-ancestral,
all canvas 1024, budget 64), custom arms for A/Bs, JSONL output, cross-run variance gate.
Two operational quirks found and documented: SwiftPM omits mlx-swift's metallib bundle from
release executable dirs (README workaround), and stdout must be line-buffered for streamed
logs (`setvbuf`, fixed).

**Baseline round 1 (M1, 4-bit, 3 runs/arm, no cooldown)**:

| Arm | step (steady) | tok/s | peak | variance |
|---|---|---|---|---|
| recipe-adaptive-k4 (16 steps) | 8.9 ± 0.3 s | 0.45 | 6.68 GB | **FAIL 15.2%** (run 2 spiked to 11.1 s/step) |
| quality-adaptive-k1 (64 steps) | 8.8–9.0 s | 0.11 | 6.68 GB | **PASS 1.3%** |
| reference-ancestral (64 steps, t=0.7) | 9.3 s | 0.10–0.11 | 6.72 GB | **FAIL 6.8%** (run 0 had ±3.9 s std) |

The two failures are isolated slow-step outliers (p95 → ~14 s in one run each) under ~70
minutes of sustained GPU load — system interference/thermal, not code (steady-state means
agree across arms to ~0.5 s). Mitigation added: `--cooldown` between runs (default 30 s).

**Baseline FROZEN (2026-07-09, M1, 4-bit, canvas 1024, budget 64, cooldown 45 s)**:

| Arm | step (steady) | tok/s | peak | cross-run variance |
|---|---|---|---|---|
| recipe-adaptive-k4 (16 steps) | 9.7 ± 0.1–0.5 s | **0.41** | 6.68 GB | **PASS 0.2%** (155/155/155 s) |
| quality-adaptive-k1 (64 steps) | 8.8–9.0 s | 0.11 | 6.68 GB | **PASS 1.3%** |
| reference-ancestral (64 steps, t=0.7) | 9.9–11.1 s | 0.09–0.10 | 6.72 GB | **FAIL 6.9%** — caveat below |

Ancestral caveat (**measured twice**, 6.8% and 6.9%): its ~11-minute runs accumulate
sporadic slow-step bursts (p95 spikes to ~14.6 s) that inter-run cooldowns cannot prevent —
laptop thermals/system interference during long sustained load, not code (its
*within-run steady medians* are stable at ~9.9–10.2 s). Guidance: A/B comparisons on the
ancestral arm must use steady-state step medians from the JSONL, not run totals; the two
adaptive arms' totals are safe comparators. The Studio re-baseline (§7) should re-check the
5% gate there — a desktop should hold it on all arms. Note also ancestral still carries
~+1 s/step of sampler cost vs adaptive — the remaining `[S, V]` fp32 elementwise traffic
(known S3.2 item).

- Add a Sumi suite to `Tools/diffusion-bench`: TPS, per-step wall-clock, steps (fixed), peak memory, sync-point count, per-phase split (canvas init / forward / sampler). TPF and steps/block don't apply (no blocks); document the metric mapping in the bench README.
- **Accept**: <5% variance across 3 runs on the M1 at 4-bit, canvas 1024, 64 steps. **This number is the guide's uncertainty flag #8** (claimed 2–5 tok/s, speculative) — measuring it is itself a deliverable; if it lands ≪0.5 tok/s, revisit whether S4 or step-count reduction is the right lever before writing Metal.

### S3.2 status (2026-07-09): two landed, A/B in flight; two open

- ✅ Gumbel-max ancestral + reduced two-candidate form (S1.4 / S2 follow-up).
- ✅ Canvas sweep — answered by the lever experiments (1024 operating point).
- ✅ One-eval-per-step discipline.
- **Landed + measured vs the frozen 155 s recipe baseline (2026-07-09)**:
  (a) **LSE-confidence adaptive sampler** (André's structural fix): `conf = exp(l_max −
  LSE) − exp(l_z − LSE)` from max/LSE/gather — no `[S, V]` softmax materialisation, and no
  fp16-blurring risk at the top-k decision margins (the rejected alternative: downcasting
  logits). Tempered commits sample via `argmax(l/T + gumbel)` — softmax-free too.
  Parity-gated token-exact against the reference fixtures. **Measured: 149/153/147 s
  (PASS 2.2%) ≈ −3.5%, peak memory 6.68 → 6.30 GB.**
  (b) **`asyncEval` pipelining** (engine `pipelined:` flag + bench `--pipelined` arm):
  **André's occupancy challenge confirmed** — the blocking per-step eval WAS paying real
  launch/idle overhead. Measured: 201/138/136 s; steady state **136–138 s ≈ −9% vs the
  eval arm** (~0.8 s/step recovered). Run 0's 201 s is a first-run artefact (kernel
  compilation racing the piled-up dispatch queue) — pipelined arms need a warmup run
  before timing (bench TODO). Combined (a)+(b): recipe arm 155 → ~137 s, **0.47 tok/s**,
  and the occupancy story revises to: part of the "compute-bound" gap was dispatch
  overhead after all; the batching dismissal stands only for the *remaining* gap.
- **Open** (recorded, not yet built):
  (c) **gate+up (and Q/K/V) load-time weight concat** — one GEMM instead of two/three;
  valid under row-wise affine quantization (scales/biases stack with output rows).
  Estimated small single-digit %; touches loader + module wiring.
  (d) **SwiGLU-chain fusion via MLX `compile()`** — try graph compilation (automatic
  elementwise fusion, shapes are step-invariant) before any hand-written fused kernel;
  could also subsume much of (b)'s launch overhead. The hand-fused Metal `swiglu` from
  metal-shader-guide §2.5 stays behind the profiling gate.

### S3.2 Cheap wins (each measured against S3.1, kept only if they pay)

- **Gumbel-max ancestral sampling** already in from S1.4 — verify the claimed ~30% per-step memory saving (**inferred** in the guide; measure).
- **Never materialise the `[B,S,V]` posterior in fp32** — keep bf16 with fp32 only at the normaliser/reductions (respecting the reference's fp32-upcast-before-softmax at the *logits* stage, which is parity-relevant; the *posterior algebra* precision is ours to choose — re-run toy parity after).
- **Lazy-eval discipline**: one graph per step, single `eval` at step end; no per-op syncs (M5 house style).
- **Canvas-length sweep** (512/1024/2048): quality-vs-speed table for the bench README (anchored delimiter needs room — guide §8.4).

---

## 6. Phase S4 — Fused kernels + step-count optimisation

### S4.1 = sumi-M7' — Fused off-by-one attention: SUPERSEDED, kernel never needed

*(Original spec below this section is retained in git history only; the flash-with-sink
Metal kernel described by the guide was made obsolete before it was written.)* Resolution
path, all measured (details §4 Finding 2 and the logbook): eager → SDPA × sigmoid(row-LSE)
two-pass (algebraic identity; time-neutral, memory −40%) → **MLX native SDPA `sinks:`
(one zero-logit sink per head ≡ `softmax_one`): 22.7 ms/layer ≡ plain SDPA, 0.03 GB** —
the theoretical floor, reached with zero Metal. The two-pass form is retained as
fallback/oracle (`attendFastLSE`).

### Metal / mega-kernel verdict (2026-07-09, requested by André — full grounds in the logbook §5)

**No hand-written Metal kernel or whole-step mega-kernel is warranted for Sumi. Estimated
total remaining ceiling for ALL kernel work: ~4–8% of step time (~0.3–0.7 s of 8.3 s),
most of it already claimed by `asyncEval` (−9%, landed) and cheaply claimable by MLX
`compile()`.** Measured grounds: (1) the step is quantized-GEMM-bound (~14.6 TFLOP at
canvas 1024) and the fusion micro-bench showed those GEMMs carry zero recoverable launch
overhead; (2) attention already sits at the fused-SDPA floor via native sinks — the planned
S4.1 kernel would at best tie it; (3) dispatch/idle overhead (~0.8 s/step) is what a
mega-kernel would remove, and pipelining already removes it. **Phase-1 §13.1 escape-hatch
trigger (loop overhead >10–15% after fusion): not met — the hatch stays closed.** The
needle-movers are step count (early exit, 2.6×), k (4×), hardware (Studio ~5.5×), and
MLX-upstream qmm improvements — none of them app-side kernels.

### S4.2 status (2026-07-09): SchED-style early exit LANDED — 2.6× on ancestral

Implemented in `SumiEngine` (`EarlyExit{stableSteps, minSteps, stableFraction}`, off by
default; bench flags `--early-exit/--min-steps/--stable-fraction/--denoise-end`): the
convergence signal is argmax-prediction stability over the **active window**, checked with
one Bool readback per step; on exit the window takes a terminal greedy commit of the stable
argmax (required for ancestral — its sampled canvas carries posterior noise mid-schedule;
a no-op for greedy at a fixed point, which is what `SumiEarlyExitTests` gates). Findings,
in the order they were learned (all measured, M1, canvas 1024, budget 64):

1. Whole-canvas stability **never fires** — the random tail churns forever. The signal
   must watch the content window (`denoise_end` scoping).
2. Exact stability (`stableFraction: 1.0`) **never fires either** — real Sumi keeps
   flipping 1–4 content tokens indefinitely (matches the revision-churn finding). SchED
   must be fractional here.
3. **`stableFraction: 0.9`, R=3, ancestral t=0.7: exits at 26/64 steps — 230 s vs 594–659 s
   baseline (2.6×), 0.278 tok/s, correct clean answer.** The reference-default sampler now
   approaches recipe-arm throughput at its own quality level.
4. Greedy + `denoise_end` is a known-bad combo: the permanently-random frozen tail
   degenerates its output ("- the -" loops) and it never exits. Greedy stays a debug
   sampler; do not ship it with a scoped window.

**Companion results (same battery)**:
- **S3.2(c) projection fusion: measured DEAD** (`SumiProjectionFusionBench`): separate ≡
  row-concatenated quantized GEMMs within 0.2% at both QKV and gate+up shapes (equivalence
  asserted). MLX's qmm is launch-overhead-free at these sizes — do not build the loader
  concat. Negative result recorded per house rules.
- **LM-head 4-bit (S3.2/M8-style sweep item): adopt-candidate.**
  `SumiDiffusionModel.quantizeLMHead()` (post-load, in-memory; bench `--quantize-lm-head`):
  **8.3 s/step (133 s recipe arm, best number yet, PASS 0.0% variance), peak 5.75 GB
  (−0.55 GB), 0 confident-position top-1 flips** at p_max ≥ 0.3/0.5/0.7, clean generations.
  The naive all-positions flip rate reads 93% — flat random-tail distributions flip freely
  and harmlessly; `SumiLMHeadPrecisionTests` documents the metric lesson. **Gate upgraded
  2026-07-09 (André's margin critique)**: the p_max-threshold gate was itself
  pooled-metric-vulnerable; the adoption gate is now **margin-conditioned** on real
  mid-generation canvases — measured: zero flips at top-1/top-2 margin ≥ 0.10 across all
  bins, largest flipped margin 0.0890 ≈ p99 margin-perturbation 0.0894 (flips confined to
  below the noise scale — calibrated-benign), adaptive k=4 committed tokens identical at
  jointly-selected positions; selection sets tie-break-diverge on 2/4 canvases (trajectory
  identity not on offer). Formal adoption still pending a task-level quality run (logbook
  §3.3 for full numbers).

**M1 state of play after S4.2**: recipe-adaptive-k4 at ~133 s / 0.48 tok/s (with 4-bit
head), ancestral-with-early-exit at ~230 s / 0.28 tok/s; Studio extrapolation ~1.5–2.5
tok/s. Remaining levers are second-order (MLX `compile()`, ancestral's residual sampler
traffic, thermal-stable benching on a desktop).

### S4.2 original spec (SUPERSEDED by the status block above — kept for the stretch list)

The paper-claimed 3.8–4× transferred to uniform-state as a measured **2.6×** on ancestral,
via the fractional-stability adaptation (see status block). Remaining stretch items, each
behind its own bench arm when picked up: τ-leaping-informed schedule (med), Attn-Sampler
commit order (med), linear temperature schedule (med, half-day), MLX `compile()` sweep.

**Accept (phase)**: bench table before/after each item; negative results recorded in this file per house rules. **Met** — see the logbook §4 for the full before/after table including the recorded negatives (projection fusion, greedy+window, exact-stability exit).

---

## 7. Studio-deferred checklist (queue for office return)

In priority order — the first two are the *pre-existing* blockers, unchanged by this plan:

1. LLaDA2.1-mini M4 BF16 real-weight forward parity (`generate_core_fixtures.py --dtype bfloat16` + `NEODIFFUSION_FIXTURE_DIR` run).
2. LLaDA2.1-mini M5 BF16 loop parity (`generate_loop_fixtures.py` + `NEODIFFUSION_LOOP_FIXTURE_DIR` run).
3. Sumi BF16 fixtures (real weights): full-forward (S1.3) + greedy generation traces (S1.5) → run the same config-driven tests.
4. Sumi `sumi-M6'` gate: BF16 greedy reference completions → 4-bit Levenshtein ≤3 check.
5. Sumi BF16 bench numbers at canvas 2048 (Studio) alongside the M1 4-bit numbers.

---

## 8. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Conversion OOMs on the 16 GB M1 (S2.1) | med | streaming per-shard fallback; worst case defer S2 to Studio, S1/S3-toy/S4-kernel still proceed |
| Tokenizer is a custom fork, not cl100k (guide flag #1) | low | resolved day one in S0.1 before any dependent code |
| Toy-parity green but real-weight red (numerical issue only visible at scale) | low-med | same exposure Phase 2 accepted for LLaDA; the sink-fp32 and outlier-logit tests in S1.2/S4.1 target the known suspects |
| `SamplingPolicy` extraction (S0.2-5) balloons | med | timeboxed; documented fallback keeps Sumi unblocked |
| M1 TPS lands ≪ the speculative 2–5 tok/s (guide flag #8) | unknown | S3.1 measures before S4 invests; SchED (S4.2) is the cheaper lever than Metal if steps, not step-cost, dominate |
| Sumi quality disappoints on commonsense tasks (guide §10) | n/a to engine | known model property; the port's value is the uniform-state + off-by-one machinery and the M1 iteration loop, not the model's benchmark scores |

---

## 9. Order of work, one line each

S0.1 fetch + decisions → S0.2 refactor prefix (tests green throughout) → S1.1 config/tokenizer → S1.2 off-by-one attention → S1.3 stack + forward parity → S1.4 samplers → S1.5 end-to-end greedy parity (**basic inference**) → S2 4-bit real weights on M1 → S3 bench baseline + cheap wins → S4.1 fused off-by-one attention (**optimised kernels**) → S4.2 SchED + schedule stretch goals → §7 Studio queue on return.
