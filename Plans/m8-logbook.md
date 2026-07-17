# M8 Quantization-Sweep Logbook — LLaDA2.1-mini, M1-scoped

**Period**: 2026-07-10 → (ongoing)
**Scope**: the M1-runnable subset of phase-2 §4 M8, per the plan agreed with André
(phase-2 M8 "M1-scoped plan", items 1–5). Studio-deferred: BF16 interactive baseline,
scored-set comparison at speed, final shipped-default call if M1 evidence is ambiguous.
**Method**: continues the M6 campaign discipline (`m6-logbook.md` §0) — provenance labels
on every claim, one axis per experiment, negative results recorded. **All conclusions
from `envValid` JSONL rows only** (m6 §5 rule 1); cold-start excluded via the
`warmupIncluded` flag; immediate same-process reruns as contamination checks.
**Hardware**: dev M1 16 GB, mlx-swift 0.31.6, release builds.
**Raw records**: `scratch/llada_bench.jsonl`, `scratch/m8_*.log`, corpus + reference
fixtures under `scratch/m8_drift/`.

## Planned experiments (designed before measuring)

| # | Item | Design | Verdict criterion |
|---|---|---|---|
| E1 | Trajectory diagnostics (plan item 4) | engine `Output.metrics` extension: per-step Γ/Δ counts, mean transferred-token confidence, EOS block index — in-graph, read from the existing eval batch (sync-neutral) | full suite stays green; speculation invariance unaffected |
| E2 | Drift corpus (plan item 2) | ~24 real mid-generation windows (4 prompts × blocks × steps) from the 4-bit engine, cache-disabled, with FP32 logits of the active block | corpus is *real canvases* (Sumi §3.3 rule), fixed and versioned |
| E3 | Streamed-BF16 reference | reference decoder classes, layer-streamed (corpus-batched: all windows through layer *i*, then free ⇒ 33 GB read once total), strict mask | reference logits for E2's windows; sanity: BF16 top-1 self-agreement with 4-bit ≥ the M4 toy-scale expectation is NOT assumed — measured |
| E4 | Drift metrics | top-1 agreement, top-k overlap (k=5), max/mean |Δp|, **confidence-margin changes** (the quantity Γ/Δ thresholds consume), margin-conditioned flip bins per Sumi §3.3 | recorded, not thresholded (M4 gate wording) |
| E5 | lm_head 4-bit axis | in-memory `MLXNN.quantize` post-load; margin-conditioned flips on E2 corpus + bench arm + E1 trajectory comparison | adopt-candidate iff flips confined below measured noise scale AND bench-neutral |
| E6 | group-32 routed experts | conversion variant; drift + bench vs frozen g64 | quality gain vs +size; recorded |
| E7 | 6-bit routed experts | conversion variant; **memory risk**: ~13 GB peak on 16 GB — envValid rule decides measurability | may be unmeasurable on M1; that is itself a finding |
| E8 | Blind mask scoring (plan item 3) | paired fixed-seed strict/referenceBias outputs, shuffled labels → scoring sheet for André + scripted checks (code executability, EOS/degeneration) | promotes the M6 smoke-level claim; André scores |

## Timeline

| # | What | How tested | Result / note |
|---|---|---|---|
| 1 | E1 trajectory diagnostics landed: `Output.metrics` gains `transfersPerStep`/`editsPerStep`/`meanTransferConfidencePerStep` (in-graph `[3]` stats per step, read from the existing eval batch — sync-neutral) + `eosBlockIndex`; bench rows gain `transfersTotal`/`editsTotal`/`meanTransferConfidence`/`eosBlockIndex` | full suite 66 tests green; M5 parity + speculation invariance unaffected | ✅ |
| 2 | E2 drift corpus dumped: `NEODIFFUSION_M8_CORPUS=1 swift test --filter LLaDADriftCorpusDumper` — 4 prompts (chat/reason/code/essay), sampling steps {0, 4, every 16th} per block | 44 windows, 443 MB (`scratch/m8_drift/corpus.safetensors` + manifest) | real mid-generation canvases incl. masked positions (Sumi §3.3 rule) |
| 3 | E3 streamed-BF16 reference (`python3 -u Tools/m8_reference_logits.py`): reference decoder classes via synthetic-package import (fixture-dumper technique), meta-device + `assign=True` per layer (avoids ~13 GB double-allocation), eager attention, fp32 CPU, corpus-batched — checkpoint read once | ~8 s/layer, no missing-weight warnings (meta-leftover assert armed) | three false starts recorded: relative-import ImportError; `_attn_implementation=None` KeyError (fix: force eager); python stdout block-buffering hid progress (fix: `-u` — Sumi quirk 2, python form) |
| 4 | E5 prep: `DiffusionModel.quantizeLMHead()` + bench `--quantize-lm-head`; `LLaDAVariantLogitsDumper` (same-window variant logits — isolates logit drift from trajectory divergence); `m8_drift_metrics.py` `--candidate-file/--reference-file` for variant-vs-variant comparisons; E8 prep: `--blind-out` full-text pairs + `Tools/m8_blind_sheet.py` (deterministic per-prompt shuffle, key withheld, scripted degeneration checks) | builds green | |

| 5 | E3 complete: 44 reference logit sets (885 MB), 20 layers × ~8 s | `scratch/m8_reference.log` | |
| 6 | E4 first metrics (`m8_drift_metrics.py`, `scratch/m8_drift_baseline_metrics.log`): top-1 80.7% aggregate, flips@margin≥0.20 = 21/813 (2.6%), largest flipped margin 0.68 | 4-bit corpus logits vs streamed BF16 | **NOT obviously benign — but the comparison itself is unvalidated**; discriminator + slicing launched before drawing any conclusion (canvas-probe lesson) |
| 7 | Slicing (`m8_drift_slices.py`, `scratch/m8_drift_slices_baseline.log`): step-0 fully-masked windows 74.0% agree / 5.2% confident flips; **late-step windows 94.4% / 0.8%**; by prompt: chat 91.9%, reason 85.3%, essay 77.9%, code 78.1% (6.1% confident flips — worst) | population slicing per Sumi §3.2 | the aggregate is dominated by max-entropy masked positions; the decoding-relevant (late-step) population looks far closer to benign; code is the weakest regime |
| 8 | Pipeline-fidelity discriminator launched: same python pipeline, weights = **dequantized 4-bit artefact** (`--dequant-artefact`, manual U8 nibble dequant) → compare against the engine's own corpus logits | in flight | if pipeline≈engine here, the BF16 gap is real quantization drift; if not, my streamed reference has an implementation bug |

| 9 | Pipeline-fidelity discriminator result (`scratch/m8_fidelity_check.log`): same-weights comparison (python-dequant-4bit vs engine) → top-1 95.5%, |Δmargin| p99 **0.115**, **0/809 flips at margin ≥ 0.20**, largest flipped margin 0.159 | methodology noise floor measured | flips ≥ 0.20 are attributable signal; below ~0.16 unattributable |
| 10 | Pure-quantization comparison (`scratch/m8_pure_quant_drift.log`): dequant-4bit vs BF16 through the *identical* pipeline → top-1 80.5%, **22/813 flips at margin ≥ 0.20 (2.7%)**, largest flipped margin 0.85 — statistically identical to engine-vs-BF16 (80.7%, 21/813) | zero-implementation-confound comparison | → Finding M8-1 |
| 11 | E5 lm_head axis (`scratch/m8_headq4_vs_f16head.log`): head-q4 vs F16-head, same 4-bit body, engine both sides (zero pipeline confound) → flips@≥0.20 = 3/811, **largest flipped margin 0.32 > noise p99 0.23**, 11% flips in [0.10, 0.20) | margin-conditioned per Sumi §3.3 | → Finding M8-2 (REJECT) |
| 12 | Infrastructure incident: `testQuantizedArtefactRoundTrip` began segfaulting (memmove null in `loadWeights`; earlier run died with a *swiftinterface* "Not enough bits" fatal) after a burst of API-signature changes. Two-pass-quantize revert did NOT fix ⇒ not the new code path; `swift package clean` then produced a build with **no Cmlx metallib bundle at all** (87 s "full" rebuild — impossible for cold C++) | manual bisect + crash-report stack + build forensics | **root cause found**: `.build/build.db` (SwiftPM's llbuild state) survived both `swift package clean` and an `rm -rf .build/arm64-apple-macosx` — llbuild kept trusting stale task states, silently skipping compilation, linking mismatched objects (→ segfault), and never re-running the native `.metal` compilation that produces `mlx-swift_Cmlx.bundle`. **Quirk for the list: partial `.build` deletion is worse than none — remove `build.db` (and plugin state) together with the artifacts, or the next builds are silently corrupt.** Coherent rebuild in flight. (Also learned: mlx-swift 0.31.6 metallib comes from SwiftPM's *native* `.metal` compilation of `Source/Cmlx/mlx-generated/metal/` — no plugin involved; Xcode DerivedData carries its own copies) |

| 13 | Incident resolved + real bug found by the honest rebuild. (a) **Toolchain fact**: Swift 6.3.3's `swift build` does **not** compile mlx-swift 0.31.6's `.metal` sources — no Cmlx bundle is ever produced by CLI builds on this toolchain; the week's working bundles originated from Xcode builds (DerivedData) and survived in `.build` until deleted. Runtime search: executables want `mlx-swift_Cmlx.bundle` **next to the binary**; XCTest wants it **inside `NeoDiffusionPackageTests.xctest/Contents/Resources/`**. Seeding from DerivedData restores both (bench README updated). (b) **MLXNN constraint found by test**: `quantize` as a *second pass* touching only MoE layers is a sparse `update(modules:)` on the `layers` array → `unexpectedStructure` fatal. Fix: single pass via the per-module-params `quantize(model:filter:)` overload (filter returns `(groupSize, bits, mode)?`). Full suite green after: 68 tests, 15 skipped, 0 failures | rebuild + `testModelQuantizationFilter` | two quirks recorded; `build.db` partial-deletion quirk stands (timeline 12) |
| 14 | E6 g32-experts conversion: `convert_weights_streaming.py --expert-group-size 32` → `models/llada2-1-mini-4bit-g32e` (10 GB, +0.5 GB scales; mixed shapes verified) | header spot-check | loader + config plumbing for mixed quant params landed with 13(b) |
| 15 | E6 result (`scratch/m8_g32e_vs_bf16.log`, `_slices.log`): g32-experts vs BF16 → top-1 80.4%, **flips@≥0.20 = 28/813 (3.4%)** vs g64's 22/813 (2.7%) — within binomial noise; margin-noise p99 narrows 0.70→0.50 but confident flips don't improve | same fixed corpus, same-window comparison | **negative result** → Finding M8-3: expert group resolution is not the drift driver; suspicion moves to bit depth (E7) or the non-expert quantized paths (QKV/dense FFN) |

| 16 | E8 pairs generated (12 prompts × 2 masks, uncached, gen-64, `scratch/m8_blind_generation.log`); sheet built (`m8_blind_sheet.py`): 12-pair blind sheet + sealed key + scripted checks — **no degeneration signal in either arm** (max 4-gram repeat = 1 everywhere, no mask leaks, matched lengths) | E8 harness | awaiting André's scores |
| 17 | E7: 6-bit-experts artefact converted (13 GB) and evaluated on the fixed corpus. **Loads and runs on the 16 GB M1** (variant dump succeeded; slow load, memory-tight — fine for offline evaluation, NOT a serving configuration). Result (`scratch/m8_e6_vs_bf16.log`): top-1 80.3% aggregate, **flips@≥0.20 = 25/813 (3.1%)** vs g64's 22/813 — within noise; late-step slice 94.8%/0.4% (small-n) | same-window comparison | **negative result** → Finding M8-4 |

## Findings

**Finding M8-1: 4-bit g64 whole-model quantization causes real, attributable logit
drift — ~2.7% of high-confidence positions flip; NOT the benign signature.** (sourced:
timeline 6/9/10 — the three-way comparison design.) Method: the pipeline-fidelity
discriminator (same weights both sides) measured the methodology's noise floor at
|Δmargin| p99 0.115 with **zero** flips at margin ≥ 0.20; the BF16 comparison shows 21–22
such flips (2.6–2.7%, identical whether the 4-bit side is the engine or the same python
pipeline — the engine adds nothing). Largest flipped margin 0.85. Population structure
(timeline 7): late-step windows 94.4% top-1 agreement / 0.8% confident flips; **step-0
fully-masked windows are the most quantization-sensitive population (74.0% / 5.2%)** —
exactly the states where Γ selection decisions are made; code prompts weakest (78%).
Context: gotcha 5 always scoped 4-bit to task-level quality (M6 outputs were correct and
fluent), so this is a *measured characterization*, not a regression — and it gives the
E6/E7 sweep axes a sharp recovery target. Corollary for M8's final call: the shipped
default should likely spend bits on the **routed experts** (dominant parameter mass),
pending E6/E7.

**Finding M8-2: lm_head 4-bit REJECTED — fails the margin-calibrated benign signature
on LLaDA (opposite of the Sumi result).** (sourced: timeline 11, zero-confound design —
same body, only the head differs.) Largest flipped margin 0.32 **exceeds** the noise
scale (|Δmargin| p99 0.23); 3 confident (≥0.20) flips; 11% flips in the [0.10, 0.20)
bin. Sumi's head passed the same gate (0 flips ≥ 0.10, largest 0.089 ≈ noise 0.089,
sumi-logbook §3.3); LLaDA's does not — plausibly (**speculative**) because the diffusion
confidence machinery consumes head logits at *masked* positions where calibration is
most fragile. The 0.55 GB saving does not justify it; `quantizeLMHead` stays available
as a bench flag for future re-testing. Same test, two models, opposite verdicts —
the "measure per model, never port quantization conclusions" rule is now twice-proven.

**Finding M8-3: doubling expert group resolution (g64→g32) does NOT reduce quantization
drift — negative result, do not spend the 0.5 GB.** (sourced: timeline 14–15, same-window
comparison on the fixed corpus.) Confident flips 28/813 vs g64's 22/813 (within binomial
noise); aggregate top-1 unchanged (80.4% vs 80.5%); only the sub-threshold margin noise
narrows (p99 0.70→0.50). Interpretation (**inferred**): the confident-flip drift is not
dominated by expert weight-grid resolution at these group sizes — remaining suspects are
expert *bit depth* (E7 tests this directly) and the non-expert quantized paths (QKV,
attention dense, dense FFN). If E7 also fails to move the flips, the sweep's answer is
"the g64 artefact is already on the pareto knee; spend nothing" — itself a valid M8
outcome (recorded either way).

**Finding M8-4: expert bit depth (4→6 bits) does not reduce confident-flip drift either —
the routed experts are NOT the drift driver, and the sweep's answer is: ship the uniform
4-bit g64 artefact unchanged.** (sourced: timeline 17; with Finding M8-3.) Despite ~2.3×
the expert-weight precision (+3.4 GB), confident flips stay at 25/813 vs 22/813 baseline.
Combined with E5 (head REJECT) and E6 (g32 no-gain), none of the three pre-registered
axes purchases confident-position agreement. Interpretation (**inferred**): the drift is
carried by the remaining quantized paths (fused QKV, attention dense, layer-0 dense FFN)
and/or is distributed across all 4-bit tensors such that no single-axis upgrade moves it.
Consequence: **M8's shipped default = the existing g64 artefact** — the quality bar
remains the task-level one it already passes (M6 outputs, E8 checks clean), and the
~2.7% confident-flip drift is a *documented property*, not a defect to buy back with the
swept axes. Sharp follow-ups if ever needed (out of pre-registered scope, not started):
a QKV/dense-only 6-bit axis to test the remaining-paths hypothesis; Studio BF16 arms for
task-level scoring at speed.

## Results summary

All vs the layer-streamed BF16 reference on the fixed 44-window / 1 408-position corpus
of real mid-generation canvases (`scratch/m8_drift/`), except where noted. "Confident
flips" = top-1 flips at reference margin ≥ 0.20 — the only bin above the measured
methodology noise floor (fidelity check: 0/809 at ≥ 0.20, |Δmargin| p99 0.115).

| Candidate | top-1 (aggregate) | top-1 (late-step) | confident flips | max flipped margin | Δ size | verdict |
|---|---|---|---|---|---|---|
| 4-bit g64 (shipped) | 80.7% | 94.4% | 21/813 (2.6%) | 0.68 | — | baseline; drift real but task-level quality good (M6) |
| pure-quant control (same pipeline) | 80.5% | — | 22/813 (2.7%) | 0.85 | — | confirms drift = quantization, not engine/pipeline |
| + lm_head 4-bit (E5)* | — | — | 3/811 vs F16-head | 0.32 > noise 0.23 | −0.55 GB | **REJECT** (fails margin-calibrated gate) |
| g32 experts (E6) | 80.4% | 94.4% | 28/813 (3.4%) | 0.85 | +0.5 GB | **REJECT** (no improvement — Finding M8-3) |
| 6-bit experts (E7) | 80.3% | 94.8% | 25/813 (3.1%) | 0.68 | +3.4 GB (13 GB total; loads on M1, not servable there) | **REJECT** (no improvement — Finding M8-4) |

**Sweep verdict: shipped default stays the uniform 4-bit g64 artefact** (Finding M8-4).

*E5 measured head-vs-head (same 4-bit body both sides), not vs BF16.

E8 (blind mask scoring): pairs generated (12 prompts × strict/referenceBias, uncached,
gen-64); scripted checks clean in both arms (no repetition/mask-leak/length anomalies).
**SCORED (André, 2026-07-17): 10/12 ties, 1 strict win (`reason-logic`), 1 referenceBias
win (`code-regex`) — a wash** (`blinds/m8_blind/`, scored against its own `key.json`).
Verdict: no detectable task-level quality difference between the mask semantics at 4-bit;
**no basis to switch, `.strict` stays the incumbent default** and the M6 smoke-level
qualitative claim is neither promoted nor contradicted. E8 CLOSED; this was M8's last
open item.
