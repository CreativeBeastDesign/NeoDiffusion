# WP-2a — Single-model speculation on LLaDA2.1-mini: S2D2 implemented, Spiffy ceiling-gated (mechanisms win, Q-mode economics don't)

**Summary**: Third NeoDiffusion Phase-3 work package: [[self-speculative-decoding-dlm]] (S2D2) implemented end-to-end behind a `SpeculationPolicy` seam; [[spiffy]] taken to a calibrated ceiling readout instead of a runtime. Both mechanisms work strikingly well — S2D2 accepts 4–10 tokens per verified step, Spiffy's calibrated formulas cover a 27–35% forward-count saving — but neither converts on a compute-bound host against Q-mode, and the reason is now an identity, not an observation: **verification pays iff acceptance per verified step exceeds 2× the baseline's tokens-per-step, and aggressive threshold decoding already banks that parallelism.**
**Status**: closed on the dev host (2026-07-11); S2D2 landed default-off; two decisions deferred to Studio
**Type**: experiment (05-Experiments)
**Engine**: branch `wp-2a-speculation`; record `Plans/wp2a-logbook.md`
**Sources**: [[s2d2-self-speculative-decoding]] (arXiv:2603.25702), [[spiffy]] / [[auto-speculative-decoding]] / [[directed-draft-graph]] / [[offline-draft-calibration]] (arXiv:2509.18085)

---

## What was built

- **S2D2, complete**: 2L-trick verifier (full-B static shapes for lazy K-batching), block-size-1 AR view proven exactly equal to sequential causal decoding (anchor test, logits <1e-3), vectorized greedy acceptance ≡ scalar reference, min-span routing at batch boundaries, all in-graph (sync budget unchanged). Cached path only; output legitimately differs from vanilla decoding.
- **Spiffy, up to the gate**: 50-prompt calibration suite (André's prompts, 4 categories incl. German), per-step per-position trace dumps, and the Alg-2 graph builder with a ceiling readout — no runtime, by design.

## The three results

1. **Against Q-mode (τ_M2T=0.7, the shipped default), single-model speculation loses on compute** at every routing point tried (τ_span × K grid): width-corrected TPF −35% to −65%. The break-even identity places Q-mode *just* out of reach on every suite (chat needs 4.6, gets 4.1; code needs 10.7, gets 9.8).
2. **Against the conservative baseline the papers actually use (τ≈0.95), S2D2 converts exactly as the identity predicts**: +19%/+15%/+5% width-corrected (chat/reasoning/code), steps halved. The candidate serving preset that survives is **verified-conservative** (τ=0.95 + S2D2): ~25% compute premium over Q-mode with paper-documented quality gains — pending blind scores and Studio wall-clock.
3. **Spiffy's ceiling is real and large on forward count** (27.6–35.2% of forwards skippable at D=3–8, from calibrated traces under genuine Q-mode decoding — refuting the guess that variable-Γ thresholding would zero it), but sequence-dim drafting pays (1+D)×width per step, unaffordable on a compute-bound host. GO/NO-GO moves to the Studio, graph shipped.

## The recurring Phase-3 lesson (third instance)

Papers sell savings relative to *their* baselines. NeoDiffusion's Phase-2 baseline already contains: an exact prefix cache (absorbed [[elastic-cache]], WP-1a), and aggressive threshold-parallel decoding (absorbs most of single-model speculation, this WP). Before porting any dLLM acceleration, compute what the engine's baseline already banks — the wiki's concept pages should carry a "baseline-relativity" warning field.

## By-products

- **[[just-on-time-jot]] entry condition measured** from the same traces: 44.2% of positions are prediction-stable ≥2 steps before unmask, only 4.9% later edited — a scoped JOT WP is justified (roadmap Tier-3 updated).
- Reusable: `--dump-traces` per-position instrumentation; `--threshold-mask/--threshold-edit` overrides; width-corrected TPF accounting in the bench; the generalized blind-sheet tool.

## Open items

- André: score `scratch/wp2a_blind/sheet.md` (qmode vs s2d2-t95, 12 prompts — decides the verified-conservative preset's quality half).
- Studio session: (a) S2D2 arms' wall-clock (does the 2B verifier cost ≪2× there?), (b) Spiffy go/no-go under the Studio's wide-forward multiplier, using `scratch/draft_graph.json` (hash 9eb8d6295c91).
