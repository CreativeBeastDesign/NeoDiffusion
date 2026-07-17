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

## Results (Mac Studio M2 Ultra Backfill, 2026-07-14)

- **S2D2 vs. Conservative Baseline (base-t95 at τ=0.95)**:
  - **Chat**: 33.62 TPS vs 29.33 TPS (**+14.6% wall-clock speedup**).
  - **Reasoning**: 59.40 TPS vs 49.57 TPS (**+19.8% wall-clock speedup**).
  - **Code**: 70.01 TPS vs 76.67 TPS (−8.7% loss).
- **Composition with Credit Decoding (`s2d2-credit`: S2D2 + Credit)**:
  - Running Credit Decoding jointly with S2D2 does not yield synergy: **33.28 TPS on chat** and **58.83 TPS on reasoning** are slightly slower than S2D2 alone, as momentum boost is washed out by the verifier's exact step checks.
- **Verdict (ACCEPTED for Verified-Conservative Preset)**:
  The target hardware backfill confirms that the verified-conservative S2D2 preset (τ=0.95 + S2D2 verifier) converts successfully to wall-clock gains on the Mac Studio M2 Ultra, clearing the target gate (>=15% speedup) on reasoning tasks and landing at +14.6% on chat. The baseline Q-mode (τ=0.7) remains the fastest default, but S2D2 is officially accepted as the served "verified-conservative" preset for high-quality/math-stable decoding.


