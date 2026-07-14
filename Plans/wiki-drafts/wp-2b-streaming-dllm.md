# WP-2b — Streaming-dLLM cluster on LLaDA2.1-mini: dynamic τ(t) is the first un-banked win; suffix pruning and EOS early exit are pre-banked

**Summary**: Fourth NeoDiffusion Phase-3 work package: the three [[streaming-dllm]] modules taken separately. [[dynamic-confidence-aware-decoding]] (τ(t) = τ0·(1 − α(1 − r_mask))) is the first Phase-3 candidate whose lever the engine's baseline had *not* already banked — α=0.6 cuts logical steps −14.8% (chat) / −11.3% (reasoning) at Q mode, and stacks additively with MultiBD (+41%/+40% cumulative TPF-logical). [[attenuation-guided-suffix-modeling]] was closed **without implementation**: block-causal LLaDA2.x never computes a suffix (verified in the reference source: it forwards `x[:, :current_window_end]`), so there is nothing to prune and no suffix signal for the positional cue to preserve. [[early-exit-block-diffusion]] was implemented and is a measured **null**: post-EOS padding positions clear τ_mask early while the boundary EOS settles last, so a settled EOS never has masked positions after it — threshold decoding pre-banks the paper's saving.
**Status**: closed on the dev host (2026-07-11); dynamic τ landed default-off (provisional algorithmic accept), EOS early exit landed default-off (null), suffix module not built
**Type**: experiment (05-Experiments)
**Engine**: branch `wp-2b-streaming`; record `Plans/wp2b-logbook.md`
**Sources**: [[streaming-dllm]] (arXiv:2601.17917) → [[attenuation-guided-suffix-modeling]], [[dynamic-confidence-aware-decoding]], [[early-exit-block-diffusion]]

---

## What was built

- **Dynamic τ (2b-2)**: per-slot in-graph τ(t) over the slot's generated positions, α=0 ⇒ byte-identical parity; gated by cached==uncached identity, K-invariance, and a synthetic-forward liveness proof (17→7 steps). Note: the formula *loosens* the threshold as the block fills — the source note's prose ("tightens") is wrong; the formula matches the paper's motivation.
- **EOS early exit (2b-3)**: in-graph fill of still-masked positions after the first settled non-prompt EOS in the front block; trim-invariant by construction; synthetic-forward proof (17→2 steps). Trigger is Γ/Δ-settlement itself (no separate τ_eos — condition for adding one was a nonzero text-change rate, measured 0/8).
- **Suffix module (2b-1)**: nothing — the closure argument is the deliverable.

## Results (Mac Studio M2 Ultra Backfill, 2026-07-14)

- **Dynamic τ (2b-2)**:
  Evaluated on the target Studio hardware as part of the Phase 3 Combination (`p3-combo`), yielding a −14.8% (chat) and −11.3% (reasoning) step count reduction. However, the dual-active step evaluation latency (due to sequence/active-window length doubling) causes a net wall-clock TPS loss of −12.3% to −20.4%.
- **Composition with Credit Decoding (`p4-combo`: MultiBD + Dynamic-τ + Credit)**:
  Exposing Credit Decoding jointly with MultiBD + Dynamic-τ recovers serving performance on chat, reaching **45.55 TPS** (only −5.5% behind baseline).
- **Suffix Pruning (2b-1) and EOS Early Exit (2b-3)**:
  Suffix pruning is closed N/A (block-causal LLaDA2.x never computes or attends to a suffix). EOS early exit is verified as a complete mechanistic null, as threshold decoding already commits/caches post-EOS positions early in standard execution.
- **Verdict (ACCEPTED for serving presets only, default off)**:
  Dynamic τ (α=0.6) is accepted as a serving preset option (available within the `p3-combo` preset) for step-limited long-form generations, but is disabled by default for generic real-time serving.


