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

## The three results

1. **Dynamic τ accepts (provisional)**: winner α=0.6 (the paper's own optimum, transferring across τ0 and model family): chat 216→184 logical steps (TPF-logical 1.89→2.39), reasoning 97→86. Non-monotonic on chat — α=0.9's near-zero late floor buys acceptances that Δ must repair (post-steps/block 1.0→1.62). Mechanism confirmed: the converted steps are the late-block fallback-dominated tail. τ0=0.9 probe: static is dominated (+59% steps vs Q); per-step dynamics beat any single recalibrated τ0 — closes the `Optimisations.md` threshold-calibration consolidation in favor of τ(t).
2. **EOS early exit is a mechanistic null** (byte-identical rows, 0/8 text changes, also at nBuf=2): the paper assumes post-EOS positions linger; under threshold decoding they are high-confidence EOS predictions that unmask early, and commit-time eos_early_stop already cancels never-started blocks.
3. **Composability**: dynamic τ × MultiBD (nBuf=2, τ_add=0.5) is additive — −14.7% steps on top of MultiBD's gain (cumulative chat TPF 1.89→2.67). Matrix cells 1b×2b-2 (additive, confirmed) and 1b×2b-3 (null, no interference) close; 2a×2b-2 stays open (forbidden by precondition, untested).

## The recurring Phase-3 lesson (instances four and five — and the first counterexample)

2b-1 (suffix pruning pre-banked by block-causality) and 2b-3 (post-EOS savings pre-banked by threshold decoding) extend the WP-1a/WP-2a sequence. Dynamic τ is the counterexample that sharpens the rule: the un-banked residue is the *fallback-dominated late-block tail*, which no existing engine mechanism addresses — candidates whose lever lives there can still win.

## Open items

- André: score `scratch/wp2b_blind/sheet.md` (static vs α=0.6, 8 prompts) — decides the quality half of the default-on question.
- Studio session: scored-set quality gate + wall-clock for the recorded arms (`scratch/wp2b_driver.sh`); preset decision Q+α=0.6, optionally stacked with MultiBD per WP-1b's own Studio gate.
