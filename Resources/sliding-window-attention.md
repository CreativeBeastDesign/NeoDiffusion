# Sliding-Window Attention in dLLMs

**Summary**: An attention-restriction technique that limits computation to a fixed-length window of nearby tokens, dropping distant tokens from the attention computation entirely. Appears independently in DPad (suffix-side) and Elastic-Cache (active-prediction-side), with different framing in each.
**Aliases**: fixed-window attention, windowed decoding
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[dpad]]
- [[elastic-cache]]

---

## Definition

Sliding-window attention restricts attention computation to a fixed-length window of tokens near the current position, rather than the full sequence. It appears twice in this wiki with different roles:
- **DPad**: applies the window to *suffix* tokens (future, not-yet-generated positions) — only suffix tokens within the window participate in attention; farther ones are dropped (**sourced**, [[dpad]]). This is the mechanism underlying [[suffix-dropout]]'s "Sliding Window Variant."
- **Elastic-Cache**: applies the window to the *active prediction* set — restricts which MASK tokens are actively denoised at each step to a window of size β (default 16) closest to the left, caching distant MASK tokens block-wise instead of processing them (**sourced**, [[elastic-cache]]).

Both are "sliding" in the sense that the window moves with the current decoding frontier, but they window different things (future suffix vs. active MASK prediction set) for different purposes (redundancy reduction vs. compute/step reduction).

## Why it matters

Full-sequence bidirectional attention in dLLMs means every step pays O(L²)-ish attention cost over the whole sequence, most of which (per both papers) is spent on tokens contributing little useful signal at that step — either because they're low-entropy suffix "scratchpad" content (DPad) or because they're MASK tokens far from the current decoding frontier that mostly encode length-bias rather than content (Elastic-Cache; see [[elastic-cache-v2]]'s "Observation 1"). Windowing directly cuts this waste.

## Mechanism

General form: for position i attending outward, define a window of size W; only positions within distance W of i (or of the active frontier) participate in attention; positions outside are excluded from the current step's attention computation (**sourced**, [[dpad]]).

DPad additionally combines the hard window with a soft distance-decay dropout inside the window boundary (see [[suffix-dropout]]). Elastic-Cache instead caches (rather than drops) the excluded distant tokens block-wise, reincorporating them once they enter the window (**sourced**, [[elastic-cache]]) — see [[block-wise-mask-caching]] for the v2 refinement of this idea.

## Trade-offs

- **Window size**: larger windows preserve more context/accuracy but reduce speedup; both papers report task-dependent optimal window sizes (DPad's window is tuned per task; Elastic-Cache's β defaults to 16, sometimes 32, with β>32 shown to *degrade* accuracy by caching too many MASKs that later become relevant).
- **Drop vs. cache for excluded tokens**: DPad drops excluded suffix tokens outright (redundancy-reduction framing); Elastic-Cache caches excluded MASK tokens for later reuse (compute-reduction framing without discarding information). These are materially different design choices despite both being "sliding windows" — worth not conflating.
- **Task dependence**: neither paper's source note claims a universal optimal window size; both flag this as needing per-deployment tuning.

## Apple Silicon implications

- Fixed-size windows map naturally onto Metal threadgroups sized to the window rather than the full sequence — both source notes independently suggest this.
- **Inferred**: because Apple Silicon's L1/L2 cache sizes differ from the CUDA hardware both papers were evaluated on, the optimal window size is likely device-specific and would need re-tuning rather than being transferable as a fixed constant (flagged as an open question in both source notes, e.g. Elastic-Cache's "16 vs 32" question).
- The DPad-style precomputed, deterministic window mask is cheaper to implement than Elastic-Cache's adaptive attention-driven window boundary, since it requires no runtime decision — a useful complexity/speedup trade-off point when designing a first Metal kernel pass.

## Related concepts
- [[suffix-dropout]] (DPad's full technique, of which windowing is one component)
- [[local-attention-dllm]] (the broader category sliding-window attention is one instance of)
- [[elastic-cache]] (the caching-side use of windowing)
- [[block-wise-mask-caching]] (Elastic-Cache v2's refinement of caching distant windowed-out MASK tokens)

## Open questions
- Is there a task-independent way to choose window size, or is per-task tuning unavoidable?
- How does DPad's drop-based windowing compare directly to Elastic-Cache's cache-based windowing on the same benchmark?
- What is the optimal window size for Apple Silicon's L1/L2 cache and threadgroup sizing specifically?
