# Block-Wise MASK Caching

**Summary**: Elastic-Cache's technique of caching distant, not-yet-actively-predicted MASK tokens in fixed-size blocks rather than recomputing them every step, based on the observation that they primarily encode length-bias rather than useful content signal.
**Aliases**: distant-MASK block caching
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[elastic-cache-v2]]
- [[elastic-cache]]

---

## Definition

Block-wise MASK caching is the practice of caching the KV states of MASK tokens that fall outside the active sliding-window prediction range in fixed-size blocks (block size 32 in the source material), reusing them without recomputation until they enter the active window (**sourced**, [[elastic-cache-v2]], [[elastic-cache]]). This is the same "Block-Caching of Distant MASK Tokens" mechanism already documented in the existing [[elastic-cache]] concept page; this page is a focused, atomic breakout of it as named explicitly in the v2 source note's extracted-concepts list.

## Why it matters

Elastic-Cache v2's "Observation 1" is the justification: distant MASK tokens don't contribute meaningful semantic information at the current step — they primarily serve as a length indicator/padding, since they haven't yet entered the active prediction window where they'd actually be denoised (**sourced**, [[elastic-cache-v2]]). Recomputing their KV every step is therefore mostly wasted work. Caching them block-wise instead directly reduces per-step compute on the majority of a long sequence's masked tail. The existing [[elastic-cache]] page reports this contributes materially to speedup: ablating block-caching reduces throughput by 30–40% with minimal accuracy impact (**sourced**, [[elastic-cache]]), confirming its importance is not merely incidental.

## Mechanism

1. Partition the sequence's not-yet-active MASK tokens (those outside the current sliding-window prediction range — see [[sliding-window-attention]]'s Elastic-Cache instantiation) into fixed-size blocks (block size 32).
2. Cache each block's KV states rather than recomputing them at every step.
3. When a block's tokens enter the active window (i.e. become part of the current step's prediction target), reuse the cached KV and switch to normal per-step processing for those tokens.

This is directly complementary to the sliding-window mechanism: the window defines which tokens are "active," and block-wise MASK caching is specifically what happens to the tokens *outside* that window (**sourced**, [[elastic-cache]]).

## Trade-offs

- **Block size**: fixed at 32 in the source material; the existing [[elastic-cache]] page doesn't report sensitivity analysis on this specific parameter separate from window size β.
- **Ablation-confirmed importance**: unlike some of Elastic-Cache's other components, this one has a directly reported ablation (30–40% throughput loss when removed) — making it one of the more strongly-evidenced individual components of the overall method, worth flagging as comparatively well-supported relative to, e.g., [[quasi-ltr-generation]] in the D2Cache material, which lacks a comparable ablation.

## Apple Silicon implications

- Block-wise, contiguous storage aligns naturally with Metal resource buffers — each block can be stored and reused as a contiguous unit, as already noted in [[elastic-cache]]'s Apple Silicon implications section.
- Reduces peak memory relative to caching all tokens' full intermediate states (18.13 GB vs 19.62 GB baseline reported in [[elastic-cache]] for a 512-token GSM8K run) — directly relevant for unified-memory-constrained on-device deployment.

## Related concepts
- [[elastic-cache]] (the full method, with the ablation data and memory numbers)
- [[sliding-window-attention]] (defines the active/inactive boundary this caching applies outside of)
- [[layer-wise-kv-dynamics]] (companion depth-based observation from the same v2 source)

## Open questions
- Is block size 32 optimal, or was it chosen for convenience relative to hardware (e.g. warp/threadgroup size) rather than derived from the length-bias observation itself?
- How does block-wise MASK caching interact with per-block FP8 quantization for further memory bandwidth reduction?
- Does the "primarily length-bias" characterization of distant MASK tokens hold for all task types, or could some tasks (e.g. long-range dependency reasoning) violate this assumption?
