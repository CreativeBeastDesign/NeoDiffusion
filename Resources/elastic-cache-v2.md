# Elastic-Cache: Attention Is All You Need for KV Cache in Diffusion LLMs

**Type**: paper
**Canonical source**: 01-Inbox/KV Cache Diffusion.pdf
**Date**: 2025-12-28 (arXiv, v2)
**Relevance**: Upgraded Elastic-Cache with three key observations: distant MASK tokens can be cached block-wise, KV dynamics increase with depth (depth-aware refresh), and most-attended token has smallest drift. 45.1× speedup on long sequences, 8.7× on GSM8K. More comprehensive than earlier version.
**Status**: processed
**Last updated**: 2026-04-16

## Summary

Elastic-Cache is a training-free, architecture-agnostic strategy for adaptive KV cache management in diffusion LLMs. This version (v2) makes three key observations that improve upon earlier versions:

1. Distant MASK tokens primarily act as length-bias and can be cached block-wise beyond active prediction window.
2. KV dynamics increase with depth - selective refresh starting from deeper layers is sufficient.
3. The most-attended token exhibits smallest KV drift, providing a conservative lower bound on cache change.

Elastic-Cache jointly decides when to refresh (via attention-aware drift test) and where to refresh (via depth-aware schedule).

## Key claims

- 8.7× speedup on GSM8K (256 tokens).
- 45.1× speedup on longer sequences.
- Consistently maintains higher accuracy than baseline.
- 6.8× higher throughput than confidence-based approaches.
- Training-free, architecture-agnostic.
- Works on LLaDA-Instruct, LLaDA-1.5, and LLaDA-V.

## Mechanisms

### Observation 1: Distant MASK tokens as length-bias

- Distant MASK tokens don't contribute meaningful semantic information.
- They primarily serve as a length indicator / padding.
- Can be cached block-wise and reused across steps.
- Reduces computation on redundant future tokens.

### Observation 2: KV dynamics increase with depth

- Shallow layers stabilize quickly (encode local lexical structure).
- Deep layers continue adjusting (encode global semantic dependencies).
- Selective refresh from deeper layers is sufficient.
- Shallow layer caches can be reused across more steps.

### Observation 3: Most-attended token has smallest drift

- Tokens receiving highest attention are most stable.
- Their KV drift provides conservative lower bound for other tokens.
- If most-attended token is stable, others likely stable too.
- Can use single drift test instead of per-token testing.

### Attention-aware Drift Test

1. Identify most-attended token (by cumulative attention weight).
2. Compute KV drift for this token between consecutive steps.
3. If drift < threshold, cache is still valid; skip recomputation.
4. If drift >= threshold, recompute KV for all tokens.

### Depth-aware Refresh Schedule

1. Determine layer ℓ* to start refresh from (typically deeper layers).
2. For layers < ℓ*: reuse cached KV (no recomputation).
3. For layers >= ℓ*: recompute KV (fresh computation).
4. ℓ* can be tuned per model or per step.

## Hardware implications

- **Compute reduction**: Selective layer refresh avoids redundant computation.
- **Memory bandwidth**: Cached shallow layer KV reduces memory traffic.
- **Attention extraction**: Need to extract attention weights for drift test.
- **Block-wise MASK caching**: Reduces future token computation.

## Relevance to Apple Silicon

- Training-free = immediate applicability.
- Three observations provide clear implementation targets.
- 45.1× speedup on long sequences is highly relevant for Apple Silicon's memory constraints.
- Depth-aware schedule aligns with Metal's threadgroup organization.
- Could combine with FreeCache's block-wise freezing.

## Extracted concepts

- [[attention-aware-drift-test]]
- [[depth-aware-refresh]]
- [[most-attended-drift]]
- [[block-wise-mask-caching]]
- [[layer-wise-kv-dynamics]]
- [[selective-layer-refresh]]

## Open questions

- What is the optimal ℓ* (refresh start layer) for different model sizes on Apple Silicon?
- Can the most-attended token heuristic be replaced with learned importance scoring?
- How does depth-aware refresh interact with per-block processing (LLaDA)?
- What is the Metal kernel design for efficient attention extraction and drift computation?
- Can this be combined with MaskKV's mask-query attention?

## Related pages

- [[elastic-cache]] (earlier version concept)
- [[freecache]] (FlashDLM FreeCache)
- [[maskkv]] (attention-based importance)
- [[llada2-1-tech-report]] (base model)