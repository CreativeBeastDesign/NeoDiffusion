# Adaptive KV Caching in dLLMs

**Summary**: Techniques that dynamically decide which tokens to cache vs recompute based on observed token stability, layer sensitivity, or attention patterns. Unlike static caching, adaptive methods adjust per-step based on model behavior.  
**Aliases**: selective KV caching, importance-based caching, dynamic cache management  
**Status**: emerging  
**Last updated**: 2026-04-16  
**Sources**:
- [[dllm-cache]]
- [[freedave]]
- [[d2cache-dual-adaptive-cache]]
- [[maskkv]]

---

## Definition

Adaptive KV caching encompasses a family of techniques that selectively cache or recompute Key-Value projections during dLLM inference based on runtime signals. The core insight is that not all tokens need recomputation at every denoising step; some tokens are "stable" and can have their KV cached and reused, while others are "dynamic" and need recomputation.

Methods differ in their selection criteria:
- **Feature similarity** (dLLM-Cache): Use cosine similarity between current and cached KV to detect stability.
- **Drift testing** (Elastic-Cache): Compare KV before/after with a threshold.
- **Mask-query attention** (MaskKV): Use mask tokens' attention to prompt tokens to score importance.
- **Layer-wise budgeting** (D2Cache, MaskKV): Allocate different cache budgets to different layers based on sensitivity.

## Why it matters

dLLMs have fundamentally different KV caching requirements than AR models:
- AR models: Cache everything; KV grows monotonically.
- dLLMs: Need selective caching because bidirectional attention means unmasked tokens can affect each other.

Adaptive caching can achieve 5-30× speedup while maintaining quality by avoiding redundant computation on stable tokens.

## Mechanism

### Selection criteria variations

1. **Token stability**: After a token is unmasked, its KV projections typically stabilize within 5-10 denoising steps. Once stable, the token can be cached.

2. **Layer sensitivity**: Different transformer layers have different sensitivity to cache compression:
   - Early layers: More stable, can cache aggressively.
   - Middle layers: More dynamic, need frequent updates.
   - Later layers: Critical for output quality, need careful caching.

3. **Attention-based importance**: Tokens that receive high attention from active (masked) tokens are more important to keep fresh in cache.

4. **Feature drift**: Tokens whose KV changes significantly between steps should be recomputed; those with minimal change can be cached.

### Cache management

When cache is full:
- Evict lowest-importance tokens (based on selection criteria).
- Recompute evicted tokens on next access or accept the approximation.

### Integration with dLLM architecture

Adaptive caching works because:
- dLLMs maintain mask tokens until positions are denoised.
- Mask tokens' queries reveal what's important at each step.
- Clean tokens become progressively more stable.

## Trade-offs

- **Selection overhead**: Computing selection criteria adds overhead; must be cheaper than saved computation.
- **Accuracy vs speed**: More aggressive caching = more speed, but potential quality loss.
- **Threshold tuning**: Many methods require threshold hyperparameters that may be dataset-dependent.

## Apple Silicon implications

- Selection criteria computation (cosine similarity, attention extraction) adds kernel overhead.
- Could be fused into existing attention kernels to minimize overhead.
- Layer-wise budgeting aligns with Metal's threadgroup memory organization.
- Memory reduction valuable for unified memory efficiency.

## Related concepts

- [[elastic-cache]] (drift test-based caching)
- [[freecache]] (block-wise freezing)
- [[mask-query-attention]] (attention-based selection)
- [[layer-wise-caching]] (per-layer budgeting)
- [[token-stability]] (stability as selection signal)

## Open questions

- What is the optimal selection criteria for Apple Silicon? Feature similarity vs attention scoring vs drift test?
- Can selection be done without additional compute passes?
- How do different selection methods compare on Apple Silicon benchmarks?
- Could selection be learned (trained) rather than heuristic?

## Related pages

- [[dllm-cache]] (source)
- [[d2cache-dual-adaptive-cache]] (source)
- [[maskkv]] (source)
- [[elastic-cache]] (related approach)