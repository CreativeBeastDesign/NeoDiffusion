# MaskKV: Mask Tokens as Prophet for Fine-Grained Cache Eviction

**Type**: paper  
**Canonical source**: 01-Inbox/MaskKV.pdf  
**Date**: 2025 (arXiv)  
**Relevance**: Proposes cache eviction specifically designed for dLLM's mask token patterns. Achieves extreme compression (5% of tokens) with 94% performance retention. Critical insight: mask tokens provide signal for eviction decisions.  
**Status**: processed  
**Last updated**: 2026-04-16

## Summary

MaskKV is a training-free cache eviction framework that exploits the unique role of mask tokens in dLLMs. The key insight is that mask tokens' attention patterns reveal which prompt tokens are important vs dispensable for each denoising step.

Two innovations:
1. **Mask-query guided scoring**: Uses attention weights between mask tokens and prompt tokens to score prompt importance.
2. **Adaptive cache budgeting**: Allocates cache resources based on layer-wise importance (intermediate layers get less).

Compresses KV cache to 5% of full size while retaining 94% of performance on LongBench.

## Key claims

- Mask tokens are unique to dLLMs and carry information about which tokens matter.
- Attention from mask queries to prompt tokens indicates prompt importance for the current denoising step.
- Prompt tokens with low attention from masks can be evicted from cache.
- Adaptive budgeting: intermediate layers are less sensitive to cache compression.
- 31× acceleration at 32k prompt length with 256 KV pairs (<5% of tokens).
- Works on LongBench (long-context benchmarks) where KV cache is most constrained.

## Mechanisms

### Mask-Query Guided Scoring

At each denoising step:
1. For each prompt token, compute attention weight from all mask tokens.
2. Sum/average attention weights to get prompt token importance score.
3. Evict prompt tokens with lowest scores first.
4. Keep top-N tokens in cache based on budget.

### Adaptive Cache Budgeting by Layer

Observation: Different layers have different cache sensitivity:
- Early layers: Highly sensitive to cache compression.
- Intermediate layers: Can tolerate more compression.
- Later layers: Highly sensitive.

Budget allocation:
- Allocate more cache budget to early and late layers.
- Compress intermediate layers more aggressively.
- This is per-layer budgeting, similar to D2Cache.

### Cache Eviction Strategy

When cache is full:
1. Score all cached tokens.
2. Evict lowest-scoring tokens.
3. Recompute evicted token KV on next access (or don't - just drop them).

### Integration with dLLM Architecture

MaskKV works because:
- dLLMs maintain mask tokens until a position is denoised.
- Mask tokens' queries reveal what's important at current step.
- This is unique to masked diffusion models; AR models don't have this signal.

## Hardware implications

- Attention weight extraction required for scoring.
- This adds compute overhead but is amortized by reduced cache size.
- Memory reduction is dramatic: 20× compression.
- Can fit much longer contexts in same memory.

## Relevance to Apple Silicon

- Extreme cache compression is valuable for unified memory efficiency.
- Attention scoring could be fused into attention kernel.
- Works well for long contexts (32k) - important for LLADA2.1 applications.
- Could combine with other caching methods for further optimization.
- Metal implementation: Would need to efficiently extract attention weights for scoring.

## Extracted concepts

- [[mask-query-attention]]
- [[cache-eviction]]
- [[layer-wise-budgeting]]
- [[attention-scoring]]

## Open questions

- What is the scoring function precisely? How are attention weights aggregated?
- How does MaskKV compare to importance-based eviction strategies for AR models?
- Can the scoring be approximated for lower overhead?
- What is the interaction with different attention implementations (softmax vs linear)?
- How does adaptive budgeting affect quality? Is there a sweet spot?
- What is the Metal kernel design for mask-query scoring?

## Related pages

- [[dllm-cache]] (different caching approach)
- [[elastic-cache]] (layer-wise decisions)
- [[d2cache-dual-adaptive-cache]] (similar layer-wise budgeting)
- [[llada2-1-tech-report]] (base model)