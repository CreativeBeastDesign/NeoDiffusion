# Hierarchical Decoding

**Summary**: A training-free parallel decoding strategy that recursively partitions masked spans into smaller sub-regions, ensuring at least one token is decoded per region per forward pass, reducing local dependencies and improving semantic consistency.  
**Aliases**: hierarchical decoding, divide-and-conquer decoding  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[dinfer-framework]]

---

## Definition

Hierarchical Decoding is a parallel decoding algorithm for diffusion language models that addresses the quality degradation in naive parallel token prediction by recursively subdividing masked token spans. Instead of predicting all masked tokens simultaneously, it partitions the span into smaller regions, decodes at least one token per region in each iteration, and recurses until all tokens are revealed. This non-contiguous decoding pattern increases spacing between masked tokens, reducing conditional independence violations and improving output quality while maintaining efficiency.

## Why it matters

Naive parallel decoding in dLLMs often suffers because simultaneously predicted tokens violate conditional independence assumptions, leading to semantic inconsistencies (e.g., contradictory statements, broken code syntax). Hierarchical decoding mitigates this by:
- Breaking large masked spans into manageable sub-regions
- Ensuring each region makes progress independently
- Increasing distance between still-masked tokens, reducing local interference
- Achieving near O(log n) complexity in ideal case where each decoding splits spans effectively

This enables higher tokens-per-forward (TPF) without fine-tuning the model.

## Mechanism

Given a sequence with masked spans, for each masked region:
1. If the region is small enough (e.g., size ≤ 1), decode directly using standard threshold.
2. Otherwise, partition the region into sub-regions (e.g., by splitting at the median or at positions of highest confidence).
3. For each sub-region, attempt to decode at least one token (typically the most confident position).
4. Recursively apply to remaining masked sub-regions in subsequent iterations or within same forward pass if using hierarchical tree.

The key insight: the spatial distribution of masked tokens impacts prediction stability. By forcing non-contiguous decoding, we reduce the density of masked positions, making each prediction more reliable.

The algorithm is training-free and can be combined with other decoding strategies like threshold or credit decoding.

### Pseudocode outline

```
function hierarchical_decode(masked_span, logits):
    if size(masked_span) <= 1 or max_confidence(logits) > threshold:
        return decode_token(masked_span, logits)
    else:
        sub_spans = partition(masked_span)  # e.g., split into 2 or more
        results = []
        for sub in sub_spans:
            # decode one token in this sub-span
            tok, confidence = decode_one_token(sub, logits)
            results.append(tok)
        return results  # remaining masked positions recurse
```

In dInfer, hierarchical decoding is implemented as one of the decoding strategies within the modular framework.

## Trade-offs

- **Quality**: Reduces semantic inconsistencies compared to naive parallel decoding; empirically yields higher performance on benchmarks.
- **Efficiency**: Increases TPF by enabling more parallel predictions per forward pass without sacrificing accuracy.
- **Complexity**: Slightly more complex control flow than flat threshold decoding; but still training-free.
- **Hyperparameters**: Requires partitioning strategy (split count, position selection) and confidence thresholds.

## Apple Silicon implications

- Hierarchical decoding could be implemented efficiently in Metal as part of the decoding pass; the recursion can be flattened into a fixed-depth tree for GPU threadgroups.
- The reduced local dependencies may be particularly beneficial for Apple Silicon's memory hierarchy, as distant masked tokens do not interfere through attention.
- Could be combined with credit decoding to further boost stability: credits could be tracked per sub-region.
- On resource-constrained devices, fewer diffusion iterations (higher TPF) directly translates to lower latency and energy use.

## Open questions

- What is the optimal partitioning strategy? Barycentric (center-favoring) vs confidence-based splitting?
- How does hierarchical decoding scale with very long contexts (e.g., 8k+ tokens)? Does the recursion overhead become significant?
- Can hierarchical decoding be combined with block-wise editing (MBE) to edit multiple blocks hierarchically?
- Does the benefit of hierarchical decoding depend on model architecture or training data? (Tested on LLaDA-MoE; may vary.)
- Could hierarchical decoding be learned via training (e.g., by imitating good trajectories) rather than being heuristic?

## Related concepts
- [[credit-decoding]] (can be used within hierarchical regions)
- [[block-wise-causal-attention]] (hierarchical reduces need for full attention within masked spans)
- [[multi-block-editing-mbe]] (MBE edits blocks; hierarchical could decide edit order)
- [[dinfer-framework]] (the framework that introduced this)
- [[configurable-threshold-decoding]] (hierarchical builds upon thresholding)

## Related pages

- [[dinfer-framework]] (source note)
