# Iteration Smoothing (IterSmooth)

**Summary**: A technique that reuses discarded logit information from masked positions to enrich their embeddings across diffusion iterations, improving token confidence and decoding stability.  
**Aliases**: IterSmooth, iteration smoothing, logit-based embedding enrichment  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[dinfer-framework]]

---

## Definition

Iteration Smoothing is a training-free algorithm for diffusion language model inference that, at each denoising step, converts the logits distribution for *masked* positions into an expected embedding and injects it into the mask token's embedding. This allows uncertain positions to be enriched with distribution-level signals rather than being ignored, effectively increasing the information content per iteration.

## Why it matters

In standard dLLM decoding, only the argmax tokens are committed; the logits for other masked positions are discarded. IterSmooth reuses this otherwise wasted information by converting the full probability distribution into an expected embedding via the input embedding matrix. This enriches the representation of masked tokens, leading to:
- Higher token confidence in subsequent iterations
- Increased number of tokens decoded per forward pass (TPF improvement of 30–40%)
- Better final generation quality
- More efficient use of compute (more tokens decoded per iteration)

## Mechanism

Given at step `t`:
- Logits `z_t[i]` at position `i` (for masked positions)
- Input embedding matrix `W_emb`
- Standard mask embedding `e_mask`

Compute:
```
p_t[i] = softmax(z_t[i])                # probability distribution
Δe_t[i] = p_t[i] · W_emb                # expected embedding
e_{t+1}[i] = e_mask + α_t · Δe_t[i]     # updated masked token embedding
```

The mixing weight `α_t` increases over steps:
```
α_t = min(α_init + α_growth·t, α_preset)
```
Typical values: `α_init = 0.1`, `α_preset ∈ [0.2, 0.4]`.

Optionally, a decode-threshold schedule decays from 1.0 toward a target, ensuring only high-confidence tokens are committed early; later steps relax the threshold and rely more on distribution-level guidance.

Crucially, operation is applied **only to masked positions** to avoid altering the training distribution of already-decoded tokens.

## Trade-offs

- **Speed**: Increases TPF by 30–40%, meaning fewer diffusion iterations needed.
- **Quality**: Improves final text quality (empirically observed).
- **Compute overhead**: Small additional cost: one softmax + embedding lookup per masked position; but this is dwarfed by the model forward cost.
- **Hyperparameters**: Requires tuning `α_init`, `α_preset`, `α_growth`, and threshold schedule.
- **No retraining**: Works completely training-free.

## Apple Silicon implications

- Iteration Smoothing could be implemented in Metal as part of the embedding layer or as a custom kernel fused with the model's first layer.
- The additional compute (softmax + embedding lookup) is memory-bound; on Apple Silicon, memory bandwidth may be the limiting factor. But the reduction in total iterations likely outweighs this cost.
- The concept of enriching masked tokens with distributional information could be combined with other dLLM optimizations like credit decoding or configurable threshold decoding.
- Could be beneficial for on-device generation where latency is critical: fewer iterations = lower energy consumption.

## Open questions

- What is the sensitivity of performance to the `α_t` schedule? Could it be adaptive based on token confidence variance?
- Does IterSmooth interact positively with KV-cache strategies like vicinity refresh? Does caching interfere with smoothing?
- How does IterSmooth compare to adding a learned "uncertainty" embedding during training? Is the distributional approach as effective as a trained solution?
- Can IterSmooth be extended to include top-k samples instead of full distribution to reduce memory traffic?
- Does IterSmooth work equally well for all model architectures (LLaDA, LLaDA-MoE, etc.) and tasks (code vs math vs instruction)?

## Related concepts
- [[credit-decoding]] (both accumulate information across steps; credit tracks consistency, IterSmooth enriches embedding)
- [[soft-parallel-decoding]] (DMax's expected embedding is similar to IterSmooth's Δe; both training-free embedding enrichment)
- [[vectorized-likelihood-estimation]] (IterSmooth uses embedding projection of probabilities)
- [[dinfer-framework]] (IterSmooth is one of its components)
- [[multi-block-editing-mbe]] (both involve multi-step refinement)
- [[configurable-threshold-decoding]] (both use threshold-based token commitment; IterSmooth's decode threshold interacts with τ_mask)
- [[entropy-sum-decoding]] (both provide theoretical grounding for adaptive batch size; IterSmooth's α_t schedule similar to entropy-sum's adaptive η)
- [[per-token-early-stopping]] (both reduce unnecessary computation; IterSmooth reduces per-step cost, Jot reduces step count)

## Related pages

- [[dinfer-framework]] (source note)
