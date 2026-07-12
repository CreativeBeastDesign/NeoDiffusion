# Credit Decoding

**Summary**: A training-free acceleration algorithm for dLLMs that accumulates confidence "credits" for token predictions across diffusion steps, boosting tokens that consistently appear as top candidates to commit earlier, reducing redundant computation.  
**Aliases**: credit decoding, CreditDecoding, temporal confidence accumulation  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[dinfer-framework]]

---

## Definition

Credit Decoding maintains a per-position, per-token credit score that tracks how consistently a token has been favored by the model across denoising steps. At each step, the credit is combined with the current logits as a log-domain prior, enhancing the probability of tokens with stable predictions. This allows correct tokens that are underconfident early to stabilize and be committed earlier, avoiding repeated re-masking and re-evaluation, thereby accelerating convergence.

## Why it matters

In standard dLLM inference, each step only looks at current confidence; tokens that are ultimately correct but fluctuate below the threshold get re-masked repeatedly, wasting compute. Credit Decoding:
- Reduces redundant computation by promoting consistently predicted tokens
- Stabilizes decoding, especially in long-sequence and reasoning tasks
- Increases tokens per forward (TPF) by committing more tokens per iteration
- Works completely training-free and is compatible with other optimizations (threshold, KV-cache, compilation)

## Mechanism

For each position `i` and token `v ∈ V`, maintain credit `C_{i,v}^t`.

At step `t`, given input `x_t` and model distribution `p_θ(v|x_t)`, let `v* = argmax_v p_θ(v|x_t)` be the top candidate.

Credit update:
```
C_{i,v}^t = {
    β·C_{i,v}^{t-1} + (p_θ(v*|x_t))^γ,   if v = v*
    β·C_{i,v}^{t-1},                    otherwise
}
```
Parameters: `β ∈ (0,1)` (discount factor), `γ ∈ (0,1)` (concave transform giving larger boosts to moderate confidence).

Enhanced logits:
```
f̃_θ(x_t)_i^v = f_θ(x_t)_i^v + α·log(1 + C_{i,v}^t)
```
with `α > 0`. The enhanced distribution `p̃_θ(v|x_t) = softmax(f̃_θ(x_t)_i^v)` is then used for thresholding or sampling.

Crucially, credit is typically maintained **only within the current decoding block** to limit influence of uncertain future context and reduce memory overhead.

## Trade-offs

- **Memory**: Per-token per-vocab credits are O(V) per position; for long sequences, this can be substantial. However, restricting to current block and using approximate representations (e.g., top-k credits) can mitigate.
- **Speed**: Reduces total diffusion iterations; overhead per step is small (credit update + log addition).
- **Quality**: Improves stability and final accuracy, especially on reasoning tasks.
- **Hyperparameters**: β, γ, α require tuning; default values not explicitly given in paper (need to extract from experiments).
- **Compatibility**: Works with any decoding policy that consumes a probability distribution.

## Apple Silicon implications

- Credit storage could be a memory concern on Apple Silicon's unified memory; limiting to current block is essential.
- The log-domain combination is cheap; could be fused into the decoding kernel on Metal.
- Would integrate nicely with a Metal-based dLLM runtime; credits could be stored in threadgroup memory for the block.
- The reduction in iterations directly benefits battery life and responsiveness for on-device inference.

## Open questions

- What are the default hyperparameter values (β, γ, α) used in dInfer? Could they be scheduled adaptively?
- How does credit memory scale with vocabulary size? Can we compress credits (e.g., quantize to 8-bit, or track only top-k)?
- Does credit decoding combine well with iteration smoothing? Both use additional information; is there interference?
- Credit decoding accumulates across steps; does this introduce bias toward early predictions that might be wrong? How does β=1 vs β<1 behave?
- Could credit be shared across positions? Or is position-specific essential?
- How sensitive is credit decoding to the block size (since credits are per-block)?

## Related concepts
- [[iteration-smoothing]] (both reuse otherwise discarded information; credit tracks consistency, IterSmooth enriches embedding; could combine)
- [[soft-parallel-decoding]] (DMax's expected embedding similar to credit's log-prior boost; both training-free per-position refinement)
- [[vectorized-likelihood-estimation]] (credit uses per-token scoring)
- [[configurable-threshold-decoding]] (credit enhances distribution before thresholding; could inform adaptive τ_mask/τ_edit)
- [[per-token-early-stopping]] (both track token stability; credit accumulates over time, Jot uses single-step confidence)
- [[entropy-sum-decoding]] (both accumulate per-token signals; credit tracks consistency, entropy tracks uncertainty; could combine)
- [[in-place-chain-of-thought]] (ICE's confidence monitoring similar to credit accumulation; both could inform early exit decisions)
- [[dinfer-framework]] (credit decoding is a component)
- [[multi-turn-forward-mtf]] (MTF may also benefit from accumulated credits across turns)
- [[temporal-self-consistency-voting]] (TSE's related but distinct step-aggregation idea: credit decoding accumulates signal to decide *when a token commits during generation*, while temporal voting aggregates intermediate *answers* post-hoc to pick a final output — different point in the pipeline; whether they compose is an open question raised on both pages)

## Related pages

- [[dinfer-framework]] (source note)
