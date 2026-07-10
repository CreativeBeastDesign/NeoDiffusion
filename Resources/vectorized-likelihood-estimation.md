# Vectorized Likelihood Estimation

**Summary**: Technique to compute block-conditional probabilities for multiple blocks and timesteps in parallel within a single forward pass, crucial for scaling EBPO.  
**Aliases**: batched likelihood computation, parallel ELBO evaluation  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Vectorized Likelihood Estimation is a method to compute the probability of each token in a block conditioned on composite inputs `z_n = y_{t_n} ⊕ y_0` for multiple timesteps `t_n` simultaneously. Instead of performing separate forward passes for each timestep, the algorithm packs these computations into a single batched forward pass, significantly reducing overhead.

## Why it matters

EBPO requires evaluating log-probabilities for many (block, timestep) pairs. Naive approach would be O(#blocks × #timesteps) forward passes, which is infeasible for long contexts. Vectorization reduces this to O(#timesteps) forwards, each producing all block probabilities, enabling RL scaling to trillion-token contexts.

## Mechanism

- For each timestep `n`, construct composite input `z_n` by concatenating noised sequence `y_{t_n}` and clean target `y_0` with appropriate block-causal masking.
- Run a single forward pass that computes probabilities for all blocks in parallel (e.g., by using a batch dimension or by structuring attention to handle multiple block positions at once).
- Extract block-level log-probabilities from the output.
- Sum across blocks and timesteps with weights `w_n` to approximate `log ρ(y|x)`.

## Trade-offs

- **Memory**: Composite inputs increase activation memory; need to balance batch size vs GPU memory.
- **Implementation complexity**: Requires careful batching and masking to avoid interference between different `(n, b)` pairs.
- **Numerical precision**: Summing many log-probabilities may underflow/overflow; use log-sum-exp tricks.
- **Kernel fusion**: Opportunity to fuse likelihood extraction with softmax to reduce memory traffic.

## Apple Silicon implications

- Apple Silicon GPUs have good compute-to-memory ratios; vectorized operations should exploit this.
- The composite input size is larger than original; memory bandwidth may become bottleneck.
- Can be implemented in Metal using compute shaders with threadgroups processing multiple blocks concurrently.
- Need to profile whether packing multiple timesteps into a single shader invocation improves occupancy.

## Related concepts
- [[elbo-based-block-level-policy-optimization-ebpo]]
- [[block-wise-causal-attention]]
- [[mega-kernel-v1-fused-remask-sample]] (fusion opportunity)

## Open questions
- What is the optimal packing strategy for Metal threadgroups (e.g., pack blocks vs pack timesteps)?
- Does vectorized likelihood estimation benefit from Apple's tensor cores or is it memory-bound?