# ELBO-Based Block-Level Policy Optimization (EBPO)

**Summary**: Reinforcement learning algorithm for dLLMs that uses Evidence Lower Bound as a tractable likelihood proxy, enabling policy gradient updates at block granularity.  
**Aliases**: EBPO, block-level RL for diffusion  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

EBPO is a reinforcement learning framework designed to optimize discrete diffusion language models via policy gradients. It circumvents the intractability of sequence-level log-likelihood in diffusion models by using the Evidence Lower Bound (ELBO) as a surrogate objective. The algorithm operates at the block level, aggregating contributions from multiple diffusion timesteps in a single forward pass via vectorized likelihood estimation.

## Why it matters

Applying RL to dLLMs has been limited by computational cost and variance. EBPO makes large-scale RL feasible by vectorizing bound computation across blocks and timesteps. This enables LLaDA2.1 to integrate RL alignment (CPT+SFT+RL) effectively, improving reasoning and instruction following without prohibitive compute.

## Mechanism

- For a set of discretized timesteps `{t_n}` and weights `{w_n}`, construct composite input `z_n = y_{t_n} ⊕ y_0`.
- Use Block-Causal Mask `M` to ensure each block attends only to valid history.
- In a single forward per timestep, compute all block-conditional probabilities:
  `log ρ(y|x) ≈ Σ_n Σ_b [log pθ(y_b|z_n, x; M) - log pθ_old(y_b|z_n, x; M)]`
- Plug into clipped PPO-style surrogate objective `J_EBPO(θ)`.

## Trade-offs

- **Scalability**: Vectorization enables long-context RL; prior works limited to small contexts.
- **Complexity**: Requires careful implementation of block-causal masking and composite inputs.
- **Variance**: Still depends on quality of advantage estimator ˆA.
- **Overhead**: RL stage adds significant compute; used for alignment not base training.

## Apple Silicon implications

- Block-level operations map to Metal threadgroups; block size choices affect occupancy.
- Composite input construction (`y_t ⊕ y_0`) requires data layout that supports efficient concatenation.
- Vectorized likelihood estimation can be parallelized across threads and blocks.
- Need to evaluate whether ELBO computation can be fused with attention kernels to reduce memory traffic.

## Related concepts
- [[vectorized-likelihood-estimation]]
- [[multi-turn-forward-mtf]]
- [[mega-kernel-v1-fused-remask-sample]] (potential fusion)

## Open questions
- How does EBPO's memory footprint scale with block count on Apple Silicon?
- Can advantage estimation be computed on-GPU to reduce CPU-GPU synchronization?