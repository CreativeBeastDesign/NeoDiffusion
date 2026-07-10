# Editable State Evolution

**Summary**: Diffusion state transition that allows both unmasking and token editing, breaking the absorbing-state constraint.  
**Aliases**: draft-and-edit paradigm, editable decoding  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Editable State Evolution extends discrete diffusion beyond the traditional absorbing-state framework. Instead of only allowing `[MASK] → token` transitions, it permits `token → token` editing operations, enabling the model to revise already-decoded tokens during the generation process.

## Why it matters

Traditional dLLMs suffer from exposure bias: once a token is sampled, it cannot be changed, so errors compound. Editable evolution allows retroactive correction, which decouples initial drafting speed from final quality. This is the key innovation enabling LLaDA2.1's S Mode.

## Mechanism

At each timestep t, two sets are defined:
- Unmasking set Γ_t: masked positions where `pθ(vi|xt) > τ_mask`
- Editing set Δ_t: decoded positions where `pθ(vi|xt) > τ_edit` AND `vi != xi`

State update: `xi_{t-1} = vi` if `i ∈ Γ_t ∪ Δ_t`, else `xi_t` unchanged.

## Trade-offs

- **Flexibility**: Enables correction without restarting generation.
- **Compute overhead**: Must evaluate all positions, not just masked.
- **Stability risk**: Excessive editing can cause oscillations or degradation.
- **Threshold coupling**: τ_mask and τ_edit must be jointly tuned.

## Apple Silicon implications

- Requires attention recomputation when editing tokens that affect future positions (if using causal attention).
- Could exploit Metal's compute shader flexibility to dynamically change which positions are updated.
- Memory traffic increases due to reading all token embeddings each step.
- Opportunity: batch editing operations into a single kernel launch with divergent execution.

## Related concepts
- [[mask-to-token-m2t]]
- [[token-to-token-t2t]]
- [[configurable-threshold-decoding]]
- [[multi-block-editing-mbe]]

## Open questions
- How many edit passes are typically needed for convergence on Apple Silicon hardware?
- Can editing be amortized across multiple diffusion steps to reduce kernel launch overhead?