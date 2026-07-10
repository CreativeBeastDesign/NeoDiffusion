# Token-to-Token (T2T)

**Summary**: Editing operation that replaces a currently decoded token with a different token based on model confidence, enabling retroactive correction.  
**Aliases**: token editing, correction operation, remasking  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Token-to-Token (T2T) is an editing operation where an already-decoded token `xi` (not [MASK]) is replaced with a new token `vi` if the model's confidence in the current token is low AND confidence in an alternative token is high. This enables the model to correct its own errors during generation.

## Why it matters

T2T breaks the rigid monotonicity of traditional absorbing-state diffusion. It addresses exposure bias by allowing the model to revisit and revise already-generated tokens, effectively implementing a "draft-and-edit" paradigm. This is key to LLaDA2.1's speed-accuracy tradeoff management.

## Mechanism

At timestep t, for positions where `xi != [MASK]`, compute `vi = argmax v pθ(v|xt)`. If `pθ(vi|xt) > τ_edit` AND `vi != xi`, the token is replaced. The editing set `Δ_t` captures these positions. Combined with unmasking set `Γ_t`, the state transition applies updates on `Γ_t ∪ Δ_t`.

## Trade-offs

- **Improves fidelity**: Corrects errors introduced during aggressive M2T drafting.
- **Adds compute**: Requires evaluating all tokens (not just masked) at each step.
- **φ_edit tuning**: Too low → excessive editing, instability; too high → missed corrections.
- **Convergence**: Multiple edit passes may be needed for coherence, especially in S Mode.

## Apple Silicon implications

- T2T doubles the per-position computation compared to M2T alone (must evaluate all positions).
- Can be fused with M2T in a single kernel to reduce launch overhead.
- Memory access pattern: requires reading current token embeddings for all positions, not just masked.
- Potential for vectorized execution: same confidence threshold logic applies uniformly.

## Related concepts
- [[mask-to-token-m2t]]
- [[editable-state-evolution]]
- [[configurable-threshold-decoding]]
- [[multi-block-editing-mbe]]

## Open questions
- Does T2T benefit from Apple Silicon's unified memory or does it increase traffic?
- Can editing be scheduled to minimize recomputation of attention states?
- How does T2T interact with KV cache reuse across steps?