# Selective Layer Refresh

**Summary**: The combined policy name Elastic-Cache v2 uses for jointly applying the attention-aware drift test (when to refresh) and depth-aware refresh (where to refresh) — the integration point of the method's two main decisions.
**Aliases**: joint when/where cache refresh policy
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[elastic-cache-v2]]
- [[elastic-cache]]

---

## Definition

Selective layer refresh is the umbrella term, as used in Elastic-Cache v2's extracted-concepts list, for the combination of [[attention-aware-drift-test]] (deciding *when* a layer's cache has gone stale) and [[depth-aware-refresh]] (deciding *where*, i.e. from which layer boundary ℓ★, to actually recompute) into a single joint policy (**sourced**, [[elastic-cache-v2]]). It is not a separate mechanism beyond the composition of those two — the existing [[elastic-cache]] concept page's "Full Algorithm (Algorithm 1)" pseudocode is the concrete realization of exactly this joint policy, iterating over layers and, for each, applying the drift test to decide whether to advance the boundary ℓ★ at which recomputation begins.

## Why it matters

Naming the combination matters because the two decisions are not independent in the algorithm: the drift test's outcome at a given layer directly determines the boundary ℓ★ used for depth-aware refresh at that step (per [[elastic-cache]]'s Algorithm 1: `if σ_t^ℓ < γ: ℓ★ = ℓ`). Treating "selective layer refresh" as its own concept makes explicit that Elastic-Cache's core contribution is this specific *coupling* — using an attention-derived signal to set a depth-based boundary — rather than either half in isolation.

## Mechanism

Per [[elastic-cache]]'s Algorithm 1, for each layer ℓ from 1 to L, at each step t:
1. If ℓ > current ℓ★ (already past the stale boundary from an earlier layer this step): recompute KV for all tokens at this layer (cache-update branch).
2. Otherwise: reuse cached KV for the active query set, run [[attention-aware-drift-test]] on the most-attended token, and if drift exceeds threshold γ, set ℓ★ = ℓ (advancing the boundary, triggering full recomputation for this and all deeper layers this step).

This is the precise mechanism by which "when" (drift test) and "where" (boundary ℓ★) are fused into one pass over the layer stack, rather than being computed as two independent, separately-applied policies.

## Trade-offs

Inherits the trade-offs of both component mechanisms (see [[attention-aware-drift-test]] and [[depth-aware-refresh]]): threshold γ and boundary determination together control the same speed/accuracy dial, and because they're coupled within a single per-step, per-layer loop, tuning one in isolation from the other is not meaningful — they must be considered jointly, which the existing [[elastic-cache]] page's ablation table (window size, threshold γ, top-k tracking) already reflects.

## Apple Silicon implications

The joint per-layer loop structure (Algorithm 1) suggests the natural Metal implementation unit is a per-step, per-layer kernel dispatch that carries the running boundary state ℓ★ forward — rather than treating drift-testing and refresh-execution as separable, independently-schedulable kernels. This coupling is worth flagging explicitly when designing a Metal port, since it constrains how much the two decisions can be pipelined or parallelized independently. **Inferred** from the algorithm structure in [[elastic-cache]]; not stated directly in either source note.

## Related concepts
- [[elastic-cache]] (the full method and Algorithm 1 pseudocode this concept names)
- [[attention-aware-drift-test]] (the "when" component)
- [[depth-aware-refresh]] (the "where" component)
- [[layer-wise-kv-dynamics]] (the structural observation justifying that a depth-based boundary is meaningful at all)

## Open questions
- Can the "when" and "where" decisions be decoupled for better parallelism on Apple Silicon without losing the joint policy's effectiveness?
- What is the Metal kernel design for maintaining the running boundary state ℓ★ across a per-layer loop within a single denoising step?
