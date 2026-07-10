# Depth-Aware Refresh

**Summary**: Elastic-Cache's policy for deciding *where* (which layers) to recompute once the attention-aware drift test signals staleness — refresh from a boundary layer ℓ★ onward, reusing cache for shallower layers.
**Aliases**: depth-aware refresh schedule
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[elastic-cache-v2]]
- [[elastic-cache]]

---

## Definition

Depth-aware refresh is the "where to refresh" half of Elastic-Cache's cache policy (paired with [[attention-aware-drift-test]]'s "when to refresh"): once staleness is detected, recomputation starts from a boundary layer ℓ★ and proceeds through deeper layers, while shallower layers continue reusing cached KV (**sourced**, [[elastic-cache-v2]]). This mirrors the "Layer-Aware Cache Update" mechanism already documented in the existing [[elastic-cache]] concept page — this page is a focused breakout of that same mechanism as named in the v2 source note, not an independent technique. See the note on this page's relationship to [[elastic-cache]] under [[attention-aware-drift-test]].

## Why it matters

This is the mechanism that exploits [[layer-wise-kv-dynamics]] (KV dynamics increasing with depth): if shallow layers stabilize quickly and deep layers keep evolving, refreshing *everything* whenever staleness is detected wastes compute on shallow layers that likely haven't actually changed. Depth-aware refresh instead concentrates recomputation where drift is most likely to have occurred, extracting more speedup per detected staleness event than a flat all-layers refresh would.

## Mechanism

1. Determine layer ℓ★ — the boundary above which layers get recomputed (typically deeper layers).
2. For layers < ℓ★: reuse cached KV, no recomputation.
3. For layers ≥ ℓ★: recompute KV fresh.
4. ℓ★ can be tuned per model or per step (**sourced**, [[elastic-cache-v2]]).

This matches, layer-index-direction aside, the "Layer-Aware Cache Update" logic in [[elastic-cache]]'s existing concept page, where layers ≤ ℓ★ reuse and layers > ℓ★ recompute — the two source notes use the boundary in the same directional sense (shallow=reuse, deep=recompute) even though the v1 and v2 prose descriptions are phrased with slightly different layer-index conventions. This is noted here rather than silently normalized, since the underlying source material wasn't cross-checked line-by-line against the original paper.

## Trade-offs

- **ℓ★ selection**: whether the boundary is fixed, learned, or determined per-step is explicitly left open in the v2 source note ("ℓ* can be tuned per model or per step" — not resolved to a specific policy).
- Same fundamental trade-off as documented in [[elastic-cache]]: aggressive (high) boundaries save more compute but risk missing real drift in nominally-shallow layers; conservative (low) boundaries are safer but capture less speedup.

## Apple Silicon implications

Shares the implications already noted in [[elastic-cache]]: layer-selective recomputation on unified memory could reduce memory traffic by skipping full recomputation passes for cached-shallow layers, and the boundary could be determined empirically (e.g. fixed at the last 1/3 of layers) rather than computed dynamically, trading adaptivity for lower runtime overhead — a plausible first-pass simplification for an initial Metal kernel. Not re-derived in full here; see [[elastic-cache]] directly.

## Related concepts
- [[elastic-cache]] (the full method this is one component of)
- [[layer-wise-kv-dynamics]] (the empirical depth-drift pattern this refresh policy exploits)
- [[attention-aware-drift-test]] (the paired "when to refresh" decision)
- [[selective-layer-refresh]] (the combined when+where policy name used in the v2 source note)

## Open questions
- Is ℓ★ better fixed per model or adapted per step/prompt? The source note leaves this open.
- How does depth-aware refresh interact with per-block processing (LLaDA-style block generation)?
- Can ℓ★ be predicted statically from architecture rather than tuned empirically?
