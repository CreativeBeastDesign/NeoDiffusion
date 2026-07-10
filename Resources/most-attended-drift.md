# Most-Attended-Token Drift

**Summary**: Elastic-Cache v2's empirical observation that the token receiving the most cumulative attention exhibits the smallest KV drift across steps, making it a conservative lower-bound proxy for whether any token in a layer has meaningfully changed.
**Aliases**: most-attended-token stability, conservative drift lower bound
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[elastic-cache-v2]]
- [[elastic-cache]]

---

## Definition

Most-attended-token drift is the specific empirical claim (Elastic-Cache v2's "Observation 3") that the token receiving the highest cumulative attention at a given layer/step is also the token whose KV changes *least* between steps — and that this token's drift therefore provides a conservative lower bound on the KV drift of the layer as a whole (**sourced**, [[elastic-cache-v2]]). This is the specific justification for why [[attention-aware-drift-test]] can safely test only one token instead of all of them. The [[elastic-cache]] concept page documents the same underlying mechanic under "most-attended-token" in its mechanism and theoretical-foundations sections (Theorem A.9: "attention concentration and drift"), without naming it as a standalone concept — this page exists to give it its own atomic treatment as named explicitly in the v2 source note's extracted-concepts list.

## Why it matters

This observation is the load-bearing assumption behind the entire attention-aware drift test: if the most-attended token were *not* representative of overall layer drift (or worse, if it were the token with the *largest* drift), using it as a single-token proxy would be unsound rather than conservative. The wiki's existing [[elastic-cache]] page already cites a formal justification for this — Theorem A.9, bounding the most-attended token's drift by the average drift plus an O(√(dk R_ℓ / N)) term — which validates using it as a conservative staleness indicator (**sourced**, [[elastic-cache]]).

## Mechanism

1. At each layer and step, identify the token with the highest cumulative attention weight from the active decoding window.
2. Track this token's KV drift (change) between consecutive steps.
3. Treat this drift as a conservative estimate: if this token — theoretically the most stable — has drifted past threshold, assume the rest of the layer likely has too (**sourced**, [[elastic-cache-v2]]).

This directly underlies the single-token check in [[attention-aware-drift-test]], and the two should be read together: this page states *why* the shortcut is sound; that page states *how* it's used operationally.

## Trade-offs

- **Depends on attention concentration holding**: the "smallest drift" claim is empirical/theoretical (bounded, not guaranteed exactly zero) — Theorem A.9's bound includes a nonzero error term, meaning the proxy is conservative but not perfectly precise.
- **Single point of failure**: because the whole layer's staleness decision hinges on one token, any failure of the most-attended-token identification itself (e.g. attention ties, rapidly shifting attention focus) would propagate directly into a wrong refresh decision. Not directly addressed as a risk in either source note — flagged here as a reasonable inference from the mechanism's structure. **Inferred**.

## Apple Silicon implications

- Identifying the most-attended token requires an argmax reduction over attention weights, which is a cheap, well-supported parallel-reduction primitive on GPU architectures generally, including Metal. **Inferred**, not detailed in either source note.
- No additional hardware implications beyond what's already covered in [[elastic-cache]] and [[attention-aware-drift-test]].

## Related concepts
- [[attention-aware-drift-test]] (the operational procedure this observation justifies)
- [[elastic-cache]] (full method, including the formal Theorem A.9 bound)
- [[layer-wise-kv-dynamics]] (the companion depth-based observation from the same source)

## Open questions
- Can the most-attended-token heuristic be replaced with a learned importance/stability signal, as the v2 source note itself asks?
- What happens when attention is highly diffuse (no single clearly most-attended token) — does the conservative-bound property still hold?
- Has Theorem A.9's bound been validated empirically outside the original paper's benchmark set?
