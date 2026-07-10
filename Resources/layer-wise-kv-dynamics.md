# Layer-Wise KV Dynamics

**Summary**: The empirical observation that KV drift across denoising steps increases with transformer layer depth — shallow layers encode stable local/lexical structure, while deep layers keep adjusting to encode evolving global/semantic dependencies. The theoretical basis for Elastic-Cache's depth-aware refresh.
**Aliases**: KV drift monotonicity, depth-dependent KV volatility
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[elastic-cache-v2]]
- [[elastic-cache]]

---

## Definition

Layer-wise KV dynamics is Elastic-Cache v2's "Observation 2": KV states in shallow transformer layers stabilize quickly (encoding local lexical structure), while deep layers continue adjusting across denoising steps (encoding global semantic dependencies that keep evolving as more of the sequence resolves) (**sourced**, [[elastic-cache-v2]]). This is the empirical/theoretical basis for [[depth-aware-refresh]]'s policy of refreshing from a boundary layer ℓ★ onward while reusing shallow-layer caches. The existing [[elastic-cache]] concept page formalizes this as "Layer-wise KV drift monotonicity" (Theorem A.8: under bounded-representation and progressive-unmasking assumptions, expected KV drift increases with layer depth) — this page is a focused breakout of that same finding as separately named in the v2 source note.

## Why it matters

This is a *structural* claim about how transformer depth relates to representational stability under iterative denoising — distinct from, though related to, [[token-stability]] (which is about how a *token's* KV stabilizes over steps, not how a *layer's* KV stabilizes across depth). Layer-wise KV dynamics gives depth-aware refresh its theoretical footing: without a reason to believe shallow layers are systematically more stable than deep ones, choosing a fixed refresh boundary ℓ★ would be an arbitrary heuristic rather than a principled exploitation of model structure.

## Mechanism

The claim, as stated across both source notes:
- Shallow layers: stabilize quickly, encode local lexical structure that doesn't depend much on distant/evolving context.
- Deep layers: continue adjusting, encode global semantic dependencies that shift as more of the sequence resolves during denoising.
- Consequence: selective refresh starting from deeper layers is sufficient — shallow-layer caches can be reused across more steps than deep-layer caches (**sourced**, [[elastic-cache-v2]]).

The existing [[elastic-cache]] page provides the formal version: Theorem A.8 proves expected KV drift increases with layer depth under stated assumptions (bounded representations, progressive unmasking) — moving this from a purely empirical pattern to one with a stated theoretical justification (**sourced**, [[elastic-cache]]).

## Trade-offs

- **Assumption-dependent**: Theorem A.8's guarantee holds "under assumptions of bounded representations and progressive unmasking" — it is not an unconditional guarantee, and the source material doesn't detail how sensitive the result is to violations of those assumptions.
- **Static vs. per-prompt boundary**: because this is a monotonicity claim rather than a claim about the *exact* location of the transition, the actual boundary ℓ★ still requires empirical tuning per model (and possibly per prompt) — the monotonicity result justifies the *existence* of a good boundary, not its precise value.

## Apple Silicon implications

- Shares the implications already discussed in [[elastic-cache]]: recomputing only deeper layers while reusing shallow caches should reduce memory traffic on Apple Silicon's unified memory architecture, and the theoretical guarantees (Theorem A.8/A.9) provide a principled starting point for choosing an initial boundary before empirical re-tuning for Apple Silicon's specific hardware characteristics.
- **Inferred**: because this is a depth-structural claim rather than a content-dependent one, it plausibly transfers across hardware more reliably than content-dependent thresholds (like γ in the drift test) — untested, but a reasonable prioritization signal for what to validate first when porting to Metal.

## Related concepts
- [[depth-aware-refresh]] (the refresh policy this observation justifies)
- [[elastic-cache]] (full method, including the formal Theorem A.8 statement)
- [[most-attended-drift]] (the companion token-level, rather than layer-level, drift observation)
- [[token-stability]] (a related but distinct concept — token-level stability over steps, not layer-level stability over depth; worth not conflating)

## Open questions
- How sensitive is the monotonicity result to violations of the stated assumptions (bounded representations, progressive unmasking)?
- Does the shallow-stable/deep-volatile pattern hold uniformly across model architectures, or is it specific to LLaDA-family models evaluated in the source material?
- Can ℓ★ be predicted statically from architecture (layer count, hidden dimension) rather than tuned empirically per model?
