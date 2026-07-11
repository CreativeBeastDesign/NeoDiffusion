# Directed Draft Graph (Spiffy)

**Summary**: A generalization of AR speculative decoding's draft trees to bidirectional dLLMs, where a draft block node can have multiple parent blocks — giving multiple pathways to acceptance, a structural possibility that doesn't exist in strictly sequential AR draft trees.
**Aliases**: draft graph, multi-parent draft structure
**Status**: draft
**Last updated**: 2026-07-03
**Sources**:
- [[spiffy]]

---

## Definition

A directed draft graph is the data structure Spiffy uses to organize which draft candidates to try during [[auto-speculative-decoding]]. Unlike AR speculative decoding's draft *trees* (where each node has exactly one parent, reflecting AR's strictly sequential token ordering), a dLLM draft graph node can have *multiple* parent blocks: a draft block B is a child of block A if t_b = t_a − 1, |unmasked(A)| + S_tb = |unmasked(B)|, and unmasked(A) ⊂ unmasked(B) — meaning B extends A by unmasking additional positions consistent with A's own unmasking (**sourced**, [[spiffy]]). Multiple such parent relationships can hold simultaneously for one node, since dLLM's bidirectional, order-flexible unmasking allows several different partial-unmasking states to all validly precede the same later state.

## Why it matters

AR speculative decoding's draft trees are constrained to be trees precisely because AR generation has one fixed, sequential token order — there's only ever one way to have arrived at a given partial sequence. dLLMs unmask tokens in a flexible, non-fixed order, so the same partially-unmasked state can legitimately be reached via multiple different unmasking paths. A draft *graph* (not tree) is the structurally correct representation of this — using a tree would either lose valid draft-acceptance pathways or force an arbitrary, non-canonical choice among them. This is presented as a structural insight specific to bidirectional dLLM decoding, not merely an implementation detail borrowed from AR speculative decoding.

## Mechanism

Nodes represent candidate unmasking states (specific sets of unmasked positions with specific token values, per the c_ij candidates described in [[auto-speculative-decoding]]); edges represent valid single-block extensions per the parent-child condition above. Multiple parent edges into one node represent the multiple distinct unmasking histories that could validly precede that state. The specific graph structure actually used at inference is not constructed dynamically but fixed in advance via [[offline-draft-calibration]] (**sourced**, [[spiffy]]).

## Trade-offs

- **Structural correctness vs. AR-tree familiarity**: the multi-parent property is what makes the graph correct for dLLMs, but it also means AR speculative-decoding intuitions and implementations (built around trees) don't transfer directly — an implementation adapting AR speculative-decoding code would need explicit changes to support multiple parents per node, not just a parameter change.
- **Fixed, not dynamic**: the graph structure is calibrated offline and then fixed for inference (see [[offline-draft-calibration]]) — this trades adaptivity (a dynamically-constructed graph could in principle react to per-input characteristics) for predictable, low-overhead runtime cost.

## Apple Silicon implications

- **Inferred**: because the graph structure is fixed at calibration time (not constructed per-inference), the runtime shape of draft-candidate batching is static and known in advance — favorable for kernel specialization, since a Metal kernel can be shaped around a known, fixed draft-graph topology rather than needing to handle a dynamically varying one.
- No additional hardware-specific claims are made in the source beyond the general auto-speculation memory-footprint advantage covered under [[auto-speculative-decoding]].

## Related concepts
- [[auto-speculative-decoding]] (the drafting approach this graph structures)
- [[offline-draft-calibration]] (how the specific graph used at inference is determined)
- [[speculative-decoding-dllm]] (AR-style draft-tree speculative decoding, for structural contrast)

## Open questions
- How large does a typical calibrated draft graph get (number of nodes/edges) relative to an equivalent AR draft tree at the same draft budget D, and does graph complexity itself become a runtime cost as D grows?
- Does the multi-parent structure provide a measurable acceptance-rate advantage over a tree-constrained approximation of the same draft budget, or is the benefit mostly theoretical/structural correctness rather than a large empirical gain?
