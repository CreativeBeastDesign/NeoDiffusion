# Radix Caching

**Summary**: Caching mechanism that uses radix trees (or similar prefix-based structures) to store and reuse KV cache entries across different prompts or batch items.  
**Aliases**: KV cache radix tree, prefix caching  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Radix Caching is a KV cache optimization that stores key-value states in a radix tree (or prefix tree) structure. When processing new prompts or batch elements that share common prefixes, the KV cache for the shared prefix can be reused without recomputation. This is particularly valuable for workloads with repeated or similar prompts (e.g., server-side inference with many similar requests).

## Why it matters

SGLang's integration of radix caching for block diffusion LLMs reduces redundant computation, improving throughput in batched inference. For dLLMs with long contexts, the cost of computing KV states is significant; caching shared prefixes can yield substantial speedups, especially in scenarios like code completion where many requests share similar imports or boilerplate.

## Mechanism

- Build a radix tree where each node corresponds to a token position and stores its KV cache.
- When a new sequence arrives, traverse the tree to find the longest matching prefix.
- Reuse KV cache from the prefix node, only computing for new tokens.
- On modifications (e.g., edits during MBE), need to invalidate or update cached nodes downstream.

## Trade-offs

- **Memory overhead**: Radix tree structure adds metadata; cache may grow large.
- **Hit rate**: Depends on workload; high similarity → good hit rate.
- **Invalidation complexity**: When edits occur, cached descendants may become stale; need tracking of which sequences depend on which nodes.
- **Construction cost**: Building the tree adds some overhead but pays off over many requests.

## Apple Silicon implications

- Unified memory means cache memory competes with other uses; size must be bounded.
- Tree traversal can be done on GPU or CPU; likely CPU-side management to reduce GPU overhead.
- On-device scenarios may have less reuse (personalized prompts) → lower benefit.
- Could combine with block-wise attention: cache at block granularity to reduce tree depth.

## Related concepts
- [[block-wise-causal-attention]]
- [[multi-block-editing-mbe]] (cache invalidation challenges)
- [[sglang-rollout-engine]] (implementation context)

## Open questions
- What is the typical cache hit rate for code generation workloads on consumer hardware?
- How does radix caching interact with per-block FP8 quantization (are cached blocks stored in FP8)?