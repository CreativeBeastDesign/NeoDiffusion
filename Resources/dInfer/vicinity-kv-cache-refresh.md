# Vicinity KV-Cache Refresh

**Summary**: A KV-cache management strategy for dLLMs that selectively recomputes key/value states for masked tokens and their immediate neighbors within a small window, balancing cache reuse with accuracy by exploiting semantic locality.  
**Aliases**: vicinity refresh, KV-cache vicinity, selective KV update  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[dinfer-framework]]

---

## Definition

Vicinity KV-Cache Refresh is a method to make KV caching viable for bidirectional attention in diffusion language models. Instead of treating cached KV states as static (as in Dual Cache), it periodically recomputes KV for a small vicinity around the currently decoding block: the masked tokens themselves and their immediate neighbors (e.g., ±16 tokens). After a block finishes decoding, a full cache update ensures global consistency. This approach reduces the accuracy degradation caused by stale cached states while still providing significant compute savings compared to no caching.

## Why it matters

In autoregressive models, KV caching is straightforward: once computed, a token's KV stays unchanged. In dLLMs, any token prediction can affect all other tokens via bidirectional attention, making static caching invalid. Without caching, each forward pass must recompute the entire sequence's Transformer, which is prohibitively expensive. Vicinity refresh addresses this by:
- Reusing cached KV for distant tokens (unchanged)
- Recomputation only for tokens that could be affected by recent changes (vicinity)
- Achieving substantial speedup while maintaining accuracy

## Mechanism

Let the current decoding block be `[start:end]`. The Vicinity refresh strategy:
- **Prefix look**: recompute KV for tokens `[: start - prefix_look]` (tokens before the block, excluding a prefix buffer)
- **After look**: recompute KV for tokens `[end + after_look :]` (tokens after the block)
- Additionally, all masked tokens within the block are recomputed as part of the forward pass.
- After the block completes decoding, issue a full cache update for the entire block (now with new token values) to ensure consistency for future passes.

The idea: tokens far from the block are semantically less likely to be affected by changes within the block; skipping their recomputation is safe.

Parameters: `prefix_look` (default 16), `after_look` (default 16), `warmup` (number of iterations before enabling refresh, default 4). Warmup allows initial stabilization.

## Trade-offs

- **Accuracy**: Vicinity refresh is a heuristic; in some cases, changes may propagate beyond the vicinity, causing subtle drift. However, empirical results show it matches or exceeds Dual Cache accuracy.
- **Memory**: KV cache still stored for entire sequence; no extra memory overhead vs other caching strategies.
- **Computation**: Each iteration recomputes a window of size `prefix_look + block_size + after_look`. This is much smaller than full sequence, yielding net savings.
- **Hyperparameters**: `prefix_look` and `after_look` control the trade-off between speed and accuracy; larger windows increase recomputation but improve accuracy.

## Apple Silicon implications

- Vicinity refresh is algorithm-agnostic; it can be implemented on any hardware, including Metal.
- On Apple Silicon, memory bandwidth is lower than on H100; recomputation vs memory trade-off may shift. Smaller vicinity windows might be optimal.
- KV cache resides in unified memory; careful management needed to avoid thrashing.
- Could combine with per-block quantization to reduce KV size and bandwidth.

## Open questions

- How does accuracy vary with `prefix_look` and `after_look` across sequence lengths? Is a fixed 16 optimal for all contexts?
- Could the vicinity window be adaptive based on attention patterns? For example, tokens with high attention weights to the block should be included.
- Does vicinity refresh interact positively with iteration smoothing? Smoothing enriches embeddings; does that increase the needed vicinity size?
- How does vicinity refresh compare to learned cache invalidation (e.g., marking tokens as dirty based on gradient)?
- Can we approximate the "influence zone" of a block theoretically to set vicinity bounds?
- On Apple Silicon, would it be better to recompute entire block plus neighbors in a single kernel launch (megakernel style)?

## Related concepts
- [[editable-state-evolution]] (vicinity refresh is a concrete implementation of selective state updates)
- [[block-wise-causal-attention]] (vicinity tries to approximate the effects of full attention)
- [[per-block-fp8-quantization]] (quantizing KV could reduce recomputation cost)
- [[dinfer-framework]] (KV-cache manager component)
- [[mega-kernel-v1-fused-remask-sample]] (a megakernel might fuse vicinity recompute operations)

## Related pages

- [[dinfer-framework]] (source note)
