# Block-Wise Causal Attention

**Summary**: Attention mechanism that computes KV cache for long contexts block by block, enabling efficient inference on sequences longer than GPU memory.  
**Aliases**: block attention, chunked causal attention  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[llada2-1-tech-report]]

---

## Definition

Block-Wise Causal Attention processes long input sequences by dividing them into blocks. The KV cache is computed incrementally block by block, with each block attending only to itself and preceding blocks (causal). This allows the model to handle sequences that exceed GPU memory limits because not all KV states need to be resident simultaneously.

## Why it matters

LLaDA2.1 uses an "expansive context window" (exact size not stated but implied large). Standard attention would require storing all KV states, which scales linearly with sequence length. Block-wise attention reduces memory footprint by streaming blocks through the GPU, computing attention on-the-fly. This is essential for long-context tasks and for enabling Multi-Block Editing where earlier blocks may be revisited.

## Mechanism

- Split sequence into contiguous blocks of size B.
- For block i:
  - Compute K_i, V_i from tokens in block i.
  - For each query in block i, attend to K_j, V_j for all j ≤ i.
  - Can be implemented by caching K, V for previous blocks in GPU memory (if they fit) or recomputing them on demand.
- During generation, new blocks are appended and processed sequentially.
- For editing (MBE), may need to reload earlier blocks' KV states.

## Trade-offs

- **Memory**: Reduces peak memory by factor ≈ 1/(#blocks) if KV states are not all kept.
- **Compute**: Recomputation of earlier blocks adds overhead if KV not cached.
- **Complexity**: Block boundaries require careful handling of attention masks.
- **Bandwidth**: Streaming blocks increases memory traffic compared to resident KV cache.

## Apple Silicon implications

- Apple Silicon GPUs have limited but fast unified memory; block-wise attention can help fit larger contexts.
- Block size should be chosen to maximize occupancy within threadgroup and cache hierarchies.
- Recomputation tradeoff: Apple's memory bandwidth may make recomputing cheaper than storing all KV, depending on block count.
- MBE's need to revisit earlier blocks suggests keeping some KV cache might be worthwhile.

## Related concepts
- [[multi-block-editing-mbe]]
- [[radix-caching]]
- [[per-block-fp8-quantization]] (memory savings)

## Open questions
- What is the optimal block size for Apple Silicon's L1/L2 cache and threadgroup size?
- Should KV be kept in FP8 to save memory, and what is the quality impact?