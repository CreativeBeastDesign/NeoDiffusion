# FreeCache: Reducing Window KV Caching for Diffusion Language Models

**Summary**: A training-free KV caching strategy for diffusion LLMs that exploits the temporal stability of clean token projections by progressively freezing blocks and shrinking the active computation window.  
**Aliases**: FlashDLM FreeCache, reducing window cache, block-wise KV freezing  
**Status**: emerging  
**Last updated**: 2026-04-16  
**Sources**:
- [[flashdlm-tech-report]]

---

## Definition

FreeCache is a KV caching approximation technique specifically designed for masked diffusion language models (dLLMs). It partitions the generation sequence into fixed-size blocks (e.g., 256 tokens). During iterative denoising, once a block has been fully unmasked, its Key-Value projections become stable and are frozen for all subsequent steps. The active computation window at each step is defined as the current block being generated plus all later blocks; earlier blocks are excluded from the window and their cached KV is simply read without recomputation. This reduces the per-step computational cost as generation progresses, achieving up to 6.32× speedup on LLaDA-8B-Instruct with minimal accuracy degradation.

## Why it matters

dLLMs cannot use standard KV caching because the masking pattern changes every step; even unmasked tokens might need to attend to future masked positions that will later become unmasked. As a result, dLLMs recompute the full forward pass over the entire sequence at every denoising step, leading to O(L²) attention cost per step and high latency. FreeCache exploits the observed temporal stability of KV projections for clean tokens: after a token is unmasked and undergoes a few refinement steps, its key and value vectors change very little. Therefore, recomputing these stable tokens provides diminishing returns. By freezing entire blocks once they are complete, FreeCache avoids redundant computation while maintaining generation quality.

## Mechanism

### Core insight: temporal stability of KV projections

- Heatmap analysis (Figure 2 in FlashDLM) shows that the similarity (cosine) of a token's value projection to its previous step increases as the token stabilizes.
- Prompt tokens are stable from the start.
- Generated tokens start unstable (dark blue) but become stable (bright yellow) after a few steps once they are mostly correct.
- This allows caching approximations: once a token is stable, we can reuse its KV from the previous step instead of recomputing.

### Block partitioning and active window

1. Partition the generation sequence (excluding prompt) into N fixed-size blocks B₁, ..., Bₙ (e.g., 256 tokens each).
2. Initial forward pass: compute KV for the entire prompt + all blocks (full cost).
3. For block Bᵢ (i from 1 to N):
   - The active window Wᵢ = Bᵢ ∪ Bᵢ₊₁ ∪ ... ∪ Bₙ. The prompt is also always part of the context but its KV is already frozen (since it never changes).
   - For each denoising step while Bᵢ is not yet fully unmasked:
     - Only recompute KV projections for tokens in Wᵢ (i.e., tokens in current block and all later blocks).
     - KV for tokens in B₁..Bᵢ₋₁ (completed blocks) are read from cache and reused.
   - Once Bᵢ is fully unmasked, mark it as frozen; future steps will not recompute its KV.
   - Move to next block Bᵢ₊₁.
4. The active window size decreases as more blocks freeze, leading to progressively lower per-step FLOPs.

### Algorithmic sketch

```
Initialize: Partition generation length L into blocks B[1..N] of size B_size.
Compute initial KV for prompt + all tokens (full forward).
for i = 1 to N:
    active_window = B[i] ∪ B[i+1] ∪ ... ∪ B[N]
    while B[i] not fully unmasked:
        // Attention computation uses:
        // - Frozen KV from B[1..i-1] (from cache)
        // - Active KV for active_window (recompute this step)
        forward_pass(active_tokens = active_window)
        update_unmasking()
    freeze KV of B[i]
    // active_window shrinks automatically for next iteration
```

### Relationship to other block-based methods

- **Multi-Block Editing (MBE)**: LLaDA2.1 uses MBE to process blocks sequentially but still recomputes KV for later blocks each step; MBE does not freeze KV. FreeCache can be combined with MBE by applying the windowing within the MBE framework.
- **Elastic-Cache**: Freezes layers based on a drift test; FreeCache freezes entire blocks regardless of layer-wise drift. Could be combined: use Elastic-Cache to decide when a block is stable enough to freeze, or use per-layer freezing within blocks.
- **FastDLLM**: Uses caching approximations but not explicit block partitioning; FreeCache is more systematic about progressive reduction.

## Trade-offs

- **Speed vs. accuracy**: FreeCache introduces a small accuracy drop due to KV approximation (e.g., ~2% on GSM8K for Dream-7B: 79.68% → 77.40%). However, when combined with Guided Diffusion, accuracy can even improve due to better coherence.
- **Block size**: Larger blocks mean the active window shrinks less frequently (fewer blocks), but each block takes longer to complete; smaller blocks increase the number of freeze events but may cause more overhead from managing many blocks. Optimal size likely depends on cache hierarchy (e.g., L2 size).
- **Memory footprint**: Must store KV for all blocks until they freeze; peak memory is similar to baseline because all KV are allocated initially. However, after blocks freeze, they remain in memory; total memory usage is essentially the KV for the full sequence (same as baseline). The benefit is compute, not memory reduction.
- **Generalization**: The temporal stability observation might vary across models and tasks; but paper shows consistent results across Dream and LLaDA, suggesting it's a general property of dLLMs.

## Hardware implications

- **Compute reduction**: The active window's size decreases over time, reducing total FLOPs per step. The later blocks (higher index) will have smaller active windows, so final stages are much cheaper.
- **Memory access pattern**: For each step, we need to load KV from frozen blocks (read-only) and recompute KV for active window. This is a mix of streaming loads (active tokens) and random accesses (frozen block KV read multiple times). On Apple Silicon, frozen block KV should ideally reside in the GPU's L2 cache or shared memory to avoid bandwidth bottlenecks.
- **Kernel fusion**: Each step's attention computation could be implemented as a single kernel that conditionally recomputes KV for active tokens and reuses cached KV for frozen tokens. Requires careful indexing.
- **Parallelism**: Within a step, all tokens in the active window are processed in parallel (standard transformer). The reduction in active window size directly reduces the parallel workgroup size? Not exactly: the sequence length is still L, but the attention computation only needs to recompute Q/K/V for active tokens; however, the attention matrix still has size L² because every token attends to all others. Actually, the paper's description: "To generate a block Bᵢ, the active computation window is defined as Bᵢ and all subsequent blocks. KV projections are recomputed only for tokens within this window until Bᵢ is fully unmasked, using all prior frozen blocks and the prompt as context." This suggests that the attention computation still considers the full context (prompt + frozen blocks + active window), but the KV for frozen blocks are cached, so we avoid recomputing their Q/K/V? In standard transformer, we compute Q, K, V for all tokens each step. FreeCache likely avoids recomputing Q/K/V for frozen tokens; they are read from cache. So per-step FLOPs reduce roughly in proportion to the fraction of frozen tokens.
- **Memory bandwidth**: The KV cache for frozen tokens must be read each step; if the frozen set is large, this could still be bandwidth-intensive. However, it's better than recomputing because recomputation would be more FLOPs and also require reading weights again.

## Apple Silicon implications

- Applicable to LLADA2.1: same architecture as LLaDA; temporal stability likely holds.
- Block size selection: Should align with Apple Silicon's L2 cache size per core (likely ~1-2 MB). For d=4096, FP16 KV per token = 2×2×4096 = 16KB. A block of 256 tokens = 256×16KB = 4MB for both K and V? Actually KV cache per token is 2*d*sizeof(dtype) (for K and V). With d=4096, FP16: 2*4096*2 = 16384 bytes = 16KB per token. For 256 tokens: 256*16KB = 4096KB = 4 MB. That might exceed L2 per core but could fit in L3 or be streamed from unified memory. Could tune block size smaller (e.g., 128 or 64) to fit in shared memory.
- FreeCache could be combined with per-block FP8 quantization: frozen block KV could be quantized to reduce memory traffic at the cost of small accuracy loss. Since they are stable, quantization errors might not accumulate.
- Implementation in Metal: use a compute shader that processes the active window; frozen block KV stored in device memory but ideally cached. Could use a ring buffer or texture memory for caching.
- FreeCache's progressive reduction complements other acceleration: combine with Guided Diffusion, ICE early exit, or Model Scheduling.
- Open questions:
  - What is the actual speedup on Apple Silicon hardware? The paper reports on NVIDIA RTX 6000 Ada; memory hierarchy differs.
  - Can we dynamically adjust block size based on observed stability? Or use a hybrid where some tokens within a block freeze earlier based on per-token confidence.
  - How does FreeCache interact with hierarchical decoding? Hierarchical decoding already processes blocks in a specific order; FreeCache's windowing might need adaptation.

## Related concepts

- [[dkv-cache]] (delayed KV caching; also exploits stability)
- [[fast-dllm]] (caching approximations for dLLMs)
- [[elastic-cache]] (adaptive layer-wise caching; could complement FreeCache)
- [[multi-block-editing-mbe]] (block partitioning for dLLMs; FreeCache uses blocks differently)
- [[vicinity-kv-cache-refresh]] (caches only nearby tokens; FreeCache caches entire blocks)
- [[radix-caching]] (cache management strategy; could be applied to FreeCache's frozen blocks)
- [[llada2-1-tech-report]] (base model)

## Extracted concepts from this source

- Reducing window caching
- Temporal stability of KV projections in dLLMs
- Block-wise KV freezing
- Progressive computation reduction
- Training-free KV approximation

## Open questions

- What is the optimal block size for Apple Silicon's memory hierarchy? Should it be tuned to L2 cache size per core? How does block size affect speed vs accuracy?
- Can FreeCache be combined with Elastic-Cache's drift test to freeze tokens adaptively rather than whole blocks? Might improve accuracy.
- Could we quantize frozen block KV (e.g., FP8 or INT8) to reduce memory bandwidth further? What accuracy trade-off?
- How does FreeCache interact with Multi-Block Editing (MBE)? Could MBE's block-wise processing be combined with FreeCache's windowing?
- Does FreeCache work with different noise schedules (cosine, sigmoid) and different sampling strategies (MaskGIT, entropy-sum)? The stability property might vary.
- How does FreeCache scale to very long sequences (>4k tokens)? More blocks means later blocks have smaller active windows, which is good; but initial blocks have many active windows, so the total number of recomputations might still be high.
- Could we use FreeCache with hierarchical decoding where small blocks are processed first? The freezing schedule might change.
- What is the memory bandwidth requirement on Apple Silicon? If frozen KV must be read from DRAM each step, it could become a bottleneck. Can we keep enough frozen KV in on-chip memory?
- Could we implement FreeCache with a "lazy" update: only recompute frozen block KV if a drift threshold is exceeded (like Elastic-Cache)? This would handle cases where a frozen token later needs adjustment.
- How does FreeCache compare to simply reducing the number of denoising steps? Both reduce total work; FreeCache reduces per-step work; can be combined with step reduction.
- Is the temporal stability property universal across all dLLM architectures? Need to test on LLADA2.1.
- Could FreeCache be applied to continuous diffusion (e.g., image generation)? Possibly, but stability of hidden states might differ.
- Implementation: How to efficiently manage the active window in Metal kernels? Need to handle variable-sized active sets; could use dynamic parallelism or pre-defined block schedules.
- Does FreeCache affect the quality of the final sequence differently across tasks? The paper shows minimal accuracy changes on GSM8K, MMLU; but might be more sensitive for creative generation.
- What is the energy consumption impact on battery-powered devices? Reducing FLOPs should reduce power, but memory accesses might dominate.
- Could we use FreeCache with Model Scheduling, where the DLM switches between heavy and light weights? The caching strategy might need to reset when switching models.
- How does FreeCache interact with attention sparsity patterns? If we use sparse attention, the benefit of caching might change.

## Related pages

- [[flashdlm-tech-report]] (source note)
- [[fast-dllm]]
- [[dkv-cache]]
- [[elastic-cache]]
- [[multi-block-editing-mbe]]
- [[radix-caching]]
- [[llada2-1-tech-report]]
