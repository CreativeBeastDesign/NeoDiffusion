# Alpha-MoE Megakernel Technical Report

**Source**: Aleph Alpha technical report (December 2025)  
**Title**: "Alpha-MoE: A megakernel for faster tensor parallel inference"  
**Authors**: Szymon Ożóg, Eric Schreiber, Lukas Blübaum  
**Web**: https://github.com/Aleph-Alpha/Alpha-MoE  
**PDF**: `01-Inbox/Alpha-MoE_A-Megakernel-for-Faster-Tensor-Parallel-Inference_Report.pdf`  
**Status**: stable  
**Ingested**: 2026-04-16

---

## Summary

Alpha-MoE is a fused MoE megakernel library for FP8 W8A8 inference that combines Up-Proj + Gate GEMM → SwiGLU → activation quantization → Down-Proj into a single persistent kernel. It delivers up to 200% speedup over existing Triton kernels in vLLM and SGLang, particularly in tensor-parallel (TP) deployments with high TP sizes. The kernel is optimized for NVIDIA Hopper architecture.

## Key claims

1. **Fusion of MoE operations**: By fusing steps 2–6 of a MoE layer (Up+Gate GEMM, SwiGLU, activation quantization, Down GEMM, local combine) into one megakernel, global memory traffic is drastically reduced.
2. **SwiGLU Weight Interleaving**: Interleaving Up and Gate weight matrices in 8-row chunks ensures both GEMM outputs land on same thread, enabling SwiGLU without extra memory overhead.
3. **Persistent Down Projection**: For high TP with aggressive column sharding (e.g., moe_intermediate_size shard = 256), store full Up+Gate results in shared memory; Down Projection runs within a single block with overlapped computation/memory fetches.
4. **Tensor Parallelism efficiency**: Solves the low-dimensionality problem in TP that shifts operations from compute-bound to memory-bound.
5. **End-to-end gains**: Benchmarks on Qwen3-Next-80B and DeepSeek-R1 show consistent throughput improvements across batch sizes and expert balance.

## Mechanisms

### MoE layer steps normally:
1. Routing (gating)
2. Up Projection + Gate GEMM (two matrices)
3. SwiGLU activation
4. Quantization for next GEMM
5. Down Projection GEMM
6. Local combine on device
7. AllReduce for global combine

Alpha-MoE fuses steps 2–6. Steps 1 and 7 remain separate (routing and AllReduce).

### Technical innovations

- **SwiGLU Weight Interleaving**: Uses Hopper's WGMMA matrix tile layout where each thread holds two 8-row-separated tiles. By interleaving weight matrices in 8-row chunks, both Up and Gate outputs appear on same thread after first GEMM stage.
- **Persistent Down Projection**: With small `moe_intermediate_size` shards (e.g., 256), the Up+Gate result fits in shared memory. Down Projection GEMM is then performed within a single thread block, keeping activations resident in shared memory and overlapping row computation with global fetches.
- **Producer/Consumer pipelines, multi-stage loading, async stores**: Standard Hopper optimizations leveraged within the megakernel.

## Hardware assumptions

- NVIDIA Hopper architecture (H100) with WGMMA, tensor cores, and fast shared memory.
- FP8 W8A8 quantization (weights and activations in 8-bit float).
- Tensor Parallelism with high TP size (e.g., TP16) where matrix shards become small.

## Extracted concepts

- Mixture of Experts (MoE) inference
- MoE kernel fusion (megakernel)
- SwiGLU activation function
- FP8 W8A8 quantization
- Tensor Parallelism (TP) vs Expert Parallelism (EP)
- WGMMA (Warp-Group Matrix Multiply-Accumulate)
- Persistent kernel design
- Shared memory tiling
- AllReduce in MoE
- Routing and gating mechanisms

## Relevance to dLLMs

- LLaDA2.1-Flash (100B) uses MoE; Alpha-MoE's fusion techniques could inspire similar megakernel design for dLLM operations (M2T/T2T/MBE).
- The SWIGLU weight interleaving trick is architecture-specific but conceptually could be adapted to other fused operations (e.g., fusing attention and LM head).
- Persistent down projection suggests keeping intermediate activations in shared memory for subsequent kernels—applies to multi-block editing or ELBO computation.

## Open questions (raised by this source)

- How does Alpha-MoE's fusion compare to SGLang's unfused implementation on Hopper? What is the breakdown of gains from interleaving vs persistent down projection?
- Can the fusion approach be generalized to other activations beyond SwiGLU? What about Gated Linear Units in dLLMs?
- For Apple Silicon (Metal), there is no WGMMA; can similar fusion be achieved with compute shaders and threadgroup memory? What are the performance trade-offs?
- Does the 200% speedup hold for mixed-precision (FP16/BF16) or is it specific to FP8 W8A8?
- How does the megakernel's register pressure affect occupancy on Hopper? What would the occupancy be on Apple Silicon GPUs?
- Could Alpha-MoE's techniques be applied to fuse multiple dLLM steps (e.g., M2T + T2T editing) into a single megakernel?

## Suggested next probes

- Extract detailed performance numbers from the report's charts for TP16, different batch sizes, and expert balance.
- Understand the exact mapping of weight matrix interleaving to Hopper's WGMMA tile layout.
- Investigate SGLang's current MoE implementation to identify exactly what Alpha-MoE replaces.
- Compare Alpha-MoE's AllReduce handling with typical MoE implementations.
- Assess whether the "persistent down projection" technique can be applied to the down-projection in dLLMs' Multi-Block Editing (i.e., keep earlier block activations in shared memory for editing passes).

## Related pages

- [[alpha-moe-megakernel]] (concept page – may need update to include report details)
- [[mega-kernel-v1-fused-remask-sample]] (proposal for dLLMs)
- [[mega-kernel-v1-fused-remask-sample]] (proposal – see if Alpha-MoE's fusion pattern influences design)