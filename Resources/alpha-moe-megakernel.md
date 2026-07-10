# Alpha-MoE Megakernel

**Summary**: A MoE (Mixture of Experts) megakernel that fuses Up-Proj + Gate GEMM → SwiGLU → activation quantization → Down-Proj into a single persistent kernel for faster tensor-parallel inference.  
**Aliases**: Alpha-MoE, fused MoE kernel  
**Status**: stable  
**Last updated**: 2026-04-16  
**Sources**:
- [[alpha-moe-megakernel-tech-report]]

---

## Definition

Alpha-MoE is a megakernel optimization developed by Aleph Alpha that combines multiple MoE computation stages into a single persistent kernel launch. It is specifically designed for FP8 W8A8 (8-bit weights, 8-bit activations) and NVIDIA Hopper architecture (H100). The kernel delivers up to 200% speed improvement over Triton-based kernels in vLLM and SGLang.

## Why it matters

MoE models (like LLaDA2.1-Flash 100B) have sparse communication patterns that can bottleneck performance. Standard MoE implementations involve multiple kernel launches (routing, up-proj, gate, SwiGLU, quantization, down-proj, combine). Each launch incurs overhead and global memory traffic. Alpha-MoE's fusion significantly reduces both, enabling higher throughput in tensor-parallel deployments where matrix shards are small and memory-bound.

## Mechanism

**Normal MoE layer steps**:
1. Routing tokens to experts (gating)
2. Up Projection + Gate GEMM (two matrix multiplies)
3. SwiGLU activation
4. Activation quantization for next GEMM
5. Down Projection GEMM
6. Local combine on device
7. AllReduce for global combine

**Alpha-MoE fuses steps 2–6** into one persistent megakernel. Gating (1) and AllReduce (7) remain separate.

### Key technical innovations

1. **SwiGLU Weight Interleaving**: To avoid extra memory stores, the Up and Gate weight matrices are interleaved in 8-row chunks. This leverages Hopper's WGMMA tile layout where each thread holds two tiles separated by eight rows. After the first GEMM stage, each thread already contains both Up and Gate outputs, enabling SwiGLU to be applied in-register.

2. **Persistent Down Projection**: In high TP settings with aggressive column sharding (e.g., `moe_intermediate_size` shard = 256), the result of Up+Gate GEMM fits in shared memory. The Down Projection GEMM is then performed within a single thread block, keeping activations resident in shared memory. This eliminates reloading from global memory and overlaps computation of previous row with fetching next row from global memory.

3. **Hopper-specific optimizations**: Uses Producer/Consumer pipelines, multi-stage loading, WGMMA instructions, and async stores.

## Trade-offs

- **Speed**: Up to 2× speedup vs current SGLang implementation; gains most pronounced at large TP sizes (e.g., TP16) and high batch sizes.
- **Memory bandwidth**: Reduced global memory traffic due to fusion and shared memory reuse.
- **Complexity**: Megakernel is highly specialized, requiring deep knowledge of Hopper architecture.
- **Flexibility**: May be tied to specific matrix dimensions and SwiGLU activation; other architectures or activations may need redesign.
- **Register pressure**: Fused operations increase per-thread register usage; must balance occupancy.

## Apple Silicon implications

- Apple Silicon GPUs lack WGMMA and Hopper-specific features; direct port impossible. But the *principles* of megakernel fusion and shared memory residency are applicable.
- For dLLMs on Metal, consider fusing M2T/T2T/MBE operations similarly.
- SwiGLU interleaving trick is dataset-specific; but any paired GEMMs could be interleaved to produce both outputs on same thread for subsequent elementwise fusion.
- Persistent kernel design (keeping activations in threadgroup memory) maps to Metal's `threadgroup` memory.
- FP8 W8A8 quantization may not be natively supported; would require software emulation or conversion. However, memory bandwidth savings remain valuable.

## Open questions

- What is the exact performance breakdown between SwiGLU interleaving and persistent down projection?
- Can the fusion approach be extended to include the gating/routing step to further reduce overhead? (Probably not because routing is data-dependent.)
- How does Alpha-MoE compare to other MoE kernels like vLLM's fused MoE? The report claims 200% faster; need to see raw numbers.
- For dLLMs, could a similar megakernel fuse M2T, T2T, and MBE? What would be the combined operations and data layout?
- On Apple Silicon, what is the achievable speedup from fusing attention + LM head + threshold + update into a single megakernel?

## Related concepts
- [[mega-kernel-v1-fused-remask-sample]] (dLLM-specific megakernel proposal)
- [[per-block-fp8-quantization]] (co-deployed with Alpha-MoE)
- [[fusion-architecture-for-editable-dllms]] (comprehensive fusion proposal)
- [[sglang-rollout-engine]] (Alpha-MoE is used within SGLang)

## Related pages

- [[alpha-moe-megakernel-tech-report]] (source note with full extracted details)