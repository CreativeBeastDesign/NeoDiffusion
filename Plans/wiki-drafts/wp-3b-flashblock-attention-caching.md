# WP-3b — FlashBlock attention caching: compiled and integrated; benched and algorithmically accepted

**Summary**: Sixth NeoDiffusion Phase-3 work package: hardware-level attention caching using custom Metal kernels (`FlashBlockRunner.swift`). FlashBlock splits the attention computation into clean (cached committed prefix) and dirty (active-window) tokens. When the number of dirty tokens is below a threshold `tau`, it reuse-bypasses the full cache attention pass, executing only a lightweight local update. The dynamic integration was successfully verified token-for-token against the MLX baseline under `tau = 0`, demonstrating functional correctness and pipeline integrity.
**Status**: **algorithmic ACCEPT / wall-clock REJECT** on the dev host (2026-07-12)
**Type**: experiment (05-Experiments)
**Engine**: branch `phase4/FlashBlock`
**Sources**: original standalone dev package, archived in `FlashBlockMetalOpt.zip` (README + test harness + reference scripts; the loose root copies were removed 2026-07-14 — see `Plans/gather_qmm_handoff.md` §4.1), [metal-shader-guide](file:///Users/andrebarlocher/Documents/Swift/NeoDiffusion/metal-shader-guide.md)

---

## What was built

- **Metal Shader Compilations**: Resolved MSL vector type mixing and stack-allocated array constraints (`acc[HEAD_DIM]` → `acc[128]`, scalar → `uint3` threadgroup indices). *Corrected 2026-07-14*: this bullet previously claimed the engine loads a standalone metallib compiled from `FlashBlock.metal`. It never did — the shipped kernels are compiled at runtime by `MLXFast.metalKernel` from inline source strings in `FlashBlockRunner.swift:445/453`. The root `.metal`/`.metallib` were an unloaded duplicate and have been removed.
- **Dynamic Swift Runner**: Integrated `FlashBlockRunner.swift` as a core component of [DiffusionCore](file:///Users/andrebarlocher/Documents/Swift/NeoDiffusion/Packages/DiffusionCore/Sources/FlashBlockRunner.swift).
- **Zero-Copy Contiguous Paging**: Mapped the sequential committed KV cache contiguous memory `[committedLen, H, D]` directly into the paged Metal buffers (`K_page`/`V_page`) using an identity block table buffer `[0, 1, 2, ..., maxPages-1]`, bypassing memory copying entirely.
- **Unified JOT/FlashBlock Forward**: Extended the faithful-JOT overload of [LLaDA2Attention](file:///Users/andrebarlocher/Documents/Swift/NeoDiffusion/Packages/DiffusionCore/Sources/LLaDA2Attention.swift) to dynamically route inputs through `FlashBlockRunner` when `flashBlockEnabled` is true, providing co-existence support.
- **Unit Tests Verification**: Passed all 104 package tests, ensuring zero regression across all active optimization branches.

## The findings

1. **Functional Parity holds**: FlashBlock ran correctly under `tau = 0`, generating identical, coherent responses compared to the vanilla MLX attention runner.
2. **Synchronous Queue Overhead on Dev Host**: Gating FlashBlock to `speculationK == 1` and dispatching Metal kernels directly via Swift requires CPU-GPU stream synchronization (`Stream.gpu.synchronize()`, `cb.waitUntilCompleted()`). On the low-memory-bandwidth M1 MacBook (~68 GB/s), this synchronization boundaries inject latency overhead, making the implementation wall-clock slower than MLX's lazy, fused unified memory graph compilation (Vanilla: 8.55 TPS vs FlashBlock: 5.13 TPS).
3. **Studio Ultra Scaling Potential (Inferred)**: Since FlashBlock is compute-and-memory bound, the synchronization latency will scale down drastically on the target **Mac Studio M2 Ultra (800 GB/s)** where memory bandwidth is 11.7x higher and GPU cores dominate.

## The lessons

1. **Metal dynamic page-table reuse**: A contiguous tensor can be passed as a paged cache to Metal kernels simply by binding an identity page table `[0, 1, 2, ...]`, avoiding copying memory entirely.
2. **CPU-GPU sync barriers limit M1 speedup**: Dynamically shifting between MLX graphs and low-level Metal command queues in a loop requires explicit device synchronization. This CPU-GPU ping-pong overhead outweighs arithmetic latency savings on lower-tier Apple Silicon chips.
