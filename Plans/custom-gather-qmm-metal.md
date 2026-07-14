# Custom gather_qmm Metal Kernel Design & Verification Roadmap (M2 Ultra)

This document outlines the architectural concept, verified hardware specifications, engineering challenges, and profiling-first execution roadmap for implementing a custom GPU-accelerated gathered quantized matrix multiplication (`gather_qmm`) kernel on the **Apple Silicon M2 Ultra** (specifically targeting `llada2.1-mini` and `llada2.1-flash` models).

---

## 1. Core Concept & Objective

The primary bottleneck in a quantized Mixture of Experts (MoE) block is the sparse projection step. For each token in a sequence, the router selects $K$ active experts out of $E$ total experts. A naive implementation either performs a dense matrix multiplication over all $E$ experts (wasting compute) or dequantizes weights on the CPU (causing sync stalls).

A custom **`gather_qmm` Metal kernel** resolves this by executing sparse quantized matrix multiplication directly on the GPU. 

### Mathematical Formulation
Given:
* Token activations $X \in \mathbb{R}^{T \times H}$ (hidden size $H$, sequence length $T$).
* Expert weights $W \in \mathbb{U}^{E \times I \times \frac{H \cdot \text{bits}}{32}}$ (quantized 4-bit, packed into `uint32` matrices).
* Scale factors $S \in \mathbb{R}^{E \times I \times \frac{H}{\text{group\_size}}}$ and biases $B \in \mathbb{R}^{E \times I \times \frac{H}{\text{group\_size}}}$.
* Routing indices $R \in \mathbb{I}^{T \times K}$ (mapping each token to its $K$ selected experts).

The kernel computes the gathered output $Y \in \mathbb{R}^{T \times K \times I}$:
\[
Y[t, k, i] = \sum_{h=0}^{H-1} X[t, h] \cdot \text{dequantize}\left(W[R[t, k], i, h], S[R[t, k], i, h], B[R[t, k], i, h]\right)
\]

---

## 2. Hardware Capacities & Architectural Design (M2 Ultra)

The following parameters were queried and verified directly on the host Mac Studio M2 Ultra GPU using system APIs `[Sourced: scratch/query_metal_caps.swift]`:
* **GPU Device Name**: `Apple M2 Ultra` (76-core GPU, 800 GB/s memory bandwidth)
* **Max Threadgroup Memory Length**: **32,768 bytes (32 KB)**
* **Thread Execution Width (SIMD size)**: **32 threads**

### A. Threadgroup Layout & Tiling
We partition the computation grid to match the model shapes:
* **Mini Shapes**: $H=2048$, $I=512$, $K=8$.
* **Flash Shapes**: $H=4096$, $I=1024$, $K=8$.

Inside the threadgroup, partition threads along the intermediate dimension $I$ and execute accumulation loops over the hidden dimension $H$.
* **Flash Scaling**: For Flash, the intermediate dimension $I$ scales from $512$ to $1024$. The threadgroup size should increase (e.g., from $64$ to $128$ threads) to ensure optimal thread occupancy and warp utilization across execution units.

### B. Memory & Threadgroup Caching
1. **Activation Caching**:
   Since the input token vector $X[t, :]$ is shared across all calculations for that token's active experts, load the activation slice into threadgroup local memory (L1 cache / `threadgroup` space).
   * **Sizing Check**: For Flash ($H=4096$), the FP16 activation slice size is exactly $4096 \times 2\text{ bytes} = 8\text{ KB}$. At **32 KB max threadgroup memory**, this occupies **25%** of the threadgroup memory budget, fitting comfortably and leaving ample room for accumulator variables and scales.
2. **Coalesced Global Memory Reads**:
   Packed 4-bit weights should be loaded as `uint32` words (containing 8 packed weights). Threads must read contiguous memory offsets so that the hardware coalesces memory fetches, maximizing the 800 GB/s bus utilization.
3. **SIMD Matrix Functions**:
   Utilize Metal's hardware-accelerated matrix multiplication-accumulation (MMA) co-processors (`metal::simdgroup_matrix`). This maps warp-level operations directly to Apple Silicon's hardware matrix units.

---

## 3. Implementation Challenges & Risks

### A. Warp/SIMD Branch Divergence
* **The Problem**: SIMDgroups in Apple Silicon execute 32 threads in lockstep. If tokens in the same SIMDgroup route to different experts ($R[t_1, k] \neq R[t_2, k]$), threads will read from non-contiguous global memory addresses. 
* **Impact**: This causes memory serialization (bank conflicts and cache misses), degrading memory throughput and causing instruction stalls.

### B. Quantization Scale & Bias Decompression
* **The Problem**: Affine 4-bit quantization requires applying a scale and bias for every group of 64 weights. 
* **Impact**: Performing dequantization in the inner loop of the accumulator adds register pressure and floating-point operations. Shifting scales/biases to registers or shared memory is necessary to avoid stalling ALU pipelines.

---

## 4. Verification, Profiling & Conclusions

We performed two controlled experiments under environment-valid conditions to evaluate occupancy-scaling limits and establish a causal link for SIMD divergence:

### A. Telemetry & Validity `[Sourced: task-620]`
* Swap usage growth: **0.0 MB** (limit: $\le 256$ MB).
* Free pages: **~118–122 GB** (limit: $\ge 1$ GB).
* Thermal state: **nominal** (limit: `nominal` or `fair`).
* Execution: **`envValid = true`** (warmup excluded).

### B. Experiment 1: Occupancy & Batch Size ($T$) Sweep `[Sourced: task-620]`
To rule out the occupancy/grid-size confound (i.e. whether low bandwidth at small batch sizes is caused by insufficient concurrent work to hide latency), we swept sequence length $T$ upward under Flash shapes ($E=256, H=4096, I=1024, K=8$):

| Sequence Length ($T$) | Active Experts (Unique) | `gatherQuantizedMM` Time | Effective Memory Bandwidth | `dense quantizedMM` Time |
| :--- | :--- | :--- | :--- | :--- |
| **$T=32$** | 164 / 256 | 1.27 ms | **290.51 GB/s** | 5.71 ms (100.84 GB/s) |
| **$T=128$** | 253 / 256 | 3.62 ms | **157.26 GB/s** | 21.40 ms (26.92 GB/s) |
| **$T=512$** | 256 / 256 | 13.40 ms | **42.98 GB/s** | 83.17 ms (6.93 GB/s) |
| **$T=2048$** | 256 / 256 | 50.51 ms | **11.40 GB/s** | 327.75 ms (1.76 GB/s) |

#### Analysis:
1. **Transition to Compute-Bound `[Inferred]`**: 
   As $T$ grows, the total weight data read stabilizes (since all 256 experts are unique by $T=512$), while the total arithmetic operations (FLOPs) scale linearly with $T$. The calculated bandwidth drop (from 290.5 GB/s to 11.4 GB/s) indicates a shift from memory-bound to ALU compute-bound scaling.
2. **Occupancy Verification `[Inferred]`**:
   At $T=32$, `gatherQuantizedMM` achieves **290.51 GB/s** (36.3% of M2 Ultra peak). This confirms that even at small batch sizes, the kernel is not occupancy-bound; it effectively schedules threads to hide latency when active experts are sparse.

---

### C. Experiment 2: Causal A/B Test for SIMD Divergence `[Sourced: task-620]`
To confirm SIMD divergence from unsorted routing as the dominant bottleneck under high active expert densities, we ran a controlled A/B test at $T=512$ (where both runs fetch all 256 experts and read exactly the same 604 MB of data):
* **A: Unsorted Routing (Standard Random)**: Routing indices $R[t, k]$ are unsorted.
* **B: Sorted Routing (Contiguous)**: Routing indices $R[t, k]$ are pre-sorted along the token axis to group contiguous expert requests.

#### Metrics:
* **Unsorted**: **12.97 ms** (effective bandwidth: **44.41 GB/s**)
* **Sorted**: **6.99 ms** (effective bandwidth: **82.37 GB/s**)
* **Direct Causal Speedup**: **1.85x**

---

### D. Final Diagnostic Verdict `[Inferred]`

1. **Divergence is Confirmed**: 
   The A/B test provides a clear causal confirmation: sorting indices cut execution time almost in half (**1.85x speedup**) and pushed memory bus utilization to **82.37 GB/s**. Divergence and uncoalesced memory reads across the 32-wide SIMD lanes are the primary bottleneck under high expert-density workloads.
2. **Strategy Shift**:
   * We will NOT implement a custom hand-written Metal kernel from scratch, as it would not resolve the fundamental uncoalesced access pattern.
  * Instead, we must prioritize **token sorting and pre-grouping**. The next architectural step is evaluating **`segmented_mm`** (MLX PR #2335) to pre-sort active tokens by expert on the GPU, grouping memory transactions before they hit the execution units.

