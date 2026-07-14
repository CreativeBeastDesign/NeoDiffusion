# MoE Dispatch Benchmarks & Performance Conclusions (M2 Ultra)

This document records the benchmark results of various Mixture of Experts (MoE) dispatch mechanisms evaluated on the Mac Studio M2 Ultra, along with the performance conclusions derived from them.

---

## 1. Environment & Telemetry Validity

Every benchmark run captured system memory and thermal telemetry to guarantee that macOS paging, memory pressure, or thermal throttling did not inject timing anomalies:
* **Host Hardware**: Mac Studio M2 Ultra (`Mac14,14` with 192 GB Unified Memory, 76-Core GPU).
* **Validation Parameters**:
  * Swap growth during run: **0.0 MB** (limit: $\le 256$ MB).
  * Free memory at start: **>150 GB** (limit: $\ge 1024$ MB).
  * OS thermal state: **nominal** (limit: `nominal` or `fair`).
  * Environment Validity: **`envValid = true`** for all recorded rows.

---

## 2. Measured Dispatch Timings

The micro-benchmarks evaluated three mathematical-equivalent implementations for sparse expert projection:
1. **`gatherQuantizedMM` (Production Path)**: The native MLX gathered quantized matrix multiplication.
2. **Dequantize + `gatherMM`**: Dequantizes active experts to 16-bit float first, followed by a standard gathered matrix multiplication.
3. **Dense `quantizedMM` (ALL Experts)**: Performs quantized matrix multiplication over all 256 experts at once (compute-everything upper bound).

### A. Flash Shapes
*Dimensions: $E=256$ experts, $I=1024$ intermediate dimension, $H=4096$ hidden dimension, $T=32$ active tokens, $K=8$ active experts per token.*

| Dispatch Variant | Time per Op (s/op) | Time per Op (ms) | Relative Speed |
| :--- | :--- | :--- | :--- |
| **`gatherQuantizedMM`** | **0.0014** | **1.4 ms** | **1.0x** (Fastest) |
| **Dequantize + `gatherMM`** | 0.0058 | 5.8 ms | 4.14x slower |
| **Dense `quantizedMM` (ALL Experts)** | 0.0056 | 5.6 ms | 4.00x slower |

* Parity check: Max delta $|\Delta| = 3.33786 \times 10^{-6}$ (Passes parity gate).

### B. Mini Shapes
*Dimensions: $E=256$ experts, $I=512$ intermediate dimension, $H=2048$ hidden dimension, $T=32$ active tokens, $K=8$ active experts per token.*

| Dispatch Variant | Time per Op (s/op) | Time per Op (ms) | Relative Speed |
| :--- | :--- | :--- | :--- |
| **`gatherQuantizedMM`** | **0.0006** | **0.6 ms** | **1.0x** (Fastest) |
| **Dequantize + `gatherMM`** | 0.0017 | 1.7 ms | 2.83x slower |
| **Dense `quantizedMM` (ALL Experts)** | 0.0017 | 1.7 ms | 2.83x slower |

* Parity check: Max delta $|\Delta| = 1.90734 \times 10^{-6}$ (Passes parity gate).

---

## 3. Core Performance Conclusions

### A. Native MLX Dispatch Efficiency `[Sourced]`
* On the M2 Ultra GPU, the native `gatherQuantizedMM` custom Metal kernel is highly optimized. It remains the fastest dispatch choice by a significant margin, outperforming the dequantization fallback path by **$2.8\times$ on Mini** and **$4.1\times$ on Flash**.

### B. MoE is Not the Step Latency Bottleneck `[Inferred]`
* For the Flash model (32 layers, 3 expert projections per layer = 96 routed projections per forward step):
  * **MoE compute time per step**: $96 \times 1.4\text{ ms} \approx 134.4\text{ ms}$.
  * **Measured overall step latency**: **$1.87\text{ s}$**.
  * **Conclusion**: The MoE block accounts for only **~7.2%** of the step execution time. The primary bottleneck driving the remaining $1.73\text{ s}$ of latency resides in other parts of the execution graph, such as dynamic graph recompilation overhead from variable sequence/block sizes, block-causal attention caching, or the unquantized F16 LM Head projection.
