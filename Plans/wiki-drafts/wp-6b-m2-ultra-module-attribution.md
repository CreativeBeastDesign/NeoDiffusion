# WP-6b: MoE Module Attribution & Hardware Scaling (M2 Ultra)

## 1. Executive Summary
- **Goal**: Characterize the performance of the core submodules of `llada2.1-mini` on the Mac Studio M2 Ultra GPU to identify performance bottlenecks and evaluate hardware scaling behavior relative to the M1 MacBook Pro baseline.
- **Methodology**: Execute `LLaDAMoEDispatchBench.testModuleAttribution` in a multi-run profiling loop under verified environmental conditions. Compare measured latencies with M1 baseline data to compute scaling speedups and analyze system bottlenecks.
- **Outcome**: The Mac Studio M2 Ultra achieves a **6.20x speedup** across the major evaluated modules. The LM Head achieves the largest scaling win (**9.14x** speedup) due to M2 Ultra's 800 GB/s memory bandwidth, shifting the primary bottleneck away from memory transfers toward graph compiling/scheduling overhead.

---

## 2. Experimental Results & Telemetry
All runs were completed on the Mac Studio M2 Ultra (M2 Ultra 192 GB unified memory, 76-core GPU) under verified environment constraints:
- Swap usage before/after: **0.0 MB / 0.0 MB** (Swap growth: **0.0 MB** $\le 256$ MB gate)
- Free memory pages at start: **~5,291,000 pages** ($\approx 80.7$ GB $\ge 1024$ MB gate)
- OS thermal state: **nominal** (nominal/fair gate)
- Environment validity: **`envValid = true`**

*Sourced: `NEODIFFUSION_M6_BENCH=1 swift test --filter LLaDAMoEDispatchBench/testModuleAttribution` loops in task-30 and task-40.*

### Measured Submodule Latency (s/op)
Measurements are taken over $T=32$ active tokens, $E=256$ experts, $H=2048$ hidden size, $I=512$ intermediate size, $K=8$ experts per token.

| Submodule | Run 1 | Run 2 | Run 3 | Run 4 | Run 5 | Run 6 | Mean Latency (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MoE Block [1, 32, 2048]** (quantized, warm) | 0.0023 | 0.0024 | 0.0023 | 0.0024 | 0.0024 | 0.0022 | **2.33 ms** |
| **Router Alone [32, 2048]** | 0.0010 | 0.0011 | 0.0010 | 0.0010 | 0.0010 | 0.0010 | **1.02 ms** |
| **LM Head [1, 32, 2048] $\rightarrow$ [1, 32, 157184]** (F16) | 0.0035 | 0.0035 | 0.0035 | 0.0035 | 0.0035 | 0.0035 | **3.50 ms** |
| **Dense FFN [layer-0 shape]** (quantized) | 0.0008 | 0.0008 | 0.0008 | 0.0008 | 0.0007 | 0.0008 | **0.78 ms** |

---

## 3. Scaling Comparison: M1 vs. M2 Ultra

We compare the mean latencies measured on the M2 Ultra against the frozen dev baseline recorded on the M1 MacBook Pro (16 GB unified memory, 8-core GPU). 

*Sourced for M1: `Plans/m6-logbook.md` line 59.*
*Inferred: Speedups calculated via division of sourced M1 and M2 Ultra latencies.*

| Submodule | M1 MBP Baseline (ms) | M2 Ultra Mac Studio (ms) | Hardware Speedup |
| :--- | :--- | :--- | :--- |
| **MoE Block** | 11.30 ms | 2.33 ms | **4.85x** |
| **Router Alone** | 1.70 ms | 1.02 ms | **1.67x** |
| **LM Head** (F16) | 32.00 ms | 3.50 ms | **9.14x** |
| **Dense FFN** | 2.30 ms | 0.78 ms | **2.95x** |
| **Combined Measured Modules** | **47.30 ms** | **7.63 ms** | **6.20x** |

---

## 4. Key Findings & Architectural Analysis

### A. Memory-Bandwidth Saturation of the LM Head `[Sourced / Inferred]`
- The LM Head (`Linear` size `[2048, 157184]` in float16) holds $157,184 \times 2,048 \times 2 \text{ bytes} \approx 643.8 \text{ MB}$ of weights.
- On the M1 MBP (theoretical memory bandwidth: 68 GB/s), loading this matrix requires a theoretical minimum of $9.47\text{ ms}$. The measured time of $32.0\text{ ms}$ represents **~30% bandwidth efficiency** (inferred).
- On the M2 Ultra Mac Studio (theoretical memory bandwidth: 800 GB/s), loading this matrix requires a theoretical minimum of $0.80\text{ ms}$. The measured time of $3.50\text{ ms}$ represents **~23% bandwidth efficiency** (inferred).
- **Conclusion**: The LM Head scales nearly linearly with memory bandwidth (**9.14x speedup** vs. **11.76x bandwidth scaling**). It remains highly memory-bound, making it the highest single-submodule cost on both platforms.

### B. MoE Weight Coalescing and Scheduling Overhead `[Inferred]`
- The MoE block achieves a **4.85x speedup** on the M2 Ultra. This is below the pure memory bandwidth scaling factor of $11.76\times$ (800 GB/s vs 68 GB/s).
- **Analysis**: Because the active token count $T=32$ is small, the dynamic gathering of expert weight slices (`gatherQuantizedMM`) suffers from cache underutilization and launch latency overhead. The M2 Ultra GPU's massive execution width (76 cores) remains under-occupied by small-batch, dynamic gather operations, leaving performance dominated by scheduling overhead rather than raw memory throughput.

### C. Router Compute Constraints `[Inferred]`
- The router (FP32 matmul + sigmoid + group top-k) is the most compute-bound module of the set, achieving only a **1.67x speedup**. This reflects the overhead of the serial sorting operations (top-k) on the GPU, which do not scale well with massive hardware parallelism at small batch sizes.

---

## 5. Architectural Recommendations & Speculative Paths

- **Speculative Path 1 (LM Head Quantization)**: Quantizing the LM Head to 4-bit (reducing weight memory footprint from 643.8 MB to ~161 MB) is projected to reduce LM Head latency from $3.50\text{ ms}$ to **$\approx 0.90\text{ ms}$** on the M2 Ultra, assuming constant bandwidth efficiency. However, this must be evaluated against the strict acceptance drift gates.
- **Speculative Path 2 (Prefill/Decode Branching)**: Given the high serial sorting overhead in the router at small sequence lengths ($T \le 128$), bypassing sorting entirely during the decode stage remains critical. Prefill stages ($T \ge 256$), however, should enforce global GPU sorting to capitalize on memory coalescing in the MoE block.
