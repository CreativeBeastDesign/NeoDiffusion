# WP-6b: MoE Module Attribution & Hardware Scaling (M2 Ultra)

> ## ⚠️ CAVEAT — added 2026-07-14, read before quoting any number here
>
> **The raw latencies below are real and tightly repeatable. Every quantity *derived* from them
> — the forward budget, the "hardware speedups" in §3, and the causal analyses in §4B/§4C — is
> not usable.** The measurements are sound; the inferences drawn from them are not.
>
> **The disqualifying arithmetic** *(sourced)*: `llada2.1-mini` runs 19 MoE layers per forward,
> so this report's MoE block figure extrapolates to **19 × 2.33 ms = 44.3 ms — 161% of the
> measured 27.5 ms Studio forward**, before counting attention (20 layers), lm_head (3.50 ms),
> dense FFN (0.78 ms), or norms. The in-situ forward is `denoiseSeconds / forwardsEvaluated`
> over 315 clean rows of `scratch/llada_bench.jsonl` (q-cached, T=32.0 exactly on every row).
> A module attribution that exceeds 100% of the thing it attributes is measuring something else.
>
> **Root cause** *(inferred — the contradiction is sourced and airtight, but nobody has isolated
> the sync cost directly)*: `LLaDAMoEDispatchBench` times ops as `for _ in 0..<reps { eval(body()) }`.
> Each rep pays a full graph-eval/sync that production never pays — a real forward fuses its ops
> into one lazy graph. **The distortion is host-dependent**, which is why it went unnoticed: on the
> M1 the same harness reconciles (19 × 11.3 = 215 ms of the 340 ms forward = 63%, remainder ≈ 90 ms
> attention — m6-logbook Finding 4), because the fixed sync cost is negligible against the M1's slow
> compute. Against the Studio's ~5–10× faster compute it dominates. **M1-era conclusions from this
> harness are probably sound; Studio-era ones are not.** Note a simple fixed-overhead model does not
> reconcile it either — the router measures 1.02 ms while `gatherQuantizedMM` at T=32 measures
> 0.68 ms, so there is no single floor constant. Do not build on a specific overhead figure.
>
> **Consequence**: this report does **not** size Case B (dequant overhead). That question remains
> open and is being answered by in-situ causal ablation instead. See
> [`Plans/gather_qmm_handoff.md`](../gather_qmm_handoff.md) §5.3 and §5.5.
>
> **What survives**: the per-run latencies in §2 (as isolated-and-synced op cost, an ordinal
> quantity), and the §4A observation that the LM head is memory-bound — though §4A's numbers are
> microbench-derived too and will be superseded by the ablation run.

## 1. Executive Summary
- **Goal**: Characterize the performance of the core submodules of `llada2.1-mini` on the Mac Studio M2 Ultra GPU to identify performance bottlenecks and evaluate hardware scaling behavior relative to the M1 MacBook Pro baseline.
- **Methodology**: Execute `LLaDAMoEDispatchBench.testModuleAttribution` in a multi-run profiling loop under verified environmental conditions. Compare measured latencies with M1 baseline data to compute scaling speedups and analyze system bottlenecks.
- **Outcome**: ~~The Mac Studio M2 Ultra achieves a **6.20x speedup** across the major evaluated modules. The LM Head achieves the largest scaling win (**9.14x** speedup) due to M2 Ultra's 800 GB/s memory bandwidth, shifting the primary bottleneck away from memory transfers toward graph compiling/scheduling overhead.~~ **RETRACTED — see caveat above.** These are not hardware speedups (§3). The real M1→Studio forward speedup is **12.4×** (340 ms → 27.5 ms, sourced); this report's module ratios systematically *understate* the Studio because its numbers carry proportionally more harness overhead.

---

## 2. Experimental Results & Telemetry
All runs were completed on the Mac Studio M2 Ultra (M2 Ultra 192 GB unified memory, ~~76-core GPU~~ **60-core GPU** — corrected 2026-07-14 from `system_profiler` on the machine itself; 76-core is the other M2 Ultra bin. Bandwidth is 800 GB/s on both, so §4A's efficiency figures stand; §4B's "massive execution width (76 cores)" occupancy story was keyed to the wrong number as well as the wrong measurement) under verified environment constraints:
- Swap usage before/after: **0.0 MB / 0.0 MB** (Swap growth: **0.0 MB** $\le 256$ MB gate)
- Free memory pages at start: **~5,291,000 pages** ($\approx 80.7$ GB) — *gate citation corrected 2026-07-14*: the floor is no longer an absolute 1024 MB but **1/16 of physical RAM = 12 GB on the Studio** (m6-logbook rule 1a; the absolute floor was M1-calibrated and silently passed paged-out Studio rows). 80.7 GB clears either version — **no impact on this result**, citation only.
- OS thermal state: **nominal** (nominal/fair gate)
- Environment validity: **`envValid = true`**

> The environment here is genuinely clean — none of the caveat above is an environmental
> objection. The distortion is in the *harness*, not the machine.

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

## 3. Scaling Comparison: M1 vs. M2 Ultra — ⚠️ NOT HARDWARE SPEEDUPS (retracted 2026-07-14)

> **This table does not measure hardware.** It divides two numbers contaminated by the harness's
> per-rep `eval()` sync to *different degrees* — negligibly on the M1, dominantly on the Studio.
> The ratio therefore measures how much of each host's number is overhead, not how much faster the
> M2 Ultra is. **Do not quote "MoE Block 4.85×", "Router 1.67×", or "Combined 6.20×".**
>
> **The sourced figure**: real M1→Studio speedup on a whole forward = **340 ms / 27.5 ms = 12.4×**
> (m6-logbook Finding 4 vs `scratch/llada_bench.jsonl`). Every ratio below understates it, which is
> the signature of the bias: the Studio's small true compute is buried under a fixed overhead the
> M1's large true compute hides.
>
> Retained below strikethrough for the record (house rule: record negative results, don't delete).

We compare the mean latencies measured on the M2 Ultra against the frozen dev baseline recorded on the M1 MacBook Pro (16 GB unified memory, 8-core GPU). 

*Sourced for M1: `Plans/m6-logbook.md` line 59.*
*~~Inferred: Speedups calculated via division of sourced M1 and M2 Ultra latencies.~~ **RETRACTED** — the division is invalid; see above.*

| Submodule | M1 MBP Baseline (ms) | M2 Ultra Mac Studio (ms) | ~~Hardware Speedup~~ RETRACTED |
| :--- | :--- | :--- | :--- |
| **MoE Block** | 11.30 ms | 2.33 ms | ~~4.85x~~ |
| **Router Alone** | 1.70 ms | 1.02 ms | ~~1.67x~~ |
| **LM Head** (F16) | 32.00 ms | 3.50 ms | ~~9.14x~~ |
| **Dense FFN** | 2.30 ms | 0.78 ms | ~~2.95x~~ |
| **Combined Measured Modules** | **47.30 ms** | **7.63 ms** | ~~6.20x~~ |

*Real forward-level speedup, for reference:* **12.4×** (sourced).

---

## 4. Key Findings & Architectural Analysis

### A. Memory-Bandwidth Saturation of the LM Head `[Sourced / Inferred]`
- The LM Head (`Linear` size `[2048, 157184]` in float16) holds $157,184 \times 2,048 \times 2 \text{ bytes} \approx 643.8 \text{ MB}$ of weights.
- On the M1 MBP (theoretical memory bandwidth: 68 GB/s), loading this matrix requires a theoretical minimum of $9.47\text{ ms}$. The measured time of $32.0\text{ ms}$ represents **~30% bandwidth efficiency** (inferred).
- On the M2 Ultra Mac Studio (theoretical memory bandwidth: 800 GB/s), loading this matrix requires a theoretical minimum of $0.80\text{ ms}$. The measured time of $3.50\text{ ms}$ represents **~23% bandwidth efficiency** (inferred).
- **Conclusion**: The LM Head scales nearly linearly with memory bandwidth (**9.14x speedup** vs. **11.76x bandwidth scaling**). It remains highly memory-bound, making it the highest single-submodule cost on both platforms.

### B. MoE Weight Coalescing and Scheduling Overhead ~~`[Inferred]`~~ ⚠️ RETRACTED — measures the harness, not production
> **The label was right; the subject was wrong.** This inference is sound reasoning applied to a
> number that describes the *measuring apparatus*. The MoE block's modest apparent speedup is
> substantially the harness's own per-rep `eval()` sync, not production kernel-launch overhead or
> GPU under-occupancy. Production never issues these ops in isolation — it fuses 19 layers × 3
> projections into one lazy graph. **Whether real under-occupancy exists at T=32 is untested**; it
> is a live hypothesis, but this measurement cannot support it. The in-situ ablation
> (`gather_qmm_handoff.md` §5.5) is what would.

- ~~The MoE block achieves a **4.85x speedup** on the M2 Ultra. This is below the pure memory bandwidth scaling factor of $11.76\times$ (800 GB/s vs 68 GB/s).~~
- ~~**Analysis**: Because the active token count $T=32$ is small, the dynamic gathering of expert weight slices (`gatherQuantizedMM`) suffers from cache underutilization and launch latency overhead. The M2 Ultra GPU's massive execution width (76 cores) remains under-occupied by small-batch, dynamic gather operations, leaving performance dominated by scheduling overhead rather than raw memory throughput.~~

### C. Router Compute Constraints ~~`[Inferred]`~~ ⚠️ RETRACTED — measures the harness, not production
> Same defect. The router's apparent 1.67× is attributed here to "serial sorting operations" and
> "CPU-GPU synchronization boundaries" — but the CPU-GPU synchronization in question is **the
> benchmark's own**, injected by `eval()` once per rep. It is not a property of the production
> router. Note the internal tell: the router measures **1.02 ms** while a whole
> `gatherQuantizedMM` at T=32 measures **0.68 ms** — a bare matmul + sigmoid + top-k costing more
> than a 256-expert gathered quantized GEMM is not a plausible compute story, and is the clearest
> single sign that these numbers are overhead-dominated rather than compute-dominated.

- ~~The router (FP32 matmul + sigmoid + group top-k) is the most compute-bound module of the set, achieving only a **1.67x speedup**. This reflects the overhead of the serial sorting operations (top-k) on the GPU, which do not scale well with massive hardware parallelism at small batch sizes.~~

---

## 5. Architectural Recommendations & Speculative Paths

- ~~**Speculative Path 1 (LM Head Quantization)**: Quantizing the LM Head to 4-bit (reducing weight memory footprint from 643.8 MB to ~161 MB) is projected to reduce LM Head latency from $3.50\text{ ms}$ to **$\approx 0.90\text{ ms}$** on the M2 Ultra, assuming constant bandwidth efficiency. However, this must be evaluated against the strict acceptance drift gates.~~
  > ⚠️ **STRUCK 2026-07-14 — this re-opens a settled question, and the answer was NO.**
  > **M8 Finding M8-2 REJECTED lm_head 4-bit on quality** (`Plans/m8-logbook.md:72`, table row
  > `:120`): head-q4 vs F16-head with the same 4-bit body and the engine on both sides (zero
  > pipeline confound) gave **largest flipped margin 0.32 > noise p99 0.23** — it fails the
  > margin-calibrated gate. CLAUDE.md records the outcome: *"lm_head stays 16-bit — the §2.7 open
  > item is closed."*
  >
  > Two independent reasons this path is dead, either sufficient:
  > 1. **The quality rejection stands regardless of speed.** A latency projection cannot reopen a
  >    gate that was failed on drift. The last sentence above ("must be evaluated against the strict
  >    acceptance drift gates") describes an evaluation that *already happened and failed*.
  > 2. **The projection's input is contaminated.** The 3.50 ms → 0.90 ms extrapolation rests on the
  >    microbench number this caveat retracts, and assumes constant bandwidth efficiency.
  >
  > M8's own lesson applies here — *never port quantization verdicts between models* (the same axis
  > was accepted for Sumi and rejected for LLaDA2.1). By the same discipline, do not port a verdict
  > across *gates*: a speed argument does not overturn a quality rejection.
- **Speculative Path 2 (Prefill/Decode Branching)**: Given the high serial sorting overhead in the router at small sequence lengths ($T \le 128$), bypassing sorting entirely during the decode stage remains critical. Prefill stages ($T \ge 256$), however, should enforce global GPU sorting to capitalize on memory coalescing in the MoE block.
  > *Note (2026-07-14): right conclusion, wrong stated reason.* This bullet leans on §4C's retracted
  > router claim, but the conclusion survives on independent evidence: WP-6a's sorting sweep, and
  > the **measured** fact that production T is pinned at exactly 32.0 on every one of 315 clean
  > Studio rows (`gather_qmm_handoff.md` §2.7). Decode already bypasses sorting via the shape-aware
  > threshold, so this is a description of current behaviour, not a proposal. The prefill half is
  > **speculative** and currently unreachable: LLaDA2.x chunks prefill at `blockLength`, so T never
  > reaches 256 unless that hyperparameter is raised.
