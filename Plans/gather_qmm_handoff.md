# Hand-off: gather_qmm Kernel Investigation — Dequantization Overhead at Small Batch Sizes

**From**: Andre Bärlocher (via Perplexity research session)
**To**: Claude Opus / Claude Code
**Date**: 2026-07-14
**Target hardware**: Mac Studio M2 Ultra (**60-core GPU** — corrected 2026-07-14 from `system_profiler`
on the actual machine; this doc and the WP-6a/WP-6b drafts all said 76-core, which is the *other*
M2 Ultra bin. Memory bandwidth is 800 GB/s on both, so bandwidth-efficiency figures are unaffected;
any *compute*-peak or occupancy claim keyed to core count is overstated by ~27%.)
**Target models**: llada2.1-mini (H=2048, I=512, E=256, K=8), llada2.1-flash (H=4096, I=1024, E=256, K=8)
**Repo context**: LLaDA2SparseMoEBlock.callAsFunction, DiffusionEngine+Entry.swift

---

## 1. Origin and Scope of This Investigation

We set out to design a custom Metal `gather_qmm` kernel to accelerate sparse quantized MoE
projection (4-bit weights, affine group quantization) on Apple Silicon, motivated by three
theorized bottlenecks:

- **Case A**: SIMD/warp divergence from unsorted expert routing indices.
- **Case B**: Dequantization overhead (scale/bias application) in the inner accumulator loop.
- **Case C**: Apple toolchain friction (manual `.metallib` compilation/seeding).

Rather than building the kernel outright, we ran a structured sequence of benchmarks to test
whether a custom kernel was actually necessary, and if so, which bottleneck it should target.
**Conclusion: a from-scratch kernel targeting Case A (divergence/tiling) is NOT justified.
Case B (dequantization overhead) remains open and is the correct next target.**

> **Status as of 2026-07-14 — SIZED, and the recommendation has changed. Read §5.6 then §5.7.**
>
> **`gather_qmm` is 42.5% of the served forward** (11.75 ms of 27.63 ms), router a further 13.8%,
> MoE total 60.6%. Measured by in-situ causal ablation on the Studio across two independent runs
> (routed-MoE 56.2% then 56.3%), sanity gate PASS at exactly 100%, control reproducing the served
> baseline to within +0.5%.
>
> **But `gather_qmm` is memory-bound (40–58% of the 800 GB/s peak), so a *perfect* kernel caps out
> at 17.9% end-to-end — and dequant (pure ALU) cannot help directly under a memory bound.** The
> only surviving Case B mechanism is indirect: register pressure capping occupancy and hence
> memory-level parallelism. §3 step 1 must therefore measure **bandwidth and occupancy**, not ALU.
>
> **The bigger lever is `blockLength`** (§5.7): expert bytes *per token* fall ~32% at B=64 and ~61%
> at B=128, against a MoE that is 60.6% of the forward — dwarfing the kernel ceiling, and costing a
> config sweep rather than an mlx-swift fork. **Sweep that first.**
>
> Case A is closed and measured (§2.7). Case C (§4) is open and decides *how* any kernel ships,
> not *whether*. Two attempts to size Case B from the dispatch microbench failed identically —
> extrapolated to a forward, their numbers exceed the whole forward (§5.2: 38.9 ms; §5.3: 44.3 ms;
> the forward is 27.53 ms). **Do not quote Studio magnitudes from `LLaDAMoEDispatchBench`
> (WP-6a/WP-6b) — they are ordinal at best (§5.3).** The M1-era figures from that harness, by
> contrast, are corroborated: it put MoE at ~63% of the M1 forward, and the in-situ Studio
> measurement says 60.6%.
>
> **Before profiling, read §5.6's last section**: 56.2% bounds *router + expert GEMMs together*.
> Case B is a fraction of the GEMM part alone, and one extra ablation arm splits them cheaply.

---

## 2. Key Findings (in order of discovery)

### 2.1 Stock MLX `gatherQuantizedMM` already exists and is well-optimized
MLX ships a production `gather_qmm` primitive (added v0.25.0, ~2x speedup for batched MoE
prompt processing) and a `segmented_mm` op purpose-built for MoE grouped/segmented matmul.
Do not reinvent these — evaluate against them first for any future kernel work.

### 2.2 Initial micro-benchmark (Flash shapes, T=32)
| Kernel | Time | Eff. Bandwidth |
|---|---|---|
| `gatherQuantizedMM` (stock, sparse gather) | 1.4 ms | 13.5 GB/s (~1.7% peak) |
| Dequantize + `gatherMM` | 5.8 ms | — |
| Dense `quantizedMM` (all experts) | 5.6 ms | 107.9 GB/s (~13.5% peak) |

Dense achieved 8x higher effective bandwidth despite reading 32x more data — initial
hypothesis: SIMD divergence from unsorted routing indices (Case A).

### 2.3 Occupancy sweep ruled out low-occupancy as sole cause
Sweeping T ∈ {32, 128, 512, 2048} on Flash shapes showed:
- At T=32: gatherQuantizedMM achieves 290.5 GB/s (36.3% of peak) — NOT occupancy-bound.
- As T grows past 512 (all 256 experts active), the kernel transitions from
  **memory-bound to compute-bound** scaling (FLOPs scale linearly with T while weight
  data volume plateaus).

### 2.4 Causal A/B test confirmed divergence is real, but regime-dependent
At T=512 (fixed data volume, varying only routing order):
- Unsorted: 12.97 ms (44.4 GB/s)
- Sorted (pre-sorted indices): 6.99 ms (82.4 GB/s)
- **1.85x causal speedup from sorting alone.**

### 2.5 Exact crossover point is shape-dependent — not a single constant
Sweeping T finely with real GPU-side sort overhead (~0.45–0.52 ms, flat across T) included:

| Shape | Crossover T | Speedup at crossover |
|---|---|---|
| Flash (H=4096) | 128 | 2.00x |
| Mini (H=2048) | 192 | 1.865x (at T=256; T=128 is a 0.972x regression) |

Dispatcher was implemented with shape-aware threshold:
`let crossover = flat.dim(1) >= 4096 ? 128 : 192`

*Corroborated on the Studio (2026-07-14), record `Plans/wiki-drafts/wp-6a-moe-adaptive-dispatch.md`*:
the M2 Ultra sweep reproduces this shape — Mini T=128 is a 0.972× regression, T=256 a 1.865×
net win, so the Mini crossover falls in (128, 256) and 192 is a fair interpolation. Note the
sweep never measured T=192 directly; the constant is interpolated, not observed. Harmless
today (§2.7: production T is pinned at 32), but measure it before trusting the constant if
`blockLength` is ever raised into that band.

### 2.6 `segmentedMM` beats our custom sorted path — but doesn't support quantization
`segmentedMM` is 15–20% faster than sorted `gatherMM` on **unquantized** weights, but has no
quantized-weight support, so it cannot currently replace `gatherQuantizedMM` for
llada2.1-mini/flash's 4-bit layers. This is a gap in the MLX op, not evidence for a
from-scratch kernel — worth tracking upstream.

### 2.7 Production T distribution invalidates the case for a sorted-path kernel today
LLaDA2.0 is architecturally a **block-diffusion** model: prompt prefill is chunked in
DiffusionEngine+Entry.swift at `blockLength` (B=32 or B=64), meaning the model's forward-pass
T is architecturally bounded by B or 2B — well below both crossover thresholds (128/192).

**Result: the sorted-dispatch path we built is currently dormant in production.** It remains
correct and gated in the dispatcher, but is not exercised under current config. It should be
preserved and documented as contingent on `blockLength` — if that hyperparameter is ever
raised (block diffusion literature shows blockLength is a real quality/throughput tuning
knob, not a fixed constant), the sorted path activates automatically.

**Confirmed empirically on the Studio, 2026-07-14** (upgrade: this was *inferred from
architecture* when written; it is now *measured*). Across all 315 clean steady rows of the
WP-3/WP-4 Studio backfill (`scratch/llada_bench.jsonl`, served default `q-cached`),
`tokensProcessedInForwards / forwardsEvaluated` = **exactly 32.0 on every row, every suite**
(chat, reasoning, code). Production T is not merely *bounded* by B — it is pinned at B, with
no spread whatsoever. The sorted path is dormant, and §2.5's crossover (128 Flash / 192 Mini)
sits 4–6× above the regime the engine actually runs in. Nothing about the sorted-dispatch
verdict is in doubt.

### 2.8 Net verdict on custom kernel scope
Every hypothesized bottleneck (Case A, divergence) was either already solved by an existing
library primitive (`gather_qmm`, `segmented_mm`) or found to be inapplicable at production
batch sizes. **No evidence supports writing a from-scratch tiling/sorting kernel.**

Note the scope of this verdict: it closes **Case A only**. Case B is open (§3) and Case C is
open (§4). An earlier draft of this doc stated that Case C "was correctly avoided by not
building this" — that is not a resolution, it is a deferral that holds exactly as long as no
kernel is built. Since §3 now scopes kernel work, Case C is live again. Corrected 2026-07-14
(André).

---

## 3. Open Problem: Dequantization Overhead at Small Batch Sizes (Case B)

This is the one bottleneck we did **not** rule out, and it directly matches production
conditions (T=32–64, unsorted regime, which is the actual serving path).

### What we know
- Affine 4-bit quantization requires applying per-group (group_size=64) scale and bias
  during dequantization, inside the accumulator's inner loop.
- This adds register pressure and floating-point ops per accumulation step, and is a
  documented, still-open problem in the broader quantized-inference literature — recent
  work (e.g., fused dequant-multiply kernels) specifically targets converting theoretical
  memory savings from quantization into realized speedups, because dequant overhead can
  erode gains at small batch/low-arithmetic-intensity regimes.
- GPU kernel dispatch overhead on Apple/Metal is comparatively modest (tens of microseconds
  per dispatch based on comparable GPU API characterizations), so the bottleneck is more
  likely in-kernel (register pressure, ALU stalls from interleaved dequant+FMA) than
  dispatch-level.

### What we have NOT yet measured
- Whether register pressure or ALU stalls are actually the dominant cost at T=32/64 for
  `gatherQuantizedMM` (no Instruments/Metal counter profiling has been done on this
  specifically — all prior profiling was bandwidth-focused, not occupancy/ALU-focused).
- Whether pre-computing dequantized scale/bias into threadgroup shared memory (rather than
  recomputing per-thread in the inner loop) meaningfully reduces this cost.
- Whether `simdgroup_matrix` (Metal's MMA co-processor path) is being used effectively in the
  stock kernel's small-T path, or whether it falls back to scalar accumulation.

### The question for you (Claude Opus / Claude Code)
Given the production regime is unsorted, small-T (32–64), quantized 4-bit MoE dispatch:

1. Profile `gatherQuantizedMM` at T=32/64 with Metal System Trace / Instruments, specifically
   isolating ALU occupancy, register spill, and memory stall percentages — distinguish
   whether the residual gap (vs. theoretical peak) is dequant-bound or something else
   entirely (e.g., dispatch-level overhead at this small scale).
2. If dequant-bound: prototype a targeted change to the inner loop (shared-memory scale/bias
   caching, reduced-precision intermediate accumulation, or restructuring the loop to batch
   dequant operations ahead of the FMA chain) rather than a full kernel rewrite — scope this
   as narrowly as the delivery mechanism allows. **Read §4 before costing this work: the
   phrase "patch the existing kernel" describes the most expensive available route, not the
   cheapest, and the scoping in this section was written before that was understood.**
3. Evaluate whether `simdgroup_matrix` MMA paths are actually engaged for these small-T
   quantized shapes, or whether a scalar fallback is silently active.

**Constraint from prior investigation**: avoid re-deriving tiling/sorting logic — that
question is closed. Scope any new kernel work narrowly to the dequantization/register-
pressure problem only.

---

## 4. Open Problem: Delivery Mechanism for a Kernel Change (Case C)

**Status: UNRESOLVED. Blocks §3 step 2, and must be settled before that work is costed.**

Case C was listed in §1 as a hypothesized bottleneck and then never tested — the investigation
built no kernel, so the toolchain was never exercised. It is not closed.

### 4.1 What the repo already proves — Case C is solved for *additive* kernels
*Sourced (code read 2026-07-14).* WP-3b's FlashBlock does **not** go through SwiftPM `.metal`
compilation. `FlashBlockRunner.swift:445/453` builds its kernels with `MLXFast.metalKernel(...)`,
passing Metal source as inline Swift strings compiled by MLX at runtime. There is no
compile/package/reload cycle: edit the string, rebuild Swift, run.

This narrows the general SwiftPM/Metal limitation described in the hand-off note. That
limitation is real and accurately stated, but for *new, standalone* kernels this project has
already routed around it — and the route is proven in production code, not theory.

*Resolved 2026-07-14.* The repo root previously held a loose copy of the original standalone
FlashBlock dev package (`FlashBlock.metal`, `FlashBlockOpt.metal`, `FlashBlockOptRunner.swift`,
`Flashblock README.md`, `FlashBlock.metallib`, `FlashBlockMetal.zip`). None of it was in the
SwiftPM build — no target has `path: "."` — and `FlashBlockOptRunner.swift` loaded kernels the
old way, via `device.makeLibrary(URL:)`. `FlashBlock.metal` was a verbatim duplicate of the
inline strings in the shipped runner (body confirmed line-by-line, incl. the `acc[128]` fix),
and `Tools/seed-metallib.sh` compiled it to a metallib that nothing ever opened. All loose
copies removed; `FlashBlockMetalOpt.zip` retained as the sole archive (verified strict superset
of the other zip; carries the un-integrated Opt kernels + test harness + reference scripts —
see §4.4). Build outputs now gitignored.

### 4.2 Why that escape hatch does not cover the §3 patch
*Inferred, high confidence — from the mechanism, not from an attempt.* `gather_qmm` is not our
kernel. It lives in mlx-swift's Cmlx sources and ships precompiled inside
`mlx-swift_Cmlx.bundle`. `MLXFast.metalKernel` only **registers new** kernels; it offers no
hook to override or patch a kernel already inside that bundle. So "patch the inner loop of
`gatherQuantizedMM`" has no cheap delivery path. The two real routes:

- **(a) Fork mlx-swift, patch the kernel, rebuild the Cmlx metallib.** This collides head-on
  with the toolchain quirk this repo already lives with: `Tools/seed-metallib.sh` exists
  *because* `swift build` on this toolchain does not compile mlx-swift's `.metal` sources at
  all. It works by copying a prebuilt bundle out of `~/Library/Developer/Xcode/DerivedData/`
  (line 9), and fails outright if no Xcode build exists. A forked kernel would therefore need
  an **Xcode DerivedData rebuild of mlx-swift on every edit**, then a reseed, for each
  iteration — plus a permanent fork to carry forward across mlx-swift upgrades. Case C in its
  full, original form.
- **(b) Reimplement the small-T quantized path as a new `MLXFast.metalKernel`** and dispatch
  to it from `LLaDA2SparseMoEBlock` alongside the existing shape-aware threshold (§2.5).
  Delivery is then free — the FlashBlock route, already proven here.

### 4.3 The unresolved tension
§3 step 2 says "scope this as a patch, not a new kernel, per lessons learned above." Given
4.2, that instruction inverts the actual cost: the **patch** is the expensive route (a)
and the **new kernel** is the cheap route (b). The §2.8 lesson it appeals to argued against a
from-scratch *tiling/sorting* kernel — a statement about which *bottleneck* to target, not
about delivery mechanism. Applying it to (b) overextends it: a narrow dequant-path kernel
reusing stock tiling is not the thing §2.8 rejected.

**Decision needed before §3 step 2 is costed** (André):
1. Does profiling (§3 step 1) actually indict dequant? If not, Case C stays moot — **do this
   first; it is cheap and gates everything below.**
2. If yes: route (b) by default, on cost grounds. Route (a) only if profiling shows the win
   requires changing stock tiling/MMA structure that (b) would have to re-derive — which
   would reopen §2.8's closed question and should be escalated, not decided in-flight.
3. Either route needs an M6-style A/B against stock `gatherQuantizedMM` at real T=32/64
   shapes under the `.agents/AGENTS.md` acceptance gates. Route (b) additionally needs a
   correctness gate vs. stock output, since it is a reimplementation.

**Untested assumption in 4.2** *(speculative)*: that MLX exposes no override hook for a
built-in primitive's kernel. Nobody has checked the Cmlx registration path for one. If such a
hook exists, route (a)'s cost collapses and this section needs rewriting. ~30 min to falsify;
do it before committing to (b).

### 4.4 Salvage note: un-integrated FlashBlock "Opt" kernels
*(Case C content continues; the Case B sizing update is §5.)*
*Sourced (archive inspected 2026-07-14).* `FlashBlockMetalOpt.zip` contains a second, larger
kernel variant (`FlashBlockOpt.metal`, 40 KB vs 21 KB) that was **never ported** into the
shipped `FlashBlockRunner`. Its kernels are named differently — `flashblock_external_pass_opt`
and, notably, `flashblock_compose_and_proj`, which fuses the **output projection** into the
attention epilogue. That is Phase-3 §4 item 2 ("attention epilogue fusion") already written.
It is un-benched and un-integrated, not dead — kept deliberately. Anyone picking up the
epilogue-fusion track should read it before writing a new kernel. Porting it means moving the
source into inline strings per §4.1, *not* reviving `FlashBlockOptRunner.swift`'s
`makeLibrary(URL:)` path.

---

## 5. Sizing Case B: what the Studio backfill settles, and what still blocks it

**Added 2026-07-14, after the WP-3/WP-4 Studio backfill (`scratch/llada_bench.jsonl`, 315 clean
steady rows, host Mac14,14 / M2 Ultra, all `envValid` under the amended m6-logbook rule 1a).**
Short answer to "do we have enough data to act on §3 yet?": **no — but we now know two things
that change how §3 must be approached, and one of them invalidates the obvious way to size it.**

### 5.1 New anchor: the production forward is ~27.5 ms at T=32
*Sourced* — `q-cached` (served default), `denoiseSeconds / forwardsEvaluated`:

| suite | forward | T per forward |
|---|---|---|
| chat | 28.66 ms | 32.0 |
| reasoning | 27.50 ms | 32.0 |
| code | 26.72 ms | 32.0 |

This is the denominator any Case B claim must be expressed against. A dequant win is worth
`(MoE dispatch share) × (fraction of that recovered)` of ~27.5 ms, and nothing more.

### 5.2 Blocker: the dispatch microbench numbers do **not** compose into a forward budget
*Sourced (bench source read 2026-07-14).* The natural move — take WP-6a's Mini T=32 figure
(unsorted `gatherQuantizedMM` = 0.6827 ms) and multiply up to a per-forward MoE cost — is
**invalid, and provably so**:

- `LLaDAMoEDispatchBench` H2 times **one routed projection** (`Tests/DiffusionCoreTests/LLaDAMoEDispatchBench.swift:48`).
- A forward runs 19 MoE layers (layer 0 is dense), each `SwitchGLU` issuing three
  `gatherQuantizedMM` calls (gate, up, down) ⇒ **57 calls per forward**.
- 57 × 0.6827 ms = **38.9 ms — larger than the entire measured 27.5 ms forward.**

The contradiction is the finding. The microbench wraps every rep in its own `eval(body())`, so
each timed call pays a full graph-eval/sync that production never pays: in the real forward,
MLX fuses all 57 dispatches into one lazy graph evaluated once. **The microbench measures
isolated-and-synced op cost, not in-situ op cost, and overstates the latter by enough to
exceed the whole forward.** Consequences:

1. **Case B cannot be sized from any existing number.** The prize is unknown, not small — this
   is not a negative result, it is missing data.
2. This caveat applies to *every* ranking derived from that harness, including WP-6a's
   dispatch verdicts and the CLAUDE.md claim that `gatherQuantizedMM` is the fastest MoE
   dispatch. Those are comparisons *between* variants measured the same way, so the **ranking**
   plausibly survives (the sync overhead is common-mode); the **magnitudes** do not transfer,
   and no absolute share-of-forward may be quoted from them. *(Ranking-survives is **inferred**,
   not measured — common-mode cancellation assumes the sync cost is variant-independent, which
   nobody has checked.)*

### 5.3 The cheap unblock was tried and **FAILED** — H4 cannot size Case B either
*Attempted 2026-07-14 (André, Studio run; record `Plans/wiki-drafts/wp-6b-m2-ultra-module-attribution.md`).*
The plan was: run `LLaDAMoEDispatchBench.testModuleAttribution` (H4, one whole quantized MoE
block, warm, real shapes) on the Studio, and read `19 × block` against §5.1's 27.5 ms forward as
an upper bound on MoE share. It was run — 6 repetitions, tight spread (2.20–2.40 ms), **mean
2.33 ms**. The upper bound it produces is **vacuous**:

> **19 × 2.33 ms = 44.3 ms = 161% of the entire 27.5 ms forward** — MoE alone, before counting
> attention (20 layers), lm_head (3.50 ms), dense FFN (0.78 ms), or norms.

An upper bound above 100% constrains nothing. **Case B remains unsized.** §5.2's caveat did not
merely apply to H4 — it swallowed it.

**The distortion is host-dependent, which is why it went unnoticed.** Run the same harness on the
M1 and the arithmetic reconciles: 19 × 11.3 ms = 215 ms of the 340 ms steady forward (63%), plus
lm_head 32 ms, dense FFN 2.3 ms, remainder ≈ 90 ms for attention + norms — it sums to ~100%
(m6-logbook Finding 4). The harness's fixed per-rep `eval()`/sync cost is negligible against the
M1's slow compute and dominant against the Studio's ~5–10× faster compute. **The M1-era
conclusions from this harness are probably sound; the Studio-era ones are not.** *(Mechanism is
**inferred** — the arithmetic contradiction is **sourced** and airtight, but no one has isolated
the per-rep sync cost directly. A naive fixed-overhead model does not fully reconcile it either:
the router measures 1.02 ms while `gatherQuantizedMM` at T=32 measures 0.68 ms, so the floor is
not a single constant. Do not build on a specific overhead figure.)*

**Consequences for the WP-6b report, beyond Case B** — these are corrections, not quibbles:
1. **Its §3 "speedup vs M1" table is not measuring hardware.** It divides two
   differently-contaminated numbers. The real M1→Studio forward speedup is 340/27.5 = **12.4×**;
   the report's "combined measured modules" says 6.20×. Module ratios systematically *understate*
   the Studio because its numbers carry proportionally more overhead. "MoE Block 4.85×" and
   "Router 1.67×" are not hardware speedups.
2. **Its §4.2 and §4.3 inferences describe the measuring apparatus, not production.** §4.2 blames
   the modest MoE speedup on production kernel-launch overhead and under-occupancy; §4.3 blames
   the router's "serial sorting operations and CPU-GPU synchronization boundaries". Both are at
   least partly the *harness's* per-rep sync. Neither is safe as written.
3. Minor: its §1 cites the superseded validity rule ("Start Free Memory ≥ 1024 MB"). The floor is
   now 1/16 of RAM = 12 GB on the Studio (m6-logbook rule 1a). Measured 80.7 GB passes either
   way — no impact on the result, but the citation is stale.
4. Minor: its §5B puts Mini break-even at T=128 (1.02×), where WP-6a's sweep put T=128 at 0.972×.
   Two Studio runs disagreeing across break-even is noise, and irrelevant at production T=32 —
   but it further undercuts the microbench's apparent precision.

### 5.4 What has NOT changed
- **§3 step 1 (the Instruments/Metal-counter profile) is still ungated and still first.** No
  ALU/register/stall profiling has been done. Nothing in the backfill touches this.
- **§4 (Case C, delivery mechanism) is unchanged and still decides *how*, not *whether*.** Its
  4.2 open item (does MLX expose an override hook for a built-in primitive's kernel?) remains
  ~30 min to falsify and unattempted.
- **§2.8's Case A verdict is strengthened, not weakened** — see §2.7's measured confirmation.

### 5.5 Recommended order (revised 2026-07-14 — the microbench route is exhausted)
Rung 1 (H4 on the Studio) was tried and failed (§5.3). **No microbench-derived number can size
Case B on this host**, because the harness's per-op timing is inflated past the point of
arithmetic possibility. Only in-situ measurement remains:

1. **Size MoE's share in situ.** Two candidates, either sound:
   - *(a) Causal ablation, cheapest — reuses the existing bench.* **BUILT 2026-07-14, awaiting a
     Studio run** (see §5.6). Time a real forward with the MoE block swapped for a cheap
     passthrough that keeps the residual stream alive; the forward delta is MoE's true in-situ
     share. Output correctness is irrelevant — this is a timing experiment. Immune to §5.2/§5.3's
     artifact because both arms time a **whole forward**, so any fixed harness cost is
     common-mode and cancels.
   - *(b) Metal System Trace / Instruments on a real forward* → per-kernel GPU attribution
     directly. This is §3 step 1 and was always the principled first move.
2. Only if MoE's real share is material: the §3 step 1 ALU/register/stall profile, in-situ.
3. Only if profiling indicts dequant: §4.3's route decision — (b) new `MLXFast.metalKernel` by
   default on cost grounds, not (a) "patch the inner loop".

**Also worth fixing regardless**: the harness itself. Timing `for _ in 0..<reps { eval(body()) }`
measures isolated-and-synced op cost, which is not a quantity production has. Amortising the sync
(building one graph over many reps, with inputs varied enough to defeat CSE) would make the
harness's numbers mean something on fast hosts. Until then, treat every Studio number it emits —
WP-6a's and WP-6b's alike — as ordinal at best.

**Standing methodological note for whoever picks this up.** The Studio backfill's headline
lesson generalizes to kernel work: every arm ran in its own process, making all its A/Bs
cross-process (±6–10% thermal drift, m6-logbook Finding 7), which is why several sub-10%
"wins" evaporated between two clean runs. Any Case B result will live in exactly that
sub-10% band. Bench it **arms-in-one-process** (`--arms` takes a list) or it will not be
resolvable from drift, and do not let the microbench's flattering ±0.3% within-process
repeatability be mistaken for the precision of a cross-process comparison.

### 5.6 In-situ attribution: RUN. **Case B is ALIVE — S = 56.2%** (2026-07-14)

Implements §5.5 rung 1(a). Run on the Studio (`Mac14,14`, 60-core M2 Ultra), release build,
144 rows, **all `envValid`, one process** (`31453`), thermal nominal throughout, 93 GB free,
T = 32.0 on every row. Record: `scratch/attribution.jsonl`.

#### The measured forward budget (sourced)

| component | ms | share of the 27.53 ms forward |
|---|---|---|
| **routed MoE** (router + 57× `gatherQuantizedMM`) | **15.48** | **56.2%** |
| shared expert | 1.21 | 4.4% |
| *MoE total* | *16.69* | *60.6%* |
| attention (incl. KV-cache growth) | 3.79 | 13.8% |
| remainder (lm_head + norms + loop) — *by subtraction, inferred* | 7.06 | 25.6% |
| **sum** | **27.53** | **100.0%** |

**Sanity gate: PASS** — every term positive, sum exactly 100%. This is the gate WP-6b's
attribution failed at 161%, and it is why this budget is usable and that one was not.

**Control: PASS, decisively.** `attr-full` = **27.53 ms** vs the served `q-cached` baseline's
**27.50 ms** — **+0.10%**, measured in a *different process on a different day*. The refactor did
not perturb the served path, and the whole measurement chain reproduces an independent number.

**Independent corroboration**: the M1 microbench put MoE at ~63% of its forward (m6-logbook
Finding 4); this in-situ Studio measurement says **60.6%**. Two different hosts, two different
methods, ~3pp apart. Note this *rehabilitates* the M1-era figure specifically (§5.3 predicted the
M1 numbers were probably sound while the Studio ones were not — that prediction now has evidence).

**Robustness** (all three checks pass):
- *Arm-major ordering* (the §5.6 known limitation): per-arm run0→run2 drift is +0.2%/+0.1%/−0.4%/−0.1%
  — no session ramp. And the control pins `attr-full`, which ran **first**, to an independent
  cross-process value; had ordering inflated it, it would read high. It does not.
- *Content sensitivity* (AGENTS.md): paired per-prompt S over 12 prompts — median **56.4%**,
  range [52.4, 59.9], **every single prompt clears the 40% gate**.
- *Memory-locality bias* (pre-registered, deliberately uncorrected): makes S an **over**-estimate.
  At 56.2% against a 40% gate it would have to be enormous to flip the verdict.

#### Verdict against the pre-registered rule

`S = 56.2% > 40%` ⇒ **PROFILE JUSTIFIED.** Proceed to §3 step 1 (Instruments/Metal-counter
profile), **in situ, not on the microbench**. Case B is not merely alive: routed MoE is the single
largest item in the forward, larger than everything else combined.

**Prediction failed, recorded per house rule.** Before the run, Claude expected S < 15% ("a cheap,
decisive negative — the best outcome available"). It was wrong by ~4×, and in the direction that
costs more work. The lesson is the session's own thesis turned on itself: *the microbench was not
merely imprecise on the Studio, it was actively misleading about where the time goes*, and no
amount of reasoning about it substituted for one in-situ measurement.

### 5.7 The split: `gather_qmm` = 42.5% of the forward (2026-07-14, run 2)

`.moeExpertGEMMs` added (keeps the router, skips only `experts(...)`) and re-run with 5 arms,
180 rows, all `envValid`, one process, thermal nominal. Record: `scratch/attribution2.jsonl`.

**Replication first**: routed-MoE `S` = **56.3%**, against run 1's **56.2%**. Independent run,
0.1pp apart. Control `attr-full` = 27.63 ms vs the served baseline's 27.50 ms (+0.49%).

| component | ms | share of 27.63 ms forward |
|---|---|---|
| **`gather_qmm` expert GEMMs (57×)** | **11.75** | **42.5%** |
| router (incl. combine arithmetic) | 3.81 | 13.8% |
| shared expert | 1.17 | 4.2% |
| attention (incl. KV growth) | 3.93 | 14.2% |
| remainder (lm_head + norms + loop) — *inferred by subtraction* | 6.97 | 25.2% |
| **sum** | **27.63** | **100.0%** |

Sanity gate: **PASS** (all positive, exactly 100%). The DCE hazard specific to this arm — MLX
eliding the router once its consumer is gone — is excluded behaviourally, not by argument:
`ModuleAblationTests.testExpertGEMMsAblationStillRunsTheRouter`. Timing alone could never have
caught it (an elided router and a free router look identical).

#### What this budget explains, retroactively

- **FlashBlock's "algorithmic accept / wall-clock reject" (WP-3b) is now explained.** It optimises
  attention — **14.2%** of the forward. Deleting attention *entirely* buys 14.2%. There was never
  room for the win it was chasing.
- **The router is ~as expensive as all of attention** (13.8% vs 14.2%) — an FP32 matmul plus two
  `argSort`s per layer. Nobody has looked at it, because the microbench's inflated numbers made
  every module look overhead-dominated. Possible target; unexamined.
- **The microbench overstated the router ~5×**: 1.02 ms/call × 19 = 19.4 ms, which exceeds the
  entire 15.56 ms routed-MoE budget. Another instance of §5.2/§5.3.

#### Roofline: `gather_qmm` is bandwidth-bound, and that undercuts Case B's *mechanism*

*Inferred — arithmetic from `config.json`, not a profile.* Per forward the experts cost
~1.77 MB each (3 × [2048×512] at 4-bit + fp16 scales/biases), ~162 distinct experts touched per
layer (32 tokens × 8 experts over 256), × 19 layers ≈ **5.45 GB read per forward**. At 800 GB/s
that is a **6.81 ms floor** against the measured 11.75 ms ⇒ **~464 GB/s achieved, ~58% of peak**.

**Therefore: a perfect 100%-bandwidth kernel saves 4.94 ms = 17.9% end-to-end. That is the hard
ceiling on *all* `gather_qmm` kernel work, dequant included.** Case B is a fraction of that.

**Critical caveat, and it cuts against me**: the distinct-expert count assumes uniform routing. The
real router is group-limited (top-4 of 8 groups ⇒ ~128 reachable per token), which *concentrates*
routing ⇒ **fewer** distinct experts ⇒ fewer bytes ⇒ **lower** achieved bandwidth (~317 GB/s, ~40%
of peak under a concentrated model). So the uncertainty makes `gather_qmm` look **less**
memory-bound and leaves **more** headroom — it does not license "memory-bound, so Case B is dead".

The honest state: `gather_qmm` runs at **40–58% of peak bandwidth**, so it is memory-bound *with
real headroom*. Dequant is pure ALU (weights are read 4-bit and expanded after), so it cannot
matter *directly* under a memory bound — but it can matter **indirectly**, if register pressure
from the dequant chain caps occupancy and hence memory-level parallelism. **That is the only
surviving Case B mechanism, and it is exactly what §3 step 1 must test**: measure achieved
bandwidth and occupancy on the `gather_qmm` kernel, not ALU time.

#### The lever this measurement actually points at: `blockLength`

A memory-bound MoE is beaten by **reading less**, not by a faster kernel. Experts are fetched
*per forward*; more tokens per forward amortises them over more tokens:

| blockLength | distinct experts/layer | experts read **per token** | vs T=32 |
|---|---|---|---|
| 32 (current) | 162 | 5.06 | 100% |
| 64 | 221 | 3.46 | **68%** |
| 128 | 251 | 1.96 | **39%** |
| 256 | 256 | 1.00 | 20% |

Expert bytes per token fall **~32% at B=64** and **~61% at B=128**, against a MoE that is 60.6% of
the forward. **That dwarfs the 17.9% ceiling on all kernel work — and costs a config change rather
than an mlx-swift fork (§4's Case C).** It is not free: `blockLength` is a real quality/throughput
knob in block diffusion, and at B≥128/192 §2.7's dormant sorted-dispatch path auto-activates
(§2.5's crossover — note T=192 was never measured directly). It needs its own sweep with a quality
gate, exactly like WP-1b's τ_add.

**Recommended order (revised by this data):**
1. **Sweep `blockLength` ∈ {32, 64, 128}** with the §0 quality floor. Biggest lever, no Case C, and
   the ablation arms now make its MoE effect directly measurable. *This is the new first move.*
2. Only then §3 step 1's profile, scoped to **bandwidth/occupancy on `gather_qmm`** — and only to
   test the register-pressure-caps-MLP mechanism, since the direct ALU story is dead.
3. Kernel work (§4's route decision) only if (2) indicts occupancy **and** (1) has been banked.

#### Superseded: S bounds (router + experts), not `gather_qmm` alone

`.moeRoutedExperts` skips the router **and** the expert GEMMs, so **56.2% is a ceiling on both
together**, not on `gatherQuantizedMM` by itself — and Case B (dequant overhead) is a fraction of
the GEMM part only. The microbench cannot split them (its router figure, 1.02 ms, × 19 layers =
19.4 ms, already exceeds the whole 15.48 ms routed-MoE budget — another instance of §5.2/§5.3).

**Cheapest next refinement, before Instruments**: add a `.moeExpertsOnly` ablation that keeps the
router and skips only the expert GEMMs. `no-routed − no-expertsOnly` then splits the 15.48 ms into
router vs `gather_qmm`. One enum case, one branch, one re-run — and it sharpens the profiling
target from "56.2% of the forward" to the actual op under investigation.

---

#### Original pre-registration (retained — the rule was fixed before any data was seen)

**Pre-registered decision rule.** Let `S = (attr-full − attr-no-routed) / attr-full`, the
routed-MoE share of the forward:

| `S` | verdict |
|---|---|
| **< 15%** | **Case B is DEAD.** Close §3. Even a heroic 30% dequant recovery of *part* of that op yields <5% end-to-end — under the serving bench's resolvable floor. |
| **15–40%** | Marginal. No kernel work until §3 step 1's profile *specifically indicts dequant* (vs. gather / occupancy / router). |
| **> 40%** | Profile justified; proceed to §3 step 1, in-situ. |

**How to run** (Studio; all four arms in **one process** — that is the point):
```
diffusion-bench llada --arms attr-full,attr-no-routed,attr-no-moe,attr-no-attn \
  --runs 3 --gen-length 128 --no-early-stop --json scratch/attribution.jsonl
```

**Read `ms/forward` = `denoiseSeconds / forwardsEvaluated`. Never TPS.** Ablated arms emit
garbage by design, so their trajectories diverge (different steps/block, different token counts) —
but every forward is the same graph at T=32 regardless, which is what makes a garbage-output arm
a valid timing probe. A TPS comparison across these arms is meaningless.

| arm | delta yields |
|---|---|
| `attr-full` | **control** — must reproduce `q-cached` ms/forward (≈27.5 ms). If it does not, the harness changed the served path: stop, do not read any delta. |
| `attr-no-routed` | `full − no-routed` = router + 57× `gatherQuantizedMM` = **Case B ceiling** |
| `attr-no-moe` | `no-routed − no-moe` = shared-expert share |
| `attr-no-attn` | `full − no-attn` = attention share (**incl. KV-cache growth** — the cache never fills in this arm; not "attention math only") |

Remainder (`full − attention − MoE`) = lm_head + norms + loop overhead, **by subtraction —
inferred, not measured** (same honesty as m6-logbook Finding 4). **Sanity gate: every term must be
positive and the sum must not exceed 100%. If it does, the ablation is wrong — report it, do not
massage it.** That is precisely the failure mode that discredited WP-6b.

**Two non-obvious things found while building it**, both worth knowing before anyone touches this:

1. **The served default does *not* use the JOT/capacity forward overload.** `DiffusionEngine+Entry`
   branches: JOT/FlashBlock configs take the `jotCaches:` overload (`:152`), but the served default
   (elastic/JOT/FlashBlock all off) falls through to the plain **M5 cached** overload (`:166`).
   Wiring the ablation only into the former — as this doc's §5.5 originally implied — yields a
   switch that never fires on the default path, producing a **fake null** indistinguishable from
   "the module is free". `Tests/DiffusionGenerationTests/ModuleAblationTests.swift:testAblationActuallyBitesOnDefaultPath`
   exists specifically to catch that class of error, and the JSONL now carries an engine-level
   `moduleAblation` **effective echo** so it is checkable from the data rather than by inspection.
2. **Per-arm params now exist, and this is the real fix for the cross-process drift problem.**
   `LLaDAArm` previously carried only `(name, mode, cached, mask)`; every lever (`jotEnabled`,
   `creditDecodingEnabled`, `nBuf`, …) was a **process-global CLI flag**. *That is the structural
   reason every arm in the WP-3/WP-4 backfill ran in its own process* — it was forced by the bench,
   not an oversight, and it is why sub-10% effects were unresolvable (±6–10% cross-process drift vs
   ~0.33% within-process CV). `LLaDAArm.overrides` fixes it generally.

**Known limitation — arm-major ordering** *(found 2026-07-14; not fixed, deliberately)*. The bench
loops `for arm { for run { … } }` (`LLaDABench.swift:1174`), so each arm occupies a **contiguous
time block** within the process — arm identity is confounded with elapsed time. One process removes
the ±6–10% *cross-process* drift, but a within-process ramp (thermal, fragmentation) still tracks
arm order. Scale check: within-arm/within-prompt CV is ~0.33% median, but run-to-run suite-mean
spread reached **10.96% max** in the backfill, so this is not always negligible.

- **For this experiment: acceptable.** The expected deltas (~27–37%, see below) dwarf the drift.
- **For credit decoding's +1.8%: NOT acceptable.** Interleaving must be run-major
  (`for run { for arm { … } }`) before that question can be settled, or the ordering bias is the
  same size as the effect. **This is the one thing standing between `LLaDAArm.overrides` and
  actually resolving credit decoding — do it as part of that work, not this one** (it touches the
  loop every existing workflow uses, so it wants its own change and its own control run).

**Dev-host smoke (2026-07-14) — pipeline validation, explicitly NOT a measurement.** Ran on the
M1 (n=3–4 prompts, gen-16, chat only) purely to prove the plumbing emits what the Studio needs:
effective echoes correct per arm, all four arms sharing one `processId`, `totalMemoryMB` present.
The numbers are dev-host and worthless as a result — **but the sanity gate passes**: MoE 35.8% +
attention 37.3% + remainder 26.9% = 100.0%, every term positive, nothing exceeding the forward.
That is the first evidence the method does not blow up the way the microbench did (161%). It says
the *approach* is sound; it says nothing about the Studio's numbers, which is what the
pre-registered rule reads.
