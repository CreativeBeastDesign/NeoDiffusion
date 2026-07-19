# Step 5 kernel logbook — fused MoE-gather `MLXFast.metalKernel`

Campaign record for writing the custom 4-bit gather-QMV kernel. Plan of record: the approved
Step 5 plan (2026-07-18); entry state and evidence: `Plans/step5-kernel-handoff.md`.
House rules apply: provenance labels (sourced / inferred / speculative), negatives recorded
with the same precision as wins, envValid discipline on every bench row.

## Entry state (2026-07-18) — all *sourced*

- Host: Mac Studio M2 Ultra (`Mac14,14`) — the target hardware. mlx-swift 0.31.6 / MLX core 0.31.1, Swift 6.3.3. Branch `kernel`, clean at `aa8e0fd`.
- Baseline profile (Step 5a, `scratch/captures/moe_block_t32.gputrace`): `affine_gather_qmv_fast_float_gs_64_b_4` = 50.9% of MoE encoder cost, **32.8% occupancy, 100 regs/thread, 0 spill, 69.6% ALU / 57.1% Integer limiter, F32 util 15.1%**.
- Sized prize (`gather_qmm_handoff.md` §12): ~7.5 ms/forward not-moving-bytes; expected +10–20% TPS, ceiling ~+37%. Standing instruction: accept a recorded negative if the MSL doesn't move the needle.
- Stock algorithm reference: `quantized.h` `qmv_fast_impl` (l.750) + gather `adjust_matrix_offsets` (l.1389). 4-bit/g64 constants: 2 simdgroups × 32 threads, 4 rows/simdgroup, values_per_thread 16, K-block 512, per-thread `x_thread[16]` FP32 + `result[4]` + 6 live pointers.
- Register-reduction levers, attack order: (1) threadgroup x-staging; (2) wider threadgroups; (3) compile-time dims → strength-reduced addressing; (4) scale/bias staging. MMA structurally ruled out — do not revisit.

## Log

### 2026-07-19 — Step 5 optimization loop

**Bench protocol tightened first**: `testFusedQMVRunnerMini` reps 10→50, warmup 3 — at
~1.3 ms/op the default 10 reps sat inside cross-process drift. All ratios below are
fused/stock within one process, 3 processes each, envValid true throughout. Baseline **v0**
(the committed walking skeleton, `9219fc2`) under this protocol: **0.842 / 0.855 / 0.845**.

**Iteration 2 — ROWS 4→8 per simdgroup: KEPT, first win.** *Sourced
(`/tmp/it2_bench_{1,2,3}.log`).* Each simdgroup accumulates 8 output rows per K-block instead
of stock's 4, halving x-load traffic per output element for +4 accumulator registers. Ratio
**0.829 / 0.828 / 0.830** vs v0's 0.842–0.855 (~2 pp), maxΔ = 0.0, 9/9 tests. Working tree
now carries this variant (`qmv_rows<*, 8>`, grid OUT/16, 64 threads/tg); trace captured to
`scratch/captures/fused_v2_rows8/` (dispatch-proven, 3).

**Iteration 3 — ROWS=16: REVERTED.** *Sourced (`/tmp/it3_bench_{1,2,3}.log`).* 0.838 / 0.833 /
0.847 — worse than ROWS=8 and noisier; +16 accumulator registers overshoot.

**Iteration 4 — 4 simdgroups/tg (128 threads, ROWS=8): REVERTED.** *Sourced
(`/tmp/it4_bench_{1,2,3}.log`).* 0.844 / 0.852 / 0.844 — fewer threadgroup launches doesn't
pay; per-thread registers unchanged, so no occupancy relief either.

**Iteration 5 — dual gate+up K-loop (step-6 probe, silu still MLX-side): NOT KEPT, ambiguous.**
*Sourced (`/tmp/it5_bench_{1,2,3}.log`).* One x_thread load per K-block feeding both weight
matrices' 8 rows (register profile ≈ ROWS=16). Ratio 0.824 / 0.842 / 0.854 — fully overlaps
the ROWS=8 band, high variance, no proven gain; reverted on simplicity. *Inferred insight:*
x is ~256 KB/layer-forward — cache-resident after first touch — so x-load traffic was never
the cost; the 128 MiB weight stream is. The it2 win was plausibly wider per-thread ILP, not
x reuse. Full step-6 (in-kernel silu, single glu output, one fewer dispatch + intermediate
round-trip) remains untried and is a *different* mechanism — but note its toy-fp16 rounding-
parity wrinkle (in-kernel silu skips the outDType rounding stock applies when out dtype is
fp16; production f32 is unaffected).

**v0 trace read (André, 2026-07-19)** — *sourced (Xcode, `fused_v0_skeleton/`)*: gateUp
dispatch = 41.22% of encoder cost (buffers 512 KiB + 128 MiB = x + wG, unambiguous), **ALU
limiter 56.43% / Integer 42.97%** (stock: 69.6% / 57.1% — the compile-time-dims win is real),
**occupancy ~33.9% ±10pp** (stock: 32.8% — unchanged ⇒ still register-capped). Encoder GPU
time 374.86 µs vs stock capture's 510.22 µs. Registers not yet located in the Xcode UI.

**Capture methodology addendum** (echoes the §3 in-situ note): two lessons. (1) The capture
invocation must be the LAST thing in its shell command — build/checkout churn in the same
compound after test exit raced the trace-daemon finalization and left 40-file/182 MB torsos
(complete bundles are ~84 files/1.2 GB; always verify before handing over). (2) CORRECTED
2026-07-19 (later): incomplete finalization is a general flake, not kernel-structural — the v3
(VPT=8, no TG memory) capture also came out 40-file twice before completing on a retry, and v2
needed a retry too. Procedure: after any capture, verify ~84 files / ~1.2 GB (a 40-file/182 MB
bundle is a torso Xcode rejects with "index file does not exist") and re-run until complete.
The earlier "v1 is structurally uncaptureable" reading is RETRACTED; v1 was plain bad luck
twice and can be re-captured with retries if its diagnostic ever matters again.

**v2 trace read (André, 2026-07-19)** — *sourced (Xcode screenshots, `fused_v2_rows8/`)*:
`custom_kernel_moe_gather_qmv_down` = **126 allocated registers (stock: 100), 0 spilled,
occupancy 26.51% (stock: 32.8%)**, 38.29% of its encoder (366.09 µs encoder GPU time). The
Shaders table covered only the encoder holding `qmv_down` + steel_gemms; `qmv_gate_up` lives
in a different encoder (MLX splits command buffers) — its numbers unread, not blocking.
Also visible: the `Bf4ISigmoidACf4OMultiply` SwiGLU elementwise at **6.05%** of that encoder
(the step-6 in-kernel-silu target), and fp16↔f32 copies (7.30% + 0.97%) that are *capture-test
artefacts* (fp16 capX) — production x is f32, where those casts are no-ops.

**Mechanism verdict (inferred, now well-evidenced): occupancy recovery is NOT what pays on
this kernel.** ROWS=8 RAISED registers 100→126 and LOWERED occupancy 32.8→26.5 yet beats both
stock and v0 — the banked winnings are ALU/Integer-limiter reduction (compile-time dims) +
one fewer dispatch + wider per-thread ILP. The Step 5a "raise occupancy above 32.8%" design
target is superseded by measurement; the pre-registered step-5 exit gate ("regs <100 AND
occupancy >32.8%") is met on speed (≥1.10×: yes, ~1.20×) but NOT on its mechanism clause —
flagged to André rather than silently rewritten.

**Iteration 6 — VPT 16→8 at ROWS=8 (aimed by the v2 read): KEPT.** *Sourced
(`/tmp/it6_bench_{1,2,3}.log`).* Halves per-thread x registers (x_thread[8], 256-value
K-blocks — stock's non-fast `qmv` layout, kept at ROWS=8). Ratio **0.823 / 0.806 / 0.794** vs
it2's 0.828–0.840 — bands don't overlap, kept. **Bit-exactness is gone by design from this
iteration on**: each thread now spans 8 input values, so accumulation order differs from stock
— pure FP-reorder noise, measured **~2e-5 relative** (Mini shapes: maxΔ 8.6e-06 absolute on
O(1) outputs, 1e-3 gate passes with ~100× headroom; toy dims: ~0.01 absolute on O(500)
outputs from unscaled N(0,1) weights). **Gate adjustment, flagged not slipped**: the five
toy-dim equivalence assertions switched from absolute 1e-3 (which de-facto demanded
bit-exactness at O(500) magnitudes) to relative rtol 1e-4 (`assertMatchesStock`, ~5× headroom
over observed noise); the production-shape absolute 1e-3 gate in `testFusedQMVRunnerMini` is
UNCHANGED and load-bearing. Justification: 2e-5 relative reorder noise is far below 4-bit
quantization error, and stock itself changes accumulation order across its own qmv/qmv_fast
shape dispatch. Trace: `fused_v3_vpt8/` (complete, retry 2).

**Iteration 7 (= step-6 fusion attempt) — in-kernel SwiGLU: NEGATIVE, REVERTED (both
configs).** *Sourced (`/tmp/it7_bench_{1,2,3}.log`, `/tmp/it7b_bench_{1,2,3}.log`).* A third
kernel (`qmv_rows_glu`) dual-accumulating gate+up in one K-loop with the silu(g)·u epilogue
in-kernel, engaged only when outDType == f32 (production; fp16 toy path kept the two-output
kernel so stock's fp16 rounding points stay mirrored). At ROWS=8: 0.825–0.838; at ROWS=4
(restoring it6's 8-accumulator total): 0.821–0.831 — both ≥ it6's 0.794–0.823. The saved
dispatch + elementwise (6.05% of encoder) + intermediate round-trip never beats the doubled
accumulator pressure. Step 6's guard clause fired as designed: measured, not assumed;
reverted to it6.

Loop state after 7 iterations: settled best = **ROWS=8 + VPT=8 at ~0.79–0.82** (≈17% faster than stock at SwitchGLU
level); two consecutive no-gains since the it2 win. The remaining levers are capture-aimed
(need actual regs/occupancy per variant) — pausing wall-clock probes for André's trace reads:
`fused_v0_skeleton/` (baseline), `fused_v1_tgstage/` (why did TG staging lose — regs or ALU?),
`fused_v2_rows8/` (current best — is there register headroom for packs_per_thread=4?).

**Iteration 1 — threadgroup x-staging (lever 1): REVERTED, wall-clock negative.** *Sourced
(`/tmp/it1_bench_{1,2,3}.log`, `/tmp/v0_bench_{1,2,3}.log`).* Staged the full pre-scaled input
row in TG memory (8 KB gate/up, 2 KB down), one barrier, both passes reading TG instead of
per-thread `x_thread[16]`; bias sum reconstructed with exact power-of-2 multiplies
(bit-identical arithmetic — equivalence stayed maxΔ = 0.0, 9/9 tests). Ratio **0.878 / 0.886 /
0.878** vs v0's 0.842–0.855 ⇒ ~3.5 pp regression, consistent. Wall-clock says the barrier +
TG round-trips cost more than the register relief buys at these shapes. **Why is not yet
known** — trace captured to `scratch/captures/fused_v1_tgstage/` (dispatch-count-proven, 3);
André's read of regs/occupancy vs v0 decides whether the register-pressure lever family is
mis-aimed (regs didn't drop — compiler hoisted TG reads) or capped (regs dropped, ALU floor
ate the gain). v1 source preserved at `/tmp/v1_runner.swift` this session; the working tree
carries v0.

### 2026-07-18 — Step 0: disk cleanup (chore)
Deleted the superseded host-local artefacts approved by André (empty/superseded isolated
captures, `captures_old/`, the 8-bit-experts and mxfp4 Step-4c diagnostic models): **39 GiB
freed**, keep-list (working trace + production artefacts) verified intact, git clean. *Sourced:
subagent report, `df` before/after 274→313 GiB.*

### 2026-07-18 — Step 2: walking skeleton — **GATE MET, bit-exact and 0.80× stock**

Naive structural port of `qmv_fast_impl` (2 simdgroups × 4 rows, 16 values/thread, 512-value
K-blocks, pre-scale + masked-nibble qdot, FP32 accumulate) into `MoEGatherQMVRunner` as two
`MLXFast.metalKernel`s (`moe_gather_qmv_gate_up`, `moe_gather_qmv_down`), grid
`(64, OUT/8, T·k)`. *Sourced (this session, Studio Mac14,14, mlx-swift 0.31.6 / MLX core 0.31.1):*

- **Equivalence: max |Δ| = 0.0 vs stock** at real Mini shapes (E=256, H=2048, I=512, T=32, k=8,
  fp16 weights + f32 x — the production dtype pairing), `LLaDAMoEDispatchBench/testFusedQMVRunnerMini`,
  envValid true. Toy-scale (H=I=512) equivalence + adversarial index patterns (all-same-expert,
  reversed-distinct) < 1e-3: `MoEGatherQMVRunnerTests`, 5/5 passed.
- **Timing: fused/stock = 0.801** at SwitchGLU level (0.0012 vs 0.0014 s/op, warmup 2 + 10 reps)
  — the skeleton already beats stock, plausibly from 3 dispatches → 2. *Inferred:* not yet a
  TPS claim; single-layer microbench, no envValid JSONL row discipline beyond the echo above.
- **One real bug found and fixed** (recorded per house rules): rounding the SwiGLU intermediate
  through `x.dtype` instead of stock's *promoted* dtype (fp16 x + f32 scales → f32) cost
  ~1.4e1 absolute error through the 512-term down GEMV. Stock `gatherQuantizedMM` returns the
  promoted type; the runner now mirrors that exactly (`outDType` in `forward`).
- Eligibility guard added (`MoEGatherQMVRunner.isEligible`: g64/4-bit, H%512==0, I%512==0 — the
  qmv_fast alignment family); ineligible shapes provably fall back to stock (test:
  `testIneligibleShapeFallsBack`, dispatch-counter-verified). `MoEFusedQMVConfig.dispatchCount`
  exists so no equivalence test can silently pass against a fallen-back stock run.

### 2026-07-19 — Step 3: correctness gates — **GREEN** (Sonnet subagent, reviewed)

Broadened `MoEGatherQMVRunnerTests` to 9 tests, all passing. *Sourced (`/tmp/step3_unit.log`,
`/tmp/step3_full.log`):*
- Seed × dtype sweep (seeds {3,17,42} × x ∈ {f32, f16}, incl. the pure-FP32 arm = the AGENTS.md
  toy-config FP32 parity gate), production dtype pairing (fp16 scales + f32 x, output dtype
  matches stock), T=1 decode-shaped edge — all maxΔ < 1e-3, every arm dispatch-counter-verified.
- Fixture toy config is H=128/I=64 — **ineligible** for the kernel (not %512), and
  `CoreFixtureTests.testMoEBlock` never quantizes anyway; `testFixtureScaleDimsFallBackInert`
  proves flag-on leaves fixture-scale quantized models untouched (allClose + no dispatch).
- **Full suite, default flag state**: `Test Suite 'All tests' passed … Executed 133 tests, with
  23 tests skipped and 0 failures (0 unexpected) in 382.497 (382.510) seconds`. All 23 skips are
  the known `XCTSkipUnless` fixture/opt-in gates; count grew 120→133 from the Step 1–3 additions.

**Checkpoint reached** (per the approved plan): walking skeleton + gates green; pausing for
André before the Step 5 optimization loop.

### 2026-07-18 — Step 1: scaffolding (done)
`MoEGatherQMVRunner.swift` + `NEODIFFUSION_FUSED_QMV` flag + SwitchGLU branch, stub MSL bodies,
delegated to a Sonnet subagent per the token-economy scheme. Result pending.

**Draft prepared meanwhile** (Fable): full naive-port MSL body drafted against the stock source
(scratchpad `step2-msl-draft.metal`), pointer arithmetic cross-checked line-by-line against
`qmv_fast_impl`. One flag raised for step 2, *sourced from the trace*: the captured production
kernel is `affine_gather_qmv_fast_float_gs_64_b_4` — activation dtype **float32**, so the
equivalence/timing work must run T=float, not the float16 the stub scaffold assumes.
