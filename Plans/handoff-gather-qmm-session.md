# Handoff — the `gather_qmm` / forward-attribution session (2026-07-14 → 07-15)

**Read this first if resuming.** Full detail: `Plans/gather_qmm_handoff.md` §5–§12 (the primary
record) and `Plans/experiments-master-list.md` (WP-6b…WP-6g rows + §3 "Measurement Method").

---

## 1. Where we started and where we ended

**Started**: a hand-off asking whether to build a custom Metal `gather_qmm` kernel targeting
dequantization overhead ("Case B"), with Case A (divergence/tiling) already closed.

**Ended**: after ~15 Studio benchmark campaigns, **Case B is the only surviving lever** — and every
alternative Claude proposed along the way was measured and killed. The original hand-off's instinct
was right; the intervening two days were the cost of proving it.

## 2. The measured forward budget (Studio, 60-core M2 Ultra, 27.5 ms/forward at T=32)

| component | share | how measured |
|---|---|---|
| **`gather_qmm` expert GEMMs** | **42.9%** | `full − attr-no-experts` ✅ valid marginal |
| routed MoE total (router + GEMMs) | 56.3% | `full − attr-no-routed` ✅ replicated (56.2 / 56.3) |
| attention (incl. KV growth) | 14.3% | `full − attr-no-attn` ✅ |
| shared expert | 4.2% | ✅ |
| **router** | **1.7%** | `q-cached − reuse-999` ✅ (the ablation split said 13.2% — **wrong**, see §4) |
| lm_head | ~3% | bandwidth floor; ablation inconclusive |
| remainder (norms + sampler + loop) | ~21% | by subtraction — **inferred** |

**Control**: `attr-full` reproduces the served `q-cached` baseline to +0.1–1.4% across four
independent rounds. **Sanity gate**: terms positive, sum 100%.

## 3. Verdicts — everything is measured, nothing is speculation

| lever | verdict | number |
|---|---|---|
| **Case B (`gather_qmm` inner loop)** | **ALIVE — the only survivor** | ~7.5 ms (~27%) of it is *not moving bytes* |
| blockLength 64/128 | REFUTED | steps/block 1.72× vs 1.18× break-even → −21% / −66% TPS |
| router reuse (N=2/4) | REFUTED | −14.3% / −24.1% TPS; steps/block 1.35× |
| fused-router megakernel | REFUTED | ceiling **1.7%**, not the 13.6% Claude claimed |
| `argPartition` for top-k | REFUTED | subset of that 1.7% |
| Alpha-MoE expert fusion | REFUTED | 0.39% (intermediates are 0.9% of MoE traffic) |
| FP16 experts (drop quantization) | REFUTED | **4-bit is 1.50× FASTER** — quantization vindicated |
| Case A (divergence/tiling) | closed, now *measured* | production T = **exactly 32.0** on every row |

### Case B, precisely

- `gather_qmm` is **not** memory-bound: 7× fewer bytes made it **25% SLOWER** (`.moeFixedExperts`).
- It sits at **20% of bandwidth peak, 12% of compute peak, 207 µs/call** — bound by none of
  bandwidth, compute, or dispatch.
- FP16 probe decomposition: the same gather family moves bytes at **433 GB/s** (FP16) vs **162
  GB/s** (4-bit). 4-bit at FP16's rate would take 4.46 ms; it takes 11.92 ms ⇒ **~7.5 ms ≈ 27% of
  the forward is not moving bytes.**
- **Caveat (inferred)**: assumes `gatherQuantizedMM` *could* hit `gatherMM`'s 433 GB/s. Different
  kernels. **7.5 ms is a ceiling, not a promise.**

## 4. Methodology lessons (the session's real output — read before any new measurement)

1. **A microbench wrapping each op in `eval()` does not measure production**, and the distortion is
   **host-dependent**: the same harness reconciles on the M1 (63% MoE) and yields **161% of the
   forward** on the Studio. WP-6b is retracted; never size a module from an isolated op timing on a
   fast host.
2. **Ablation deltas are marginals, not a partition.** A delta says *"what does removing X from
   THIS config cost"*, not *"what is X worth"*. **Work that hides behind other work is free until
   the work it hides behind goes away** — the router read 13.2% in a GEMM-less arm and **1.7%**
   against a full forward (7.7× error, killed a two-day recommendation). **Rule: difference every
   arm against the FULL forward, never against another ablation.**
3. **Two arithmetic identities beat any test**: the *sanity gate* (terms positive, sum ≤ 100%) and
   the *control arm* (unablated must reproduce the baseline). Between them they caught WP-6b's
   161%, a `.lmHead` ablation reading 86% (MLX had DCE'd the entire transformer), and a
   wrong-overload wiring bug.
4. **Timing cannot distinguish "free" from "deleted."** Assert ablations **behaviourally** (output
   must change), never by reasoning about the optimiser.
5. **Per-arm params, one process.** Cross-process drift is ±6–10% vs ~0.33% within-process. This is
   why WP-4d's +1.8% was never resolvable.

## 5. Code landed this session (all default-off / diagnostic unless noted)

| what | where |
|---|---|
| `ModuleAblation` enum + in-situ ablation switch | `Packages/DiffusionCore/Sources/ModuleAblation.swift`, `LLaDA2MoE.swift`, `LLaDA2DecoderLayer.swift`, `LLaDA2MoeModel.swift`, `DiffusionEngine+Entry.swift` |
| `effectiveModuleAblation` echo | `DiffusionEngine.swift`, `DiffusionEngine+Scheduler.swift` |
| **`LLaDAArm.overrides`** — per-arm `GenerationParams` (**the fix that makes sub-10% effects resolvable**) | `Tools/diffusion-bench/Sources/LLaDABench.swift` |
| `RoutingTrace` + `--dump-routing` | `Packages/DiffusionCore/Sources/RoutingTrace.swift` |
| `ExpertDequantization` + `--dequantize-experts` (FP16 probe, ~31.5 GB, Studio only) | `Packages/DiffusionModel/Sources/ExpertDequantization.swift` |
| `routerReuseSteps` (experimental, refuted, default 0) | `GenerationParams.swift`, `LLaDA2MoE.swift` |
| **`stepIndex` FIX** (see §6) | `DiffusionEngine+Entry.swift` |
| Guard tests (9) | `Tests/DiffusionGenerationTests/ModuleAblationTests.swift` |
| `envValid` rule 1a (host-relative free floor) | `Tools/diffusion-bench/Sources/LLaDABench.swift`, `Plans/m6-logbook.md` |

Full suite: **118 tests, 0 failures**.

## 6. Bug found and fixed: FlashBlock never reused its cache

`stepIndex += 1` sat inside the **Elastic-Cache branch**, which the FlashBlock/JOT and
served-default branches `return` before reaching. With Elastic off — **every config ever benched** —
`stepIndex` stayed 0, so `isFirstStepOfBlock` was permanently `true`, and
`FlashBlockRunner.chooseStepKind` opens with `if isFirstStepOfBlock { return .refreshCache }`.
**FlashBlock paid full setup every step and never once took its reuse path.**

The parity test couldn't catch it: it verified at **τ=0**, where refresh-every-step *is* correct.
The one config that would have exposed the bug is the one the test didn't run.

**Fixed** (increments once per forward on every path; Elastic's semantics preserved exactly —
it previously incremented *after* its body and observed `stepIndex == N` during forward N, which
`stepInBlock` reproduces). WP-3b's reject **predates this fix and is invalid**. Ceiling is
unchanged either way: attention is 14.3%, so this can only explain *why* it lost.

## 7. WHERE TO PICK UP

### 7a. WP-3b re-bench — **BLOCKED: the reuse path is broken (new finding, 2026-07-15)**

Record: `scratch/wp3b.jsonl` (37/72 rows — **killed after ~23 min, hung**), `scratch/wp3b.log`.

**Two bugs, one behind the other.**

**(1) Why André's first attempt produced 0 FlashBlock rows**: `flashBlockEnabled = true` trips
`precondition(self.speculationK == 1)` in `generateCached`, and the bench defaults to
`--speculation-k 4`. **`speculationK` is an engine *constructor* argument, not a
`GenerationParams` field — so a `LLaDAArm.overrides` closure cannot set it.** Fix: pass
`--speculation-k 1` on the command line (which is also the fairer comparison — both arms at the
same K). ➜ **TODO: make the bench fail loudly when an arm's params conflict with the
process-global K, instead of trapping mid-run.**

**(2) With that fixed, FlashBlock HANGS.** It completed exactly one prompt and then spun forever on
the second. `sample` shows it live in `DiffusionEngine.denoisePhase` (`DiffusionEngine+Step.swift:170`)
at 30% CPU, actively dispatching Metal kernels — **not deadlocked, non-converging**. The one prompt
that did finish tells the story:

| `chat-capital` | TPS | steps/block |
|---|---|---|
| vanilla | 34.39 | 9.00 |
| flashblock | **5.13 (−85%)** | **28.50 (3.2×)** |

**⚠️ ATTRIBUTION IS CONFOUNDED — do not treat the cause as established.** The run was built from a
tree containing **André's uncommitted in-progress edits** to `FlashBlockRunner.swift` (a bounds
clamp: `safe_j = min(int(j), actual - 1)` on both `K_page`/`V_page` and `K_cur` tile loads) and
`LLaDA2Attention.swift`. Claude did not check for a dirty tree before running, and attributed the
hang to the `stepIndex` fix alone. **Three candidate causes remain live:**
1. the §6 `stepIndex` fix let `.reuseCache` fire for the first time ever, and that path is broken;
2. André's WIP kernel edits introduced or changed the behaviour;
3. an interaction of the two.

**To disambiguate** (cheap, ~10 min): run `wp3b-flashblock` with `--speculation-k 1` at
`HEAD` (stash the WIP) → isolates cause (1). Then re-apply the WIP → isolates (2). The `safe_j`
clamp is itself evidence that an **out-of-bounds tile read** existed in the kernel, which is a
plausible independent cause of both wrong attention and non-convergence.

*What is NOT in doubt*: FlashBlock did not converge on this build, and the one prompt that finished
was 3.2× the steps at −85% TPS.

**So WP-3b's status is worse than "invalid verdict": its core mechanism has never worked.** The old
"−40% TPS" number measured FlashBlock-as-vanilla-plus-overhead (reuse disabled by the bug); the new
behaviour is what the mechanism actually does.

**➜ Next step is NOT a re-bench — it is debugging `FlashBlockRunner`'s `.reuseCache` path** (start:
`LLaDA2Attention.swift:224`, `FlashBlockRunner.swift:462`). Only re-bench once it converges. And
remember the ceiling is 14.3% (attention's share), so this can never be a big win — **consider
whether it is worth the debugging at all.**

**Correction (Claude was wrong twice here)**: the denoising loop is **already capped** —
`DiffusionEngine+Scheduler.swift:194` breaks at `logicalStepsTotal > 1000` with a printed warning.
There is no missing step cap and **nothing to fix**. Claude asserted the loop was unbounded without
reading it.

**And that correction sharpens the diagnosis.** The guard **never fired** (`grep -c "Guard
triggered" scratch/wp3b.log` → 0), and no row was written in ~20 minutes. If FlashBlock were merely
looping — even at its terrible 28.5 steps/block — 1000 steps would trip the guard within ~1–2
minutes and emit a row. It did neither.

⇒ **It is not looping. It is stalled inside a single step** — the process sits in
`denoisePhase` → Metal dispatch at 30% CPU with a command buffer that never completes. **That is a
GPU-level hang, not a convergence failure**, which makes an out-of-bounds tile read the leading
hypothesis — precisely what André's WIP `safe_j` clamp targets. Note the clamp **was already in the
tree for this run and did not prevent the hang**, so it is either incomplete or aimed at a
different OOB site.

*Revised: the earlier "confidence collapses / never settles" story is downgraded — the 3.2× step
inflation on the one completed prompt is real and still needs explaining, but it is a separate
symptom from the stall.*

### 7b. The next experiment — **`dyntau` standalone** (highest value, ~1 line)

**Dynamic-τ (WP-2b-2) has never been measured alone on the Studio.** `dynamicTauAlpha=0.6` appears
in exactly two arms — `p3-combo` and `p4-combo` — **both with `nBuf=2`**. Its only Studio numbers
are welded to MultiBD, which is independently −17%. The master list even says the negative is "due
to MultiBD multiplier".

It cuts logical steps **−14.8% chat / −11.3% reasoning** (hardware-independent ⇒ transfers), and
changes only the threshold, so per-forward cost is unchanged ⇒ **≈ +15–17% TPS**. Bigger than Case
B's realistic win, already implemented, zero new code:

```swift
LLaDAArm(name: "dyntau", mode: .q, cached: true, mask: .strict) { $0.dynamicTauAlpha = 0.6 },
```
```
diffusion-bench llada --arms q-cached,dyntau --runs 3 --gen-length 128 --json scratch/dyntau.jsonl
```
Open item: quality — `scratch/wp2b_blind/` is pending André's blind scores.

**Why this and not Tier 3**: the roadmap's three levers are (i) fewer steps, (ii) cheaper steps,
(iii) more tokens/forward. **This session exhausted (ii) — the forward budget caps every one of
them.** (iii) has the Width Rule against it (MultiBD *and* blockLength both died the same way).
**(i) is uncapped — it multiplies against the whole 28 ms forward** — and every Studio winner so
far (JOT +21.9%, S2D2 +14.6/+19.8%) lives in (i)/(iii), not (ii). Dyn-τ is an accepted (i) lever
that was never cleanly measured.

### 7c. Tier 3 — priced against the budget, all fail entry

- **d²Cache**: ceiling = attention's 14.3%; WP-1a measured the active-KV reuse ceiling at ~4–5%
  architecturally. Entry needs "a hypothesis for why the incumbent underperforms" — **there is no
  incumbent**; WP-1a was rejected because the slot is *empty*. Don't.
- **FreeDave**: competes with S2D2, which **won**. No hypothesis for why it underperforms. Don't.
- **Sparse-dLLM**: entry condition (>8k context) not met; ceiling sits inside attention's 14.3%.
- **LocalLeap**: the only live one — a (i) lever. **Measure its incumbent (dyn-τ) first.**

### 7d. Case B — if pursued

**§3 step 1's Instruments profile on `gatherQuantizedMM` at T=32**, now targeted at a specific
sized question: *what is the ~7.5 ms doing, if it is not moving bytes?* Measure **occupancy,
register pressure, memory-level parallelism — NOT ALU time**. `xctrace` (CLI) only exposes Metal
System Trace; the occupancy counters need Xcode's GUI debugger.

Delivery: Case C route **(b)** — a **new** `MLXFast.metalKernel` from an inline Swift string, like
FlashBlock. **No mlx-swift fork.** (Patching MLX's kernel would require forking + rebuilding its
metallib from Xcode DerivedData on every edit, forever.)

## 8. Outstanding debts

- **WP-4d Credit Decoding**: default-on at **+1.8%**, below the drift floor, never quality-gated.
  The same comparison moved +6.2% → +1.8% between two clean runs (reasoning flipped sign). **Now
  resolvable** via `LLaDAArm.overrides` + run-major interleaving.
- **WP-4b ICE**: +4pp is **not significant** (McNemar p=0.481, gained 11 / lost 7); no arm in the
  7-arm sweep reaches significance and the grid is non-monotonic.
- **WP-4a TSCV**: **the α sweep is inert** — every α ∈ {0, 0.2, 0.5, 1.0, 2.0} gives exactly 21.0%
  at `tstart=0.9`. The whole effect is the cutoff; shipping "α=1.0" is arbitrary. Per-prompt vote
  data was never dumped, so +6pp could not be paired-tested.
- **Bench loops arm-major** (`for arm { for run { … } }`), so arm identity still correlates with
  elapsed time. **Run-major interleaving is needed before trusting anything near the noise floor**
  (i.e. before WP-4d can be settled).
- Blind quality sheets pending André: `scratch/wp2b_blind/`, `wp2a_blind/`, `m8_blind/sheet.md`.
- **Hardware**: this Studio is a **60-core** M2 Ultra. WP-6a/WP-6b and older docs say 76-core
  (the other bin). Bandwidth is 800 GB/s either way, so BW figures stand; any *compute*-peak claim
  keyed to core count is ~27% overstated.

## 9. Traps that cost time this session

- **The bench APPENDS to `scratch/llada_bench.jsonl` by default.** Two of Claude's wiring checks
  ran without `--json` and wrote debug rows into André's clean backfill (removed; `.precleanup`
  copy kept). **Always pass `--json`.**
- **`pgrep -f "release/diffusion-bench"` matches the watcher's own command line** and never exits.
  Use `pgrep -x diffusion-bench`.
- **Stale `.build` after stash/checkout churn** → `SIGBUS` in `RawRepresentable.rawValue` (an enum
  ABI mismatch). `swift package clean` fixed it. This is CLAUDE.md's documented hazard, live.
- **MLX traps on direct `@ModuleInfo` assignment** → use `update(modules:)`. And a model-wide
  `update` traps with `unexpectedStructure(key: "layers")` when the path set is *gappy* (layer 0 is
  dense, so `layers.1…19` can't rebuild the array). Update the leaf module directly instead.
- **`swift test` runs `xctest` which can wipe the seeded metallib** — re-run `Tools/seed-metallib.sh`
  after `--build-tests`, and prefer `xcrun xctest` directly for repeated runs.
