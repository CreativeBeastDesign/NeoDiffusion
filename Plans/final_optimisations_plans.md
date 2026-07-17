# NeoDiffusion — Final Optimisations Plan

**Status**: approved, in progress
**Last updated**: 2026-07-17
**Author**: Claude (session 2026-07-15), reviewed by André: ☑ approved 2026-07-17 ("the plan would be good to go"), with the P5 amendments folded in (§1.4)
**Progress**: **P1 DONE** (Studio on Swift 6.3.3; mlx-swift pinned `.exact("0.31.6")` in `Package.swift` — both hosts, resolve verified no-op on the Studio). **P4 DONE** (per-row `toolchain` provenance; §1.1). **P5 DONE** (§1.4 — two of the three sheets flipped a decision). **P6 DONE** (full suite green on the new toolchain: 120 executed, 0 failures). **P2 + P3 DONE** (run-major interleaving with `--arm-major` escape hatch; speculationK pre-flight; §1.1). **§1.5 supersedes Step 1**: it was already run, and dyn-τ is reasoning-only. Next: P7 (Studio re-freeze) + the Step 1 α=0.3 arm (Studio), pairing with P8/P9; interleaving validation can run on the M1.
**Inputs**: `Plans/handoff-gather-qmm-session.md` (forward budget + methodology), `Plans/experiments-master-list.md` (verdict matrix + §1a provenance warning), `Plans/phase-3-optimisation-roadmap.md`, `Plans/phase-4-further-optimisations.md`, `Resources/alpha-moe-megakernel.md`.
**Scope**: everything still live after the 2026-07-14/15 attribution campaign, **excluding** the AR-model work packages (WP-4b Guided Diffusion, WP-4e Plan Conditioning) and PRR controller training (WP-4f), per André's direction. The M2-Ultra re-run debt (MLX version reconciliation + backfills) is a *prerequisite thread*, not an optimisation, and is folded in where it gates a step.

---

## 0. Ground truth this plan builds on

The measured Studio forward budget (WP-6c, control-validated, sums to 100%):

| component | share | status as a lever |
|---|---|---|
| `gather_qmm` expert GEMMs | **42.9%** | **Case B — only surviving "cheaper steps" lever** (~7.5 ms ≈ 27% of the forward is not moving bytes; ceiling, not promise) |
| remainder (norms + sampler + selection + loop) | **~21%** | **unattributed — inferred by subtraction only** |
| attention (incl. KV growth) | 14.3% | capped; FlashBlock broken; park |
| shared expert | 4.2% | too small |
| lm_head | ~3% | bandwidth floor; dead |
| router | 1.7% | dead (killed fused-router + argPartition) |

Governing lessons (master list §4): the **Redundancy Rule** (six confirmations — assume any paper's headline is already banked by the baseline until shown otherwise), the **Width Rule** (two confirmations — "more tokens per forward" must clear its steps/block break-even first), and the **measurement discipline** of §3 (in-situ ablation vs full forward; sanity gate + control arm; per-arm params in one process; behavioural assertion of ablations).

### Verdict on the two questions this plan answers

1. **Alpha-MoE megakernel (or a variation)?** As written: **dead by measurement** `[Sourced]` — WP-6f refuted its mechanism twice over. Its fusion saves expert-*intermediate* traffic, which is 0.9% of MoE bytes here (0.39% ceiling), and `gather_qmm` is not bandwidth-bound anyway (the coalescing probe made byte-reduction 25% *slower*). Its *spirit* — a hand-tuned persistent kernel for the MoE inner loop — is exactly **Case B route (b)** (a new `MLXFast.metalKernel`, no MLX fork), which is already the plan of record. There is no second, separate Alpha-MoE opportunity beside Case B. The Hopper-specific tricks (WGMMA interleaving, TP-sharded persistent down-proj) have no Metal analogue worth porting `[Inferred]`.
2. **Anything still viable in the phase-3 / phase-4 docs?** Phase 3: only **LocalLeap** (conditional, after Step 1 below) and the **selection-set-fusion / §13.1 escape-hatch pair** (decided by Step 3 below). Phase 4 (minus AR + PRR): **TSCV upgrades** (the best accuracy lever), **ICE statistical power**, and **settling Credit Decoding**. Everything else is correctly closed — d²Cache (empty slot, ~4–5% ceiling), FreeDave (S2D2 won), Sparse-dLLM (entry unmet), fused-router (1.7%), Window-Diffusion (non-starter), WP-4f's rule-based fallback (dyn-τ by another name).

---

## 1. Prerequisite thread (gates several steps; not itself an optimisation)

### 1.1 Gates for this plan's steps

| item | why it gates | effort |
|---|---|---|
| **P1. Studio toolchain ≥ Swift 6.3, then pin ONE mlx-swift version on both hosts — as an `.exact` pin** | §1a of the master list: hosts have never run the same MLX; every cross-host inference is confounded. `.exact` (not `from:`) is required or each host keeps rewriting `Package.resolved` in the opposite direction. Also unlocks Step 4a (newest-MLX check). | Xcode install + pin |
| ~~**P2. Run-major interleaving in `LLaDABench`**~~ **DONE 2026-07-17** — execution order is now run-major (interleaved) by default; each arm is sampled across the whole elapsed span, so arm identity no longer correlates with warmup/thermal drift. `--arm-major` restores the legacy order — the A/B that validates the interleaving itself (a good host-agnostic M1 job). Both orders verified. | Arm identity correlated with elapsed time; nothing near the noise floor (Credit's +1.8%, composability deltas) was resolvable without it. | done |
| ~~**P3. Bench fails loudly on arm/process-global `speculationK` conflict**~~ **DONE 2026-07-17** — pre-flight scans every arm's *effective* params (post-`overrides`) and, if the process `speculationK != 1`, errors before the first generation listing any arm that needs K=1 (`flashBlock`/`faithful-JOT`), instead of the engine's `precondition` crashing mid-run. Verified: `--arms q-cached,wp3b-flashblock` at K=4 fails loudly; at `--speculation-k 1` it passes the gate. Note: flashblock is **not** in the no-filter default arm set (line 621 excludes `wp3b-`/`dt-`/`lf-`/…), so the trap only bites explicit `--arms` selections. | Handoff §7a TODO — prevents a repeat of the mid-run trap. | done |
| ~~**P4. Bench emits the toolchain + resolved-MLX-version field per JSONL row**~~ **DONE 2026-07-17** — every row now carries `toolchain: {mlxCoreVersion, mlxSwiftPackage, swiftCompiler}` beside the existing `host`, echoed in the startup banner (`build: host Mac14,14 \| MLX core 0.31.1 (mlx-swift 0.31.6) \| Swift 6.3`). Schema + trust levels in the bench README. **Landed before P7, as required.** | Master list §1 made this mandatory (2026-07-15): `Package.resolved` is not evidence of what linked; without the field a cross-host row cannot be interpreted. **Confirmed the hard way this session**: the 0.31.4-vs-0.31.6 provenance of §1.5's factorials had to be inferred from a filename suffix and a commit message. | done |
| ~~**P5. André: blind sheets**~~ **DONE 2026-07-17** — sheets retrieved from the dev M1 to `blinds/{m8,wp2a,wp2b}_blind/` (they were never lost: `scratch/` is gitignored, so M1 artefacts never reach the Studio) and scored. **Two of the three flipped a decision — see §1.4.** | Three default-on/preset decisions were quality-gated on these; all three now have verdicts. | closed |

### 1.2 The P1 revalidation cascade (everything a new toolchain + MLX version invalidates)

Step counts and kernel timings are deterministic **per (host, MLX version)** — so changing the Studio's MLX version obsoletes every existing Studio row. In order, after P1:

| item | why | effort |
|---|---|---|
| ~~**P6. Full test suite green on the Studio's new toolchain**~~ **DONE 2026-07-17: `Test Suite 'All tests' passed` — 120 executed, 20 skipped, 0 failures, 374 s, on Swift 6.3.3 + mlx-swift 0.31.6 (MLX core 0.31.1).** Every load-bearing LLaDA suite ran and passed: `ForwardParityTests` (M4: top-1 48/48, max \|Δlogit\| 2.1e-06), `DenoisingLoopParityTests` (M5a/b 16/16 token-for-token; M5c sync audit in budget), `MultiBDParityTests`, `S2D2SpeculationTests` (verifier ≡ sequential-AR anchor), `WP2bTests`, `CreditDecodingTests`, `JotTests`, tokenizer parity 100/100. **The 20 skips are all `XCTSkipUnless` fixture/opt-in gates, none a LLaDA correctness test**: Sumi suites (different model, fixtures absent on this host — `SumiRealWeightSmoke` 3, `SumiLeverExperiments` 3, `SumiLMHeadPrecision` 2), `LLaDAMoEDispatchBench` 5 + the M6 bench cases (env-gated, see P8), `JotTests` 2, and `testWeightsLoader` (wants `scratch/dummy_mlx/model.safetensors` — another host-local `scratch/` artefact). **Seeding held**: `Tools/seed-metallib.sh` before the run survived `swift test`'s rebuild into the xctest bundle. | The toolchain itself changed; nothing downstream is trustworthy until this is green. **It is.** | done |
| ⚠️ **Bench-running trap found while doing P6** (record, costs an hour otherwise): **never run `swift test \| tail`** — under zsh the pipeline reports *`tail`'s* exit code, so a failing suite still exits 0, and the truncation eats XCTest's summary line. Redirect instead: `swift test > log 2>&1; echo $?`, then read `Test Suite 'All tests'`. Also: the log *ends* with the **swift-testing** runner saying `✔ Test run with 0 tests in 0 suites passed` — this project is XCTest, so that line is expected and is **not** the result. Reading it as one reports a green suite that never ran. | methodology | — |
| **P7. Re-freeze the Studio `q-cached` baseline** under the pinned MLX. ~~all existing Studio rows are mlx-swift 0.31.4~~ — **partly done already**: commit `66494bd` migrated the Studio to 0.31.6 and refreshed wall-clock benchmarks, so the Jul-16 `*_0316`/`refresh_*`/`leverfresh_*` rows are already 0.31.6. **Scope reduced to: confirm coverage of the M6 metric set and re-state the frozen numbers.** | §1a: cross-version comparisons are confounded exactly like cross-host ones. **De-risked by §1.5 F-d**: the 0.31.4→0.31.6 bump left counters byte-identical and moved wall-clock a near-uniform +6–8%. | 1 bench run |
| **P8. Re-run `LLaDAMoEDispatchBench`** before trusting the `gatherQuantizedMM` dispatch choice. **How** (found in P6): the suite is env-gated — it skipped 5 cases in the normal run; use `NEODIFFUSION_M6_BENCH=1 swift test --filter LLaDAMoEDispatchBench`. | CLAUDE.md's explicit re-run trigger: "after any MLX/GPU/shape change". The ranking has never been validated on the Studio at all (M1-only, 0.31.6). **Note §1.5 F-e**: the trigger should read on `mlxCoreVersion` (0.31.1), not the mlx-swift package pin — the dispatch kernels live in the core. | 1 bench run |
| **P9. Re-run the WP-6c `attr-full` control arm** (and, if it drifts >~2% from P7's baseline, the full attribution arms) | This plan prices every step against the 42.9/21/14.3% budget — measured under 0.31.4. If a new MLX shifts the budget, Steps 3–5's expected values shift with it. **§1.5 F-d says the trigger fires**: the bump moved wall-clock +6–8% with counters unchanged, and not perfectly uniformly per-arm (a control-vs-arm delta moved 1.3 points), so **assume the full attribution re-run is needed**. The budget shares are the one thing F-d cannot reassure you about: a uniform TPS gain says nothing about *which component* got faster, and `gather_qmm` (42.9%, Step 5's whole premise) is exactly the kind of quantized kernel ml-explore iterates on. | 1–2 bench runs |
| **P10. Spot-check the WP-6a adaptive-dispatch crossovers** (T=128/192) | Default-on, dormant, and its crossover thresholds were measured under the old MLX; a changed sort or qmm kernel moves them. Cheap to confirm, embarrassing to have silently wrong. | 1 short run |

Note: re-benching the *accepted winners* (JOT, S2D2, dyn-τ, Credit) under the new baseline is **not** a separate item — Step 2's composability sweep includes each as a single-lever control arm, so the backfill and the sweep are one campaign.

### 1.3 Pre-existing Studio debts (independent of P1, still open from Phase 2)

| item | what it validates | notes |
|---|---|---|
| **P11. BF16 parity run (M4/M5 gate)** | Token-for-token match vs the reference (temp 0, ≥20 prompts, S/Q modes, eos on/off), prefix cache off → then identical with it on. **The correctness anchor has never run.** | Per roadmap §0.1(3): if it surfaces a forward-level discrepancy, scheduling verdicts survive but absolute quality numbers re-anchor. Do it before trusting Step 6/7 accuracy numbers too far. |
| **P12. M6 both-machines gate** | The frozen dev baseline's Studio twin, on the full M6 metric set. | Largely subsumed by P7 if run on the same suites — fold them into one session. |
| **P13. M8 scored-set comparison on the Studio** | The 4-bit-vs-BF16 quality comparison on the scored set (M1's was flip-rate only). | Needs BF16 inference ⇒ Studio-only by memory; pairs naturally with P11's fixture session. |

Backfill decisions that CLAUDE.md assigns to "the Studio" and where they land in this plan: **WP-2b default-on** → Step 1 (now a 3-arm α sweep, §1.4); **WP-2a verified-conservative preset** → **CLOSED negative 2026-07-17** (P5 sheet; `wp2a-logbook.md` §8 — preset withdrawn, no Step-2 arm); **WP-1b net-TPS gate/presets** → Step 2 (nBuf=2 arm, now quality-gated per §1.4); **Spiffy runtime** → can likely be closed *on paper*: WP-6d measured the Studio's wide-forward multiplier (1.63× at 2× width), and Spiffy needs (1+D)× width at D=3–8 — apply the Width-Rule break-even to the recorded `draft_graph.json` ceiling before spending any bench time `[Inferred]`. **Note**: `draft_graph.json` is a dev-M1 artefact and, like the blind sheets, lives only in the M1's gitignored `scratch/` — retrieve it from the M1 rather than re-running the calibration.

### 1.4 P5 outcome — what the blind sheets changed (2026-07-17)

All three sheets scored against their own sealed keys (the WP-1b scoring trap — `--score` without `--out` silently merging against the default M8 key — was avoided; each run printed its key path).

| sheet | result | verdict |
|---|---|---|
| **m8** (strict vs referenceBias) | 10 ties, 1–1 | **Wash — closed.** No basis to switch; `.strict` stays incumbent. M8's last open item, and CLAUDE.md §2.7's, is closed. |
| **wp2a** (s2d2-t95 vs qmode) | qmode 3, s2d2 1, 8 ties | **Preset WITHDRAWN.** The verified-conservative preset is ~25% slower than Q-mode by `wp2a-logbook` F6 and existed *only* for a quality edge; the sheet shows none (the 2 substantive Q-mode wins name a factual error — 250 ml milk — and a regex without range validation). S2D2 stays landed, tested, default-off. |
| **wp2b** (α=0.6 vs static) | static 2, α=0.6 **zero**, 6 ties | **α=0.6 fails on chat, stands on reasoning.** Both losses show *token corruption* in the α=0.6 arm ("a big, of,, and"; "Pancakes for Two 2"; 1 tsp salt), both on chat; reasoning 4/4 ties. |

Three consequences for the steps below, all recorded in `wp2b-logbook.md` §8:

1. **α was picked on speed alone** (F1's deciders were deterministic counters; quality was deferred to this sheet) — so α=0.6 winning the sweep was never a quality statement, and the sheet says it sits past the quality knee on chat. **α=0.3 is the candidate** (chat −6.5%, post/blk 1.06).
2. **post-steps/block is a leading indicator of the defect** (F9): α=0.3 → 1.06, α=0.6 → 1.16 (2/4 chat corrupted), α=0.9 → 1.62 (F1 already flagged its churn), **nBuf=2 + α=0.6 → 1.90**. The mechanism is F1's own: a low late-block τ floor buys acceptances Δ must repair, and the repair sometimes comes up short.
3. **The one configuration already shipped as a preset is the least-scored one** (F10): `p3-combo` *is* nBuf=2 + α=0.6 at post/blk 1.90, accepted in `wp2b-logbook` §7 on step/wall-clock while the quality gate was open. Suspended — and it was net-negative wall-clock on the Studio anyway (−12.3% to −20.4%), so this costs nothing measured.

**Methodology finding (F12), applies to every step below**: the scripted checks passed *every* row — `max4gramRepeat` 1, no mask leaks, comparable lengths — on text reading "a big, of,, and". 4-gram and mask-leak heuristics detect *degeneration*, not *local incoherence*. The "objective checks clean" language in the WP-2b/WP-1b logbooks is retracted as evidence of non-degradation. An 8-prompt blind sheet is currently the **only** instrument that catches this defect class, which is an argument for keeping sheets on the critical path (§4) — and, separately, for the §2 Step 6(iii) normalized-string work to grow a local-coherence check.

**Provenance caveat** `[Inferred]`: all three sheets are dev-M1 generations and trajectories are host-dependent (M1↔Studio divergence is GPU floating-point, not MLX version). The specific corruptions will not reproduce verbatim on the Studio; the *mechanism* is a threshold-floor property, not a numerics one, so it should — and Step 1 re-measures post-steps/block as the indicator.

---

### 1.5 Step 1 is already answered, and the answer is smaller than projected (2026-07-17 analysis)

Re-analysis of the existing Studio factorials (warmup rows and `envValid: false` excluded; 143/144 rows usable in each; all thermal-nominal, swap 0). **mlx-swift 0.31.6 arm, Δ vs the `q-cached` control in the same process:**

| suite | dt-tau (α=0.6, nBuf=1) TPS | logical steps | post/blk | blind sheet |
|---|---|---|---|---|
| **chat** | **+0.2%** | **+4.8%** (worse) | 1.35 | **2/4 corrupted** |
| code | +3.5% | −8.0% | 1.65 | never scored |
| **reasoning** | **+13.8%** | **−13.6%** | 1.70 | **4/4 ties** |

**F-a — dyn-τ is a reasoning-only lever on the Studio, worth ≈+14%, not the projected +15–17% across the board.** `[Sourced]` The plan priced Step 1 off the dev-M1 step savings (chat −14.8%, reasoning −11.3%). On the Studio, reasoning converts as expected (−13.6% steps → +13.8% TPS) — but **chat's saving does not merely shrink, it reverses sign** (M1 −14.8% steps → Studio **+4.8%**). Dyn-τ makes Studio chat slower in counters *and* flat in wall-clock, while raising churn (post/blk 1.35 vs baseline ~1.0).

**F-b — chat is dead on two independent axes, which is what makes it decisive.** `[Sourced]` The wall-clock/counter evidence (no gain) and the blind sheet (visible corruption, §1.4) were produced by different instruments, on different hosts, and agree. α=0.6 on chat buys nothing and costs text quality. **This also undercuts §1.4's α=0.3-for-chat recommendation** `[Inferred]`: if the *aggressive* α makes Studio chat steps go up, the gentler one is unlikely to save much — the M1's α=0.3 chat −6.5% probably transfers no better than its α=0.6 −14.8% did. Run the arm (it is free), but expect ~0.

**F-c — counters are NOT host-independent, contra the §0.1 protocol's working assumption.** `[Sourced]` A sign reversal on a deterministic counter between M1 and Studio is a direct counterexample to "hardware-independent metrics decide; wall-clock is host-scoped". It corroborates the known M1↔Studio GPU-floating-point step divergence, but the magnitude here (−14.8% → +4.8%) means **dev-host step counts cannot be treated as portable verdicts for threshold-sensitive levers** — they gate what is worth measuring on the Studio, nothing more. Every dev-host "algorithmic ACCEPT" resting on counters alone inherits this caveat.

**F-d — the mlx-swift 0.31.4 → 0.31.6 bump left every counter byte-identical and bought a uniform ≈+6–8% wall-clock.** `[Sourced]` Same 4 arms, same 3 suites, identical `logicalSteps` (chat 627/657/570/582; code 339/312/261/204; reasoning 375/324/315/222) and identical post/blk, across the two runs. TPS: chat 47.36 → 51.13 (+8.0%), code 97.71 → 103.59 (+6.0%), reasoning 75.07 → 79.53 (+5.9%). **This substantially de-risks the P1 cascade** (§1.2): all step-count results port across the bump unchanged, and arm rankings are preserved — the bump is close to a scalar on wall-clock. It does *not* make P7/P9 optional: the shift is not perfectly uniform per-arm (dt-tau chat gained +6.7% vs the control's +8.0%, moving its delta from +1.5% to +0.2%), so P9's ">~2% drift" trigger fires and absolute WP-6c budget shares must be re-confirmed. But the cascade is a re-anchoring, not an invalidation.

**F-e — "mlx-swift 0.31.6" and "MLX 0.31.6" are different numbers; this plan's version talk is the *package* version.** `[Sourced]` P4's runtime query (`mlx_version()` against the linked library) reports **MLX core 0.31.1** on a checkout whose `Package.resolved` pins **mlx-swift 0.31.6** — the Swift package and the C++ core it bundles version independently. Everywhere this plan, the master list and the logbooks say "mlx-swift 0.31.4 / 0.31.6" they mean the package; rows now record both (`toolchain.mlxSwiftPackage` vs `toolchain.mlxCoreVersion`) so the mapping is never re-derived by hand. Practical consequence for Step 4a: "newest mlx-swift" and "newest MLX kernels" are not the same upgrade — a package bump may or may not move the core, and the core is where `gather_qmm` lives. Check `mlxCoreVersion`, not the pin, when asking whether an upstream kernel fix has landed.

**Consequences**: Step 1 collapses to one α=0.3 arm. Step 8's LocalLeap entry condition is *materially more open* than the plan assumed — dyn-τ, its Tier-3 competitor for the same slot, now demonstrably leaves chat and code unimproved on the Studio, so "a measured step-reduction gap remains on its lever" is satisfied for 2 of 3 suites. Step 2's dyn-τ arms should be **reasoning-only** unless the α=0.3 arm surprises.

## 2. Ordered next steps

Ordering principle: expected TPS/accuracy per unit effort, with measurements that *decide other steps* promoted above implementations they gate.

### Step 1 — Dyn-τ standalone on the Studio → **ALREADY RUN (2026-07-15 and -16); result below (§1.5). Only the α=0.3 arm is outstanding.**

> **This step's premise was stale within a day of the plan being written.** The 2×2 factorial (`q-cached` × `dt-tau` × `dt-mbd` × `dt-tau-mbd`) was run on the Studio at 3 runs × 12 prompts × gen-128 on 2026-07-15 (mlx-swift 0.31.4, `scratch/dyntau_factorial.jsonl`) and re-run on 0.31.6 (`scratch/dyntau_factorial_0316.jsonl`, commit `66494bd`). `dt-tau` **is** dyn-τ standalone (engine echoes verified: `dynamicTauAlpha 0.6, nBuf 1`), so "never measured without MultiBD welded on" no longer holds. **The plan's ≈+15–17% projection is refuted — see §1.5.** What remains of this step is one α=0.3 arm and a Studio blind sheet at whatever α survives.

- **What remains**: one arm — `{ $0.dynamicTauAlpha = 0.3 }` against the existing `q-cached` control, 3 runs, gen-128, all suites, in one process (add it to the factorial arm list and re-run; the α=0.6 cells are already there). Then, **if and only if** an α survives on a suite, a Studio blind sheet at that α.
- **Why α=0.3 at all, given F-b**: it is nearly free and it is the only untested point between "off" and the α that demonstrably fails chat. Expect ~0 on chat `[Inferred]`; the live question is whether reasoning does better at 0.3 than the +13.8% already banked at 0.6, and whether code's marginal +3.5% moves.
- **Rationale**: fewer-steps levers multiply against the whole 28 ms forward and are the only uncapped lever class. Threshold-only change ⇒ per-forward cost unchanged.
- **Expected outcome** (revised from ≈+15–17%): **≈+14% on reasoning only** `[Sourced]` — already measured. Chat +0.2%, code +3.5%. The served shape is a **reasoning-only per-suite preset**, not a global default.
- **Also record**: **post-steps/block per arm** — F9 makes it the leading indicator of the corruption mechanism, and it is the cheapest check on whether the dev-host defect transfers to Studio trajectories. (Already visible: Studio chat post/blk 1.35 at α=0.6 vs the M1's 1.16 — *more* churn on the host where the lever also fails to save steps.)
- **Decides**: the per-suite α; the default-on question for `dynamicTauAlpha`. **A default-on at any α needs a Studio blind sheet at that α** — the dev sheet does not transfer verbatim (§1.4 caveat). LocalLeap's entry condition is now largely decided *for* it (§1.5).

### Step 2 — Composability sweep of the accepted winners (Studio, one process)

- **What**: after P2, one interleaved campaign over the open cells: JOT × dyn-τ, dyn-τ × Credit, JOT × Credit × dyn-τ. Baseline + each accepted single as controls. **The `S2D2 × dyn-τ` cell is dropped** — WP-2a's preset is withdrawn (§1.4), so the lever it would compose is no longer a shipping candidate.
- **Rationale**: three individually-accepted levers remain (JOT +21.9% reasoning, dyn-τ ~+15% projected but per-suite-limited, Credit +1.8% unsettled) and **the stack has never been measured** — the composability matrices mark exactly these cells open. dyn-τ (thresholds) and JOT (frozen-token compute skip) touch different mechanisms, so additivity is plausible but unproven `[Speculative]`.
- **Expected outcome**: if even two of the big levers are additive, a **+35–40% served preset** — the largest remaining wall-clock prize, from arms that already exist. Negative interactions are themselves a matrix-cell result to record.
- **Output**: per-suite served presets (Q/S × chat/reasoning/code) documented for the server — the Phase-3 §6 end state. **α is now per-suite, not global** (§1.4).
- **⚠️ Quality gate (new, non-optional — §1.4 F10)**: this step *ships presets*, and it was written as a pure wall-clock campaign. **No preset ships on step/TPS evidence alone.** The winning stack needs its own blind sheet before it becomes a default or a documented preset. The precedent is exactly this step's subject matter: `p3-combo` was accepted as a preset on step/wall-clock evidence while its quality gate was open, and it turned out to be the highest-churn (post/blk 1.90), least-scored configuration measured. Record **post-steps/block for every arm** — it is the cheap leading indicator (F9), and a stack that pushes it materially above the ~1.16 that already produced visible corruption should be treated as quality-suspect until scored.

### Step 3 — Decompose the ~21% remainder with the WP-6c in-situ method

- **What**: extend `ModuleAblation` with arms for norms, sampler/selection-set construction, and loop control (K-step flag readbacks); difference each **against the full forward** (§3b rule); sanity gate + control arm + behavioural assertion, as established.
- **Rationale**: second-largest budget bucket and the only one known purely by subtraction. The method is built and validated; marginal cost is arm definitions.
- **Expected outcome**: a trustworthy split of ~21% into norms / sampler / selection / loop. **This measurement decides two roadmap items for free**: selection-set fusion (phase-3 §4.3 — build only if the sampler+selection tail is ≳8–10%) and the **§13.1 raw-Metal escape-hatch gate** (loop overhead >10–15% ⇒ spec the ICB decode-loop port; otherwise close the hatch with data and record why). No implementation until the number exists.

### Step 4 — Case B: three cheap probes *before* any custom kernel

All three sharpen or shrink the ~7.5 ms question at near-zero cost; run them ahead of the Instruments profile.

- **4a. Newest mlx-swift on the Studio** (after P1): re-run the `q-cached` baseline + `gather_qmm` timing. ml-explore iterates on the quantized kernels; part of the 27% may already be fixed upstream — **upgraded from `[Speculative]` to `[Inferred]` by §1.5 F-d**: the 0.31.4→0.31.6 bump alone bought a uniform +6–8% wall-clock with byte-identical counters, i.e. pure kernel improvement, which is direct evidence that the upstream iteration is real and lands in our hot path. Writing a custom kernel against a stale MLX risks racing a shipped fix. **Note on the pin**: P1's `.exact("0.31.6")` does not block this — bump the pin deliberately, measure, then keep or revert it. The pin exists to stop *silent, per-host* drift, not deliberate version experiments; the one rule is that both hosts move together and the P4 version field records which version produced each row.
- **4b. `sortedIndices: true`**: the resolved `gatherQuantizedMM` exposes this flag (verified in the checkout, `Ops.swift:1473`). WP-6a already sorts indices adaptively — grep whether the decode path passes the flag when indices *are* sorted; if not, one arm measures the unsorted-path tax. `[Sourced]` that the parameter exists; `[Speculative]` that it's unwired.
- **4c. Dequant-cost isolation probes**: (i) **8-bit experts** — M8 tested 6-bit for *quality* only; nobody tested 8-bit for *speed*. 8-bit skips nibble unpacking; ~16 GB fits the Studio trivially. If 8-bit at 2× bytes is comparable or faster, the mystery cost is pinned on unpacking. (ii) **`mode: .mxfp4`** (same op, verified in `Ops.swift:1472`) — different dequant inner loop (shared FP8 scale, no affine bias), same bytes; needs a re-convert via the existing streaming converter. Both are *diagnostics first*, formats second (any format change would need its own M8-style quality gate).
- **Expected outcome**: either the prize shrinks (upstream fix / flag) or the diagnosis is handed over for free (unpacking vs scheduling), making the Step 5 kernel targeted instead of exploratory.

### Step 5 — Case B: Instruments profile → custom `MLXFast.metalKernel`

- **What**: handoff §7d as written — Xcode GUI debugger occupancy counters (not `xctrace`) on `gatherQuantizedMM` at T=32, asking the sized question *what is the ~7.5 ms doing if it is not moving bytes* (occupancy, register pressure, memory-level parallelism — **not** ALU time). Then route (b): a new inline-MSL kernel à la FlashBlock; **no mlx-swift fork**.
- **Rationale**: the only surviving cheaper-steps lever; 42.9% of the forward with a measured ~27%-of-forward inefficiency upper bound. This is where the Alpha-MoE *ethos* (persistent kernel, threadgroup residency, fused dequant-into-GEMM) legitimately lands `[Inferred]` — its specific mechanisms do not.
- **Expected outcome**: realistic win is some fraction of ~27% of the forward — **plan on +10–20% TPS if the profile finds a fixable cause; accept a recorded negative if it doesn't**. The 433 GB/s comparison rate is from a *different* kernel; 7.5 ms is a ceiling, not a promise `[Sourced caveat]`.
- **Effort**: the largest item in this plan by far. That is why it sits *after* Steps 1–4, whose combined expected value is larger and whose combined cost is smaller.

### Step 6 — Accuracy: TSCV upgrades (best accuracy lever, CPU-only)

- **What**: three independent upgrades to the already-default-on WP-4a mechanism: (i) dump per-prompt votes and paired-test the +6pp (McNemar, closing the master-list debt); (ii) sweep `t_start` — the α sweep is inert, the entire effect is the cutoff, so the cutoff is the only real hyperparameter and it has never been swept properly; (iii) generalize "semantic equivalence" beyond last-number matching (normalized-string clustering) so TSCV applies outside math.
- **Rationale**: the only accepted accuracy win (+6–8pp GSM8K) `[Sourced]`, zero forward-pass cost, and currently shipping with an arbitrary hyperparameter and an unproven significance claim.
- **Expected outcome**: (i) a defensible or retracted +6pp; (ii) possibly more accuracy from the parameter that actually matters; (iii) accuracy on non-math suites `[Speculative]` — currently zero coverage there.

### Step 7 — Accuracy hygiene: settle Credit Decoding and ICE

- **7a. Credit Decoding (WP-4d)**: it is **default-on at +1.8%, below the drift floor, never quality-gated — a standing violation of the roadmap's own §0 rule.** After P2, one interleaved campaign + blind quality sheet. Expected outcome: it either earns its default (resolves to a real +2–6%) or comes out of the default set. Either result also cleans up Step 2's arm list. The sparse `[1, A]` credit-matrix representation is a follow-on only if it stays in.
- **7b. ICE (WP-4b/4c)**: +4pp at p=0.481 on GSM8K-100 with a non-monotonic Nt grid is not evidence. Re-run at GSM8K-500 for power; the one composition worth adding is ICE × TSCV (matrix: "highly compatible"; both live on the single-block path). Expected outcome: a significant +Npp or an honest demotion of the preset.

### Step 8 — Conditional tail

- **LocalLeap** (Tier 3): run **only if** Step 1 accepts dyn-τ *and* a measured step-reduction gap remains on its lever, with a hypothesis for beating the incumbent. Otherwise closed by the Tier-3 entry rule.
- **FlashBlock (WP-3b)**: **park.** Ceiling 14.3%, the `.reuseCache` mechanism has never worked, and it currently GPU-hangs (leading hypothesis: OOB tile read; André's `safe_j` clamp didn't prevent it — disambiguation recipe in handoff §7a). Touch it only as code hygiene: either debug to non-landmine status or fence it off. Do not invest for performance.
- **MultiTF checkpoint**: engine-identical swap-in if a MultiTF-post-trained LLaDA2.1 ever ships; keep as a watch item, zero work now.

---

## 3. What is explicitly NOT in this plan (and why)

| item | reason |
|---|---|
| Alpha-MoE fusion as designed | refuted: 0.39% ceiling (WP-6f); not bandwidth-bound; Hopper-specific mechanisms |
| Fused-router megakernel, `argPartition` top-k | router is 1.7%, not 13.6% (WP-6e) |
| blockLength 64/128, MultiBD default-on, any "wider forward" | Width Rule, two independent refutations (WP-1b, WP-6d) |
| FP16 experts | 4-bit measured 1.50× faster (WP-6f) |
| Router reuse | −14.3%/−24.1% TPS (WP-6e) |
| d²Cache, FreeDave, Sparse-dLLM | Tier-3 entry conditions unmet (handoff §7c pricing) |
| Elastic-Cache revivals | zero-ceiling, architectural (WP-1a) |
| lm_head work | ~3%, bandwidth floor; M8 closed quantization question |
| WP-4b Guided Diffusion, WP-4e Plan Conditioning, WP-4f PRR training | excluded by André (AR model / training) |
| WP-4f rule-based fallback | dyn-τ by another name — already banked |
| Window-Diffusion | non-starter (phase-4 doc) |
| **WP-2a verified-conservative preset** (τ=0.95 + S2D2) | **withdrawn 2026-07-17** (§1.4, `wp2a-logbook` §8): ~25% slower than Q-mode and the quality edge it existed for does not appear on the blind sheet. S2D2 itself stays landed, tested, default-off. |

## 4. Reporting

Each step lands per the existing discipline: pre-registered hypothesis in the WP logbook → interleaved one-process bench with `envValid` + effective echoes (+ the new **toolchain/MLX-version field**, master list §1) → sanity gate + control arm on any attribution → wiki draft, positive or negative. Re-freeze the Studio baseline after Step 2 (presets change the accumulated stack) and again after Step 5 if the kernel lands.

**Quality sheets are on the critical path, not a formality** (added 2026-07-17 from §1.4): two of the three P5 sheets flipped a decision, and in both cases the thing they overturned had *already been accepted* on step/wall-clock evidence while the sheet was outstanding (`wp2a-logbook` §7, `wp2b-logbook` §7). The failure mode is structural, not incidental — a lever's speed evidence arrives first, the preset gets written against it, and the quality gate becomes a formality nobody expects to fail. Two rules follow:
1. **No preset or default ships while its own quality gate is open.** "Accepted pending blind scores" is not an acceptance; state it as pending in the verdict line.
2. **`checks.md` is not a quality gate** (F12). It passed every corrupted row. Scripted 4-gram/mask-leak heuristics detect degeneration, not local incoherence — cite them as *degeneration* checks only, never as "objective checks clean".

**Blind-sheet artefacts belong in git, not `scratch/`** (added 2026-07-17; **done** — André moved them to `blinds/`, tracked as of `4c0ca34`). The sheets were generated into the M1's `scratch/`, which is gitignored and therefore invisible from the Studio: they read as *lost* from this host until retrieved by hand, and a session concluded exactly that before checking `hw.model`. Scored sheets are the evidence for a shipped decision — they belong in git. **Still M1-only and gitignored: `draft_graph.json`** (the Spiffy ceiling, needed by §1.3) and `wp2b_driver.sh`; retrieve both from the M1 rather than re-running their calibrations.
