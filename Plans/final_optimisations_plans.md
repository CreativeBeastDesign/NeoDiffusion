# NeoDiffusion — Final Optimisations Plan

**Status**: approved, in progress
**Last updated**: 2026-07-17
**Author**: Claude (session 2026-07-15), reviewed by André: ☑ approved 2026-07-17 ("the plan would be good to go"), with the P5 amendments folded in (§1.4)
**Progress (2026-07-17)**: **P1–P9 DONE except P10.** P1 (`.exact` pin, both hosts), P2/P3 (run-major interleaving + speculationK pre-flight), P4 (per-row `toolchain` provenance), P5 (§1.4 — two sheets flipped a decision), P6 (suite green: 120 exec, 0 fail), **P7 + Step 1 + P9** (one interleaved Studio session, §1.5 F-f/F-g: baseline re-frozen, α sweep done, attribution stable), **P8** (dispatch ranking holds on Studio). **Spiffy CLOSED on paper** (`wp2a` §9). Remaining prereq: **P10** (WP-6a crossover spot-check, cheap). Then: **the Studio blind sheet** (reasoning α0.3/α0.6 + code α0.3 — André approved) gates any dyn-τ default; **Step 2** composability sweep is the next real prize.
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
| ~~**P7. Re-freeze the Studio `q-cached` baseline**~~ **DONE 2026-07-17** (in the Run A process, §1.5 F-f): q-cached reproduces `factorial_0316` within noise (chat +2.6%, reasoning +0.2%, code +0.1%), variance gate PASS, 0.31.6 stamped. The frozen baseline holds; no re-anchoring. | §1a: cross-version comparisons are confounded exactly like cross-host ones. **De-risked by §1.5 F-d**: the 0.31.4→0.31.6 bump left counters byte-identical and moved wall-clock a near-uniform +6–8%. | done |
| ~~**P8. Re-run `LLaDAMoEDispatchBench`**~~ **DONE 2026-07-17** (`NEODIFFUSION_M6_BENCH=1 swift test --filter LLaDAMoEDispatchBench`, all cases pass; `scratch/p8_dispatch.log`). **`gatherQuantizedMM` is confirmed fastest at production T=32 on the Studio/0.31.6** — 1.2 ms vs dense 5.3 ms (4.4×; occupancy 296 vs 108 GB/s), advantage widening with T. Two Step-4 data points fall out: **(4b)** sorted indices ≈2× faster than unsorted (6.90 vs 13.24 ms) — WP-6a's adaptive sort pays; **(4c)** dequantized+gatherMM is **5× *slower*** than the quantized path (0.0059 vs 0.0012 s/op) ⇒ pre-dequant does not help and the gather is not naively dequant-bound (consistent with WP-6f's coalescing probe). The M1-only ranking now holds on the Studio. | CLAUDE.md's explicit re-run trigger: "after any MLX/GPU/shape change". **§1.5 F-e**: read `mlxCoreVersion` (0.31.1), not the package pin — the dispatch kernels live in the core. | done |
| ~~**P9. Re-run the WP-6c attribution arms**~~ **DONE 2026-07-17** (`scratch/p9_attribution.jsonl`, 4-arm canonical set, control `attr-full` 26.23 ms reproduces Run A's q-cached 26.49 ms at −1.0% ⇒ harness unchanged; sanity gate PASS, all-positive, sum 100.0%). **The budget is stable across the MLX bump** (vs the recorded 0.31.4 `attribution.jsonl`, identical arithmetic): routed-MoE path 56.2%→57.7%, attention 14.0%→12.8%, shared 4.4%→4.3%, remainder 25.4%→25.2% — all ≤1.5pp. **But the bump was NOT uniform (F-g)**: absolute per-forward fell 27.5→26.2 ms (−4.9%), and that came from **attention (−13%)**, not the MoE GEMM (routed path −2.5%, within noise). So Steps 3–5's pricing survives, and Step 5's Case B premise is *reinforced*: gather_qmm was not the thing 0.31.6 fixed. | This plan prices every step against the 42.9/21/14.3% budget — measured under 0.31.4. | done |
| ~~**P10. Spot-check the WP-6a adaptive-dispatch crossovers**~~ **DONE 2026-07-17** (fell out of P8's `testSortingOverheadSweep`; `scratch/p8_dispatch.log`). **The Mini (H=2048, our shape) crossover MOVED 192→~128 across the bump** — exactly the "silently wrong" risk this item flagged: at T=128 sorting was a 0.972× *regression* under 0.31.4 and is now a **1.070× win** under 0.31.6 (T=96 still 0.747× loss, so the break-even is ~(96,128]). Flash (H=4096) crossover stays ≈128. **The engine hardcodes `flat.dim(1) >= 4096 ? 128 : 192`** (`LLaDA2MoE.swift:404`), so for Mini it now bypasses sorting on T∈[128,192) where sorting would win ~1.07–2.25×. **Harmless at the current blockLength=32 serving config** (decode T=32, where sorting still loses 0.605× — bypass is correct), so this is dormant, not broken. Caveat: the sweep uses *uniform-random* indices (same as the original measurement — valid for detecting the shift, not for the real-routing absolute crossover). See F-h and the open decision below. | Default-on, dormant, and its crossover thresholds were measured under the old MLX; a changed sort or qmm kernel moves them. Cheap to confirm, embarrassing to have silently wrong. | done |

Note: re-benching the *accepted winners* (JOT, S2D2, dyn-τ, Credit) under the new baseline is **not** a separate item — Step 2's composability sweep includes each as a single-lever control arm, so the backfill and the sweep are one campaign.

### 1.3 Pre-existing Studio debts (independent of P1, still open from Phase 2)

| item | what it validates | notes |
|---|---|---|
| **P11. BF16 parity run (M4/M5 gate)** | Token-for-token match vs the reference (temp 0, ≥20 prompts, S/Q modes, eos on/off), prefix cache off → then identical with it on. **The correctness anchor has never run.** | Per roadmap §0.1(3): if it surfaces a forward-level discrepancy, scheduling verdicts survive but absolute quality numbers re-anchor. Do it before trusting Step 6/7 accuracy numbers too far. |
| **P12. M6 both-machines gate** | The frozen dev baseline's Studio twin, on the full M6 metric set. | Largely subsumed by P7 if run on the same suites — fold them into one session. |
| **P13. M8 scored-set comparison on the Studio** | The 4-bit-vs-BF16 quality comparison on the scored set (M1's was flip-rate only). | Needs BF16 inference ⇒ Studio-only by memory; pairs naturally with P11's fixture session. |

Backfill decisions that CLAUDE.md assigns to "the Studio" and where they land in this plan: **WP-2b default-on** → Step 1 (now a 3-arm α sweep, §1.4); **WP-2a verified-conservative preset** → **CLOSED negative 2026-07-17** (P5 sheet; `wp2a-logbook.md` §8 — preset withdrawn, no Step-2 arm); **WP-1b net-TPS gate/presets** → Step 2 (nBuf=2 arm, now quality-gated per §1.4); **Spiffy runtime** → **CLOSED on paper 2026-07-17** (`wp2a-logbook.md` §9, F9): applying the Width-Rule break-even to the 1.63×-at-2× multiplier collapses F7's 27–35% forward-count ceiling to a **≤+13% wall-clock upper bound** (net ≤ 0.37·s, D-independent), and that bound assumes both full draft acceptance (contradicted by F7's 35–40% token-miss) and linear width scaling to 9×. Below JOT's landed +21.9% for a full drafting-runtime build ⇒ not worth it. Reopenable only by directly timing 4×/9×-width forwards on the Studio (minutes, no runtime) if the linear-width assumption is ever worth removing. `draft_graph.json` retrieved to `blinds/`.

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

**F-f — the α=0.3 arm is now run (2026-07-17, `wp2b-logbook` §9, F15–F16); it overturns the "chat α=0.3" plan and promotes code.** `[Sourced]` Interleaved Studio run (`scratch/refreeze_step1.jsonl`, variance PASS, 107/108 envValid), Δ vs the same-process q-cached control: **chat** α0.3 +0.5% / α0.6 −0.1% (dead at both, as F-b predicted — chat closed); **reasoning** α0.3 +9.3% / α0.6 +14.1% (α0.6 wins); **code** α0.3 **+11.2%** / α0.6 +3.6% (α0.3 wins by 3×). The per-suite optimum is *not* a single α: reasoning wants 0.6, code wants 0.3, and α0.3 is the lower-churn point everywhere (post/blk 1.10–1.18 vs 1.35–1.70). Code — dismissed as "not worth a sheet" in §1.4 — is now the second-best cell in the sweep and a live preset candidate. This run also **re-froze the P7 baseline** (q-cached reproduces `factorial_0316` within noise: chat +2.6%, reasoning/code +0.1–0.2%), so P7 is largely discharged too.

**F-g — the 0.31.6 speedup came from attention, not the MoE GEMM; Case B's premise is reinforced, and Step 4a's "maybe upstream already fixed it" hope is refuted for gather_qmm.** `[Sourced]` P9 attribution (`scratch/p9_attribution.jsonl`, control-validated, sanity gate PASS) vs the recorded 0.31.4 `attribution.jsonl`, identical arithmetic: absolute per-forward fell 27.5→26.2 ms (−4.9%), but the routed-MoE path (which contains gather_qmm) barely moved (15.5→15.1 ms, −2.5%, within noise) while **attention dropped −13%** (3.9→3.36 ms). Shares are stable (all ≤1.5pp) *because* everything got a little faster and attention got a lot faster. Implication for Step 5: the ~7.5 ms / 27%-of-forward gather_qmm inefficiency is **not** the thing 0.31.6 improved, so the custom-kernel premise stands unrefuted by the bump — the cheapest of Step 4's probes (4a) is now answered *negative-for-closing-Case-B*, i.e. Case B stays open.

**F-h — the WP-6a Mini crossover drifted 192→~128 under the bump; dormant now, but the hardcoded constant is stale (P10).** `[Sourced]` `LLaDA2MoE.swift:404` picks the sort/no-sort break-even as `flat.dim(1) >= 4096 ? 128 : 192`; the re-measured Mini (H=2048) break-even is now ~(96,128] (T=128 flipped from a 0.972× regression to a 1.070× win). At the served blockLength=32 (T=32) sorting still loses (0.605×), so the decode path is correct and this is dormant — but for any T∈[128,192) (larger blockLength, batch, or speculative widths) the engine now bypasses a sort that would win 1.07–2.25×. **FIXED 2026-07-17**: `LLaDA2MoE.swift` crossover collapsed to a constant `128` (both shapes now break even there); parity suite green (bypass path intact) + a real-weight smoke at T≈168 (short prompt, `--block-length 128`) confirms the newly-switched sorted path yields coherent output. Same-decomposition caveat retained in the code comment: uniform-random indices, valid for the *shift*, not the real-routing absolute crossover.

**Consequences**: Step 1 is **done** (both α arms measured, baseline re-frozen). P7, P9, **and P10** discharged across the Run A/B/P8 session — the entire prerequisite thread P1–P13 is now closed except the pre-existing Phase-2 Studio debts (P11 BF16 parity, P12/P13). Step 8's LocalLeap entry condition is *materially more open* than the plan assumed — dyn-τ, its Tier-3 competitor for the same slot, now demonstrably leaves chat and code unimproved on the Studio, so "a measured step-reduction gap remains on its lever" is satisfied for 2 of 3 suites. Step 2's dyn-τ arms should be **reasoning-only** unless the α=0.3 arm surprises.

## 2. Ordered next steps

Ordering principle: expected TPS/accuracy per unit effort, with measurements that *decide other steps* promoted above implementations they gate.

### Step 1 — Dyn-τ standalone on the Studio → **DONE 2026-07-17 (§1.5 F-f, `wp2b-logbook` §9). Result: per-suite, α-dependent; NOT the projected global +15–17%.**

Both α arms are now measured on the Studio, interleaved with the control in one process (P2), plus the P7 baseline re-freeze in the same run:

| suite | α=0.3 | α=0.6 | verdict |
|---|---|---|---|
| chat | +0.5% | −0.1% | **dead at both — closed** |
| reasoning | +9.3% (post/blk 1.10) | **+14.1%** (1.70) | α-vs-churn trade → sheet |
| code | **+11.2%** (1.10) | +3.6% (1.65) | **α=0.3 wins 3×; newly a candidate** |

- **CLOSED 2026-07-17 — the Studio blind sheet is scored (`wp2b-logbook` §10, F17): α=0.3 for reasoning and code, α=0.6 rejected everywhere.** Reasoning flipped from the speed winner (α=0.6, +14.1%) to the quality winner (**α=0.3, +9.3%**): α=0.6 corrupted 1/4 reasoning prompts, confirming F16's churn read (post/blk 1.70). Code **α=0.3 (+11.2%)** is quality-neutral vs baseline. Chat closed (dead on TPS). The dev sheet's "reasoning α=0.6 clean" did **not** transfer (F18 → F14).
- **Served shape**: `dynamicTauAlpha = 0.3` as a **per-suite preset for reasoning + code**, global default **off** (chat unscored + only +0.5% TPS ⇒ no case to force globally). Wire where the server can select by request class — the §6 preset end-state.
- **Overturned assumptions**: "α=0.3 for chat" dead (chat dead at every α); code, dismissed in §1.4, is the surprise winner; the lower-churn α=0.3 wins outright, not the faster α=0.6. LocalLeap's Tier-3 entry (Step 8) stays open — dyn-τ leaves chat unimproved and code/reasoning ceilings are modest.

### Step 2 — Composability sweep → DONE 2026-07-17. **NEGATIVE: no stack beats dyn-τ α=0.3 alone.** Two high-value side findings.

**Run** (`scratch/step2_compos.jsonl`, one interleaved K=1 process — JOT-faithful forces K=1, P3 enforced it; 8 arms, 3 runs × chat/reasoning/code × gen-128, variance PASS, 288/288 envValid). Arms: `q-cached` + singles (`lf-jot`, `dt-tau-a03`, `lf-credit-preset`) + pairs/triple (`cmp-jot-dt`, `cmp-credit-dt`, `lf-jotcredit`, `cmp-jot-credit-dt`). dyn-τ at the Step-1-validated **α=0.3**.

**Best config per suite (TPS): `dt-tau-a03` wins outright on all three** — chat +9.3%, reasoning +12.3%, code +13.7% vs the K=1 baseline. Every pair and the triple is either net-negative or *below dyn-τ alone*: reasoning `dt-tau`(104) > `cmp-jot-dt`(103) > `cmp-credit-dt`(99); code `dt-tau`(138) > everything else (next best 114). **The +35–40% stacked-preset hypothesis is refuted — the levers do not stack.**

**F-i — JOT and Credit do not carry their weight on the Studio; the "stack" is dead.** `[Sourced]` JOT (faithful, its only legal mode) is **net-negative on chat (−26%) and code (−20%)**, +5.2% reasoning only; Credit is **0-to-negative everywhere** (chat +0.0%, reasoning −3.0%, code −9.3%). Combining them with dyn-τ interferes (reasoning: triple −0.6% vs dyn-τ alone +12.3%) — the "synergy" labels the analysis prints are synergy *toward less-negative*, never a net win. Reproduces the drift-free `leverfresh` signs (JOT code −20% ≈ recorded −23%; jot-credit code −16% ≈ recorded −17%). **⚠️ Credit-params caveat**: these Credit arms use `lf-credit-preset` (α=1.0/γ=1.0) — the master-list-flagged **doc-error config that was never shipped** (git 7bb761f: shipped is α=0.5/γ=0.5). The shipped config is chat-marginal-positive, not −11%; but even shipped it is reasoning −7% / code −3% (master list), so the "Credit doesn't help the stack" conclusion stands. Closed properly in Step 7a at shipped params + the K=1 baseline.

**F-j — P2 immediately earned its keep: JOT's "+21.9–28% reasoning" is a cold-baseline artefact; drift-free it is +5.2%.** `[Sourced]` JOT's reasoning TPS is stable (~97.3 here, 97.4 recorded) — it is the *baseline* that moved: the record's +21.9–28% compared JOT to an under-warmed q-cached (76.3 TPS); the P2-interleaved (both-arms-warm) baseline is 92.4, giving **+5.2%**. This is exactly the sub-noise-floor artefact P2 was built to remove ("nothing near the noise floor is resolvable without it"). **This contradicts a `[Sourced]` record — André's call (below); recommend a focused confirm run.** JOT's status drops from "biggest accepted lever" to "marginal reasoning-only, negative elsewhere."

**F-k — the K=4 speculation default is costing ~16–17% on reasoning and code (the highest-value lead in the sweep).** `[Sourced, cross-process]` q-cached@K=1 beats q-cached@K=4 (Run A): reasoning +16.0% (92.4 vs 79.7), code +17.0% (121.3 vs 103.7), chat +5.4%. This **reproduces and extends** the master-list's "+19% code" lead — the served `speculationK=4` default appears net-negative on reasoning/code. Caveat: K is process-global (cannot interleave K=1 vs K=4 in one process, P3), so this is cross-process and carries session/thermal drift; it needs a controlled back-to-back settle before acting. **If it holds, suite-aware K (K=1 for reasoning/code) is a bigger, simpler win than anything in Steps 2–5.**

**Served outcome**: **dyn-τ α=0.3 per-suite preset (reasoning + code) stands as the Step-1 result; no stacked preset is added.** No quality sheet needed here — nothing new ships (every combo lost). Credit's neutral-negative showing feeds Step 7a; JOT feeds F-j's confirm decision.

#### F-j and F-k CONFIRMED + decomposed (2026-07-17) — the biggest actionable win of the session is quality-free

**F-j confirmed** (`scratch/fj_jot_confirm.jsonl`, q-cached vs lf-jot, K=1, 5 runs interleaved, variance PASS): **JOT reasoning +5.5%** (chat −26.1%, code −19.8%), matching the Step-2 +5.2%. The recorded +21.9–28% was a cold-baseline artefact — JOT's own TPS is stable; the record's baseline was under-warmed.

**F-k SETTLED** (`scratch/fk_speck.jsonl`, spk-1 vs spk-4 in **one** process via the new per-arm-K override — no cross-process drift; 5 runs, variance PASS, 120/120 envValid): **K=1 beats the served K=4 on every suite — chat +6.4%, reasoning +16.2%, code +17.1% — with byte-IDENTICAL output (11/11 prompts, 0 differ).** Speculation K is output-invariant by construction (only sync scheduling changes), verified here on real weights. So this is a **quality-free speedup: no blind sheet is possible or needed.**

**The decomposition ties them together.** The recorded "JOT +21.9% reasoning vs the K=4 default" = the K=1 switch (×1.162) × JOT-on-K=1 (×1.055) ≈ **+22.6%** — arithmetically right but **misattributed**: ~16% is the free K=1 switch that JOT-faithful *forces*, only ~5.5% is JOT itself, and JOT is negative on chat/code. You get the large half for free, without JOT.

**F-l — the loop-speculation default (K=4) is net-negative on the M2 Ultra; K=1 is strictly better everywhere, quality-free.** `[Sourced]` K>1 speculative loop execution (Phase-2 gotcha 9) trades extra speculative forwards for fewer CPU-GPU syncs; on the compute-bound Studio the syncs are cheap and the speculative overhead dominates, so K=4 loses on all three suites. K=1 also uses less memory (no run-ahead) and produces identical tokens. The K=4 default is an M1-era choice (where syncs are expensive) that does not fit the Studio. **This is the single largest quality-free lever found — +6–17% by changing one served constant — and it partially un-does the loop-speculation accept.** Open decision below (serving-default change ⇒ André's call). Cross-check: reproduces and now *settles* (single-process) the master-list's "+19% code" lead.

### Step 3 — Decompose the ~21% remainder → STARTED 2026-07-17; **forward-level ablation hit a wall. The method cannot decompose this bucket; it needs engine-level per-phase timers.**

- **What was tried**: a `ModuleAblation.layerNorms` arm (skip the two per-layer RMSNorms) to split the norms term out of the remainder, per the plan. Control validated (attr-full 26.25 ms ≈ P9's 26.23; attention 12.8% / MoE 61.8% reproduced).
- **F-m — norms cannot be attributed by forward-level ablation on real weights (recorded negative).** `[Sourced]` The first wiring landed in the JOT decoder overload, not the served-default one — the `testAblationActuallyBitesOnDefaultPath` guard caught it (norms output identical to baseline; the *exact* class of silent-null the guard exists for, and the ms/forward read of ~0 would have shipped it as "norms are free"). After fixing it into the served overload the guard passes on toy fixtures — but on the **4-bit model the arm HANGS**: RMSNorm is load-bearing for numerical stability, so 20 unnormalized layers compound to inf/NaN and the loop can never clear a mask (killed after 64 min, no block progress). So the two "cheaper-steps" resolvable components are the numerically-robust ones (attn, MoE); norms is not ablatable, and **sampler/selection/loop live outside the forward entirely** — ms/forward ablation was never going to reach them.
- **What this decides**: the ~21% remainder split (and therefore selection-set fusion at ≳8–10% and the §13.1 escape-hatch at loop >10–15%) **requires engine-level per-phase instrumentation** — wall-clock timers around the sampler, the Γ/Δ selection-set construction, and the K-step loop-control readbacks in `DiffusionEngine`'s denoising loop — **not** more `ModuleAblation` arms. That is a scoped build (a per-phase timing struct on `Metrics`, gated behind the existing instrument flag), not done this session. **Recommendation: continue Step 3 as engine-phase timers; the forward-level path is closed.** The `.layerNorms` case + its guard are kept as the recorded negative (no bench arm — it would hang a session).
- Kept for reference (the original intent): *build only if the sampler+selection tail is ≳8–10%; escape-hatch if loop >10–15%.* Those thresholds now await the engine-phase measurement.

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

- **7a. Credit Decoding (WP-4d) → LARGELY CLOSED 2026-07-17.** The premise was wrong: Credit is **default-OFF**, not on — verified in both the engine (`GenerationParams.creditDecodingEnabled = false`) and the server (no override in `diffusion-server`). So there is no "standing §0 violation" and nothing ships on it. Its "+1.8%" is a chat *TPS* figure at shipped params (α=0.5/γ=0.5, `lf-credit-default`); the master list already resolved it drift-free (chat +1%, reasoning −7%, code −3% — chat-marginal, not a global win). Fresh shipped-params run at the K=1 baseline (`scratch/step7a_credit.jsonl`, 0.5/0.9/0.5 confirmed, variance PASS, 72/72 envValid): **chat +3.5%, reasoning −1.0%, code −5.6%** — reproduces the drift-free sign (chat-only marginal, negative on reasoning/code), and it changes output on 8/12 prompts (logit boost ⇒ quality-relevant, never sheeted). **Recommendation: leave Credit default-off; retire it from the accepted-lever list** — chat +3.5% for a quality-relevant, never-gated lever, negative on the other two suites, is not worth shipping. (The `[1, A]` sparse representation is moot while it's off.) 7a CLOSED.
- **7b. ICE (WP-4b/4c) → does not fit this session.** Needs GSM8K-500 for power; only `gsm8k_100` and `gsm8k_val` exist as checked-in suites. Generating a 500-prompt suite (via `Tools/download_gsm8k.py`, network + setup) is a prerequisite, not a quick run. Deferred with the entry condition unchanged: re-run at GSM8K-500, add the ICE × TSCV composition (both on the single-block path). Expected outcome: a significant +Npp or an honest demotion of the preset.

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
