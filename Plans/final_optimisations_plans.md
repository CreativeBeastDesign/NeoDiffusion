# NeoDiffusion — Final Optimisations Plan

**Status**: draft
**Last updated**: 2026-07-15
**Author**: Claude (session 2026-07-15), reviewed by André: ☐ pending
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
| **P2. Run-major interleaving in `LLaDABench`** | Arm identity currently correlates with elapsed time; nothing near the noise floor (Credit's +1.8%, composability deltas) is resolvable without it. Small loop restructure (`for run { for arm }`). | ~hours |
| **P3. Bench fails loudly on arm/process-global `speculationK` conflict** | Handoff §7a TODO — prevents a repeat of the mid-run trap. | ~1 hour |
| **P4. Bench emits the toolchain + resolved-MLX-version field per JSONL row** | Master list §1 made this mandatory (2026-07-15): `Package.resolved` is not evidence of what linked; without the field a cross-host row cannot be interpreted. Must land **before** the P7 re-freeze so the new baseline rows carry it. | ~1 hour |
| **P5. André: blind sheets** `scratch/wp2b_blind/`, `scratch/wp2a_blind/`, `scratch/m8_blind/sheet.md` | Three default-on/preset decisions are quality-gated on these. | André only |

### 1.2 The P1 revalidation cascade (everything a new toolchain + MLX version invalidates)

Step counts and kernel timings are deterministic **per (host, MLX version)** — so changing the Studio's MLX version obsoletes every existing Studio row. In order, after P1:

| item | why | effort |
|---|---|---|
| **P6. Full test suite green on the Studio's new toolchain** (currently 118 tests) + `Tools/seed-metallib.sh` after every build; remember `swift test`/`xctest` can wipe the seeded metallib, and never partially delete `.build` | The toolchain itself changed; nothing downstream is trustworthy until this is green. | ~1 session |
| **P7. Re-freeze the Studio `q-cached` baseline** under the pinned MLX (all existing Studio rows are mlx-swift 0.31.4). Every Step 1–5 delta measures against this, not the old rows. | §1a: cross-version comparisons are confounded exactly like cross-host ones. | 1 bench run |
| **P8. Re-run `LLaDAMoEDispatchBench`** before trusting the `gatherQuantizedMM` dispatch choice | CLAUDE.md's explicit re-run trigger: "after any MLX/GPU/shape change". The ranking has never been validated on the Studio at all (M1-only, 0.31.6). | 1 bench run |
| **P9. Re-run the WP-6c `attr-full` control arm** (and, if it drifts >~2% from P7's baseline, the full attribution arms) | This plan prices every step against the 42.9/21/14.3% budget — measured under 0.31.4. If a new MLX shifts the budget, Steps 3–5's expected values shift with it. | 1–2 bench runs |
| **P10. Spot-check the WP-6a adaptive-dispatch crossovers** (T=128/192) | Default-on, dormant, and its crossover thresholds were measured under the old MLX; a changed sort or qmm kernel moves them. Cheap to confirm, embarrassing to have silently wrong. | 1 short run |

Note: re-benching the *accepted winners* (JOT, S2D2, dyn-τ, Credit) under the new baseline is **not** a separate item — Step 2's composability sweep includes each as a single-lever control arm, so the backfill and the sweep are one campaign.

### 1.3 Pre-existing Studio debts (independent of P1, still open from Phase 2)

| item | what it validates | notes |
|---|---|---|
| **P11. BF16 parity run (M4/M5 gate)** | Token-for-token match vs the reference (temp 0, ≥20 prompts, S/Q modes, eos on/off), prefix cache off → then identical with it on. **The correctness anchor has never run.** | Per roadmap §0.1(3): if it surfaces a forward-level discrepancy, scheduling verdicts survive but absolute quality numbers re-anchor. Do it before trusting Step 6/7 accuracy numbers too far. |
| **P12. M6 both-machines gate** | The frozen dev baseline's Studio twin, on the full M6 metric set. | Largely subsumed by P7 if run on the same suites — fold them into one session. |
| **P13. M8 scored-set comparison on the Studio** | The 4-bit-vs-BF16 quality comparison on the scored set (M1's was flip-rate only). | Needs BF16 inference ⇒ Studio-only by memory; pairs naturally with P11's fixture session. |

Backfill decisions that CLAUDE.md assigns to "the Studio" and where they land in this plan: **WP-2b default-on** → Step 1; **WP-2a verified-conservative preset** → Step 2 (+ P5 sheet); **WP-1b net-TPS gate/presets** → Step 2 (nBuf=2 arm); **Spiffy runtime** → can likely be closed *on paper*: WP-6d measured the Studio's wide-forward multiplier (1.63× at 2× width), and Spiffy needs (1+D)× width at D=3–8 — apply the Width-Rule break-even to the recorded `scratch/draft_graph.json` ceiling before spending any bench time `[Inferred]`.

---

## 2. Ordered next steps

Ordering principle: expected TPS/accuracy per unit effort, with measurements that *decide other steps* promoted above implementations they gate.

### Step 1 — Dyn-τ standalone on the Studio ⭐ first, ~1 line

- **What**: `LLaDAArm(name: "dyntau", …) { $0.dynamicTauAlpha = 0.6 }` vs `q-cached`, 3 runs, gen-128 (exact command in handoff §7b). Already implemented, landed default-off; never measured without MultiBD welded on.
- **Rationale**: fewer-steps levers multiply against the whole 28 ms forward and are the only uncapped lever class; every Studio winner so far lives there. Logical-step reduction already measured at −14.8% chat / −11.3% reasoning `[Sourced]`; threshold-only change ⇒ per-forward cost unchanged.
- **Expected outcome**: **≈ +15–17% TPS** `[Inferred]` — bigger than Case B's realistic win, at zero new code. Quality gate rides on P5 (`wp2b_blind/`).
- **Decides**: whether LocalLeap (its Tier-3 competitor for the same slot) ever runs; the default-on question for `dynamicTauAlpha`.

### Step 2 — Composability sweep of the accepted winners (Studio, one process)

- **What**: after P2, one interleaved campaign over the open cells: JOT × dyn-τ, dyn-τ × Credit, JOT × Credit × dyn-τ, and (where the `speculationK == 1` precondition allows — check before building the arm list `[Inferred]`) S2D2 × dyn-τ. Baseline + each accepted single as controls.
- **Rationale**: four individually-accepted levers exist (JOT +21.9% reasoning, S2D2 +14.6/+19.8%, dyn-τ ~+15% projected, Credit +1.8% unsettled) and **the stack has never been measured** — the composability matrices mark exactly these cells open. dyn-τ (thresholds) and JOT (frozen-token compute skip) touch different mechanisms, so additivity is plausible but unproven `[Speculative]`.
- **Expected outcome**: if even two of the three big levers are additive, a **+35–40% served preset** — the largest remaining wall-clock prize, from arms that already exist. Negative interactions are themselves a matrix-cell result to record.
- **Output**: per-suite served presets (Q/S × chat/reasoning/code) documented for the server — the Phase-3 §6 end state.

### Step 3 — Decompose the ~21% remainder with the WP-6c in-situ method

- **What**: extend `ModuleAblation` with arms for norms, sampler/selection-set construction, and loop control (K-step flag readbacks); difference each **against the full forward** (§3b rule); sanity gate + control arm + behavioural assertion, as established.
- **Rationale**: second-largest budget bucket and the only one known purely by subtraction. The method is built and validated; marginal cost is arm definitions.
- **Expected outcome**: a trustworthy split of ~21% into norms / sampler / selection / loop. **This measurement decides two roadmap items for free**: selection-set fusion (phase-3 §4.3 — build only if the sampler+selection tail is ≳8–10%) and the **§13.1 raw-Metal escape-hatch gate** (loop overhead >10–15% ⇒ spec the ICB decode-loop port; otherwise close the hatch with data and record why). No implementation until the number exists.

### Step 4 — Case B: three cheap probes *before* any custom kernel

All three sharpen or shrink the ~7.5 ms question at near-zero cost; run them ahead of the Instruments profile.

- **4a. Newest mlx-swift on the Studio** (after P1): re-run the `q-cached` baseline + `gather_qmm` timing. ml-explore iterates on the quantized kernels; part of the 27% may already be fixed upstream `[Speculative]`. Writing a custom kernel against 0.31.4 risks racing a shipped fix.
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

## 4. Reporting

Each step lands per the existing discipline: pre-registered hypothesis in the WP logbook → interleaved one-process bench with `envValid` + effective echoes (+ the new **toolchain/MLX-version field**, master list §1) → sanity gate + control arm on any attribution → wiki draft, positive or negative. Re-freeze the Studio baseline after Step 2 (presets change the accumulated stack) and again after Step 5 if the kernel lands.
