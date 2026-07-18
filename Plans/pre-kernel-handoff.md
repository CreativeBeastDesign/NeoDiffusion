# Pre-kernel hand-off — the road from here to the custom MoE kernel (Steps 4 → 5)

**Status**: Steps 1–3 + 7a CLOSED; **Step 4 (three cheap probes) is the last gate before the kernel**; Step 5 is the custom `gatherQuantizedMM` kernel — the payoff, and where this hand-off ends.
**Created**: 2026-07-18. **Owns**: the transition from the attribution campaign into kernel authoring.
**Prereq reading**: `Plans/final_optimisations_plans.md` §2 (plan of record) · `Plans/gather_qmm_handoff.md` (Case B / kernel method) · `Plans/experiments-master-list.md` (all verdicts) · this session's `Plans/step3-handoff.md` + `Plans/wiki-drafts/step3-engine-phase-decomposition.md` (Step 3 close).

---

## 1. Where we are, in one paragraph

The 2026-07-14→18 attribution campaign has closed every *cheaper-steps* and *cheaper-loop* lever with data. **The only surviving performance lever is Case B: the 4-bit MoE `gatherQuantizedMM` inner loop, where ~7.5 ms (~27% of the forward) is measurably *not* moving bytes** (WP-6f). Everything upstream of a custom kernel is now either done or is a **cheap diagnostic that sharpens/shrinks that ~7.5 ms question before a single line of MSL is written** (Step 4). Step 3 was the last thing that could have opened a *different* front (engine-loop fusion / raw-Metal loop port) — it closed **negative** (the engine loop is ~5% of the step; forward is 94.8%), so the kernel is unambiguously the next and only performance target. Do Step 4 first: it is near-zero cost and makes Step 5 **targeted instead of exploratory**.

## 2. What is DONE (the road so far)

| item | verdict | one-line result |
|---|---|---|
| **P1** toolchain/MLX pin | DONE | both hosts Swift 6.3.3 + mlx-swift `.exact("0.31.6")` (MLX core 0.31.1). Never a `from:` range. Metallib via Xcode 26 + `Tools/seed-metallib.sh`. |
| **P5** blind quality sheets | DONE | 3 sheets scored; flipped 2 decisions. Standing rules: (1) no preset ships with an open quality gate; (2) `checks.md` heuristics are *degeneration-only* — they pass on locally-incoherent text. |
| **Step 1** dyn-τ standalone (Studio) | DONE | per-suite, α-dependent — **reasoning-only** (+13.8% TPS / −13.6% steps @ α=0.6; α=0.3 the safe point). Chat is dead (step-saving sign-reverses across hosts + visible corruption). NOT the projected global +15–17%. |
| **Step 2** composability sweep | DONE — NEGATIVE | no stack beats **dyn-τ α=0.3 alone** (chat +9.3 / reasoning +12.3 / code +13.7% vs K=1 base). JOT + Credit don't carry weight. Two side findings ↓. |
| **Step 2 side-finding F-k/F-l** | SETTLED | **K=1 beats the served K=4 loop-speculation default on every suite (+6.4/+16.2/+17.1%), byte-identical output.** Quality-free. Already shipped as **host-aware default** (server picks K=1 on ≥64 GB hosts — CLAUDE.md F-l). |
| **Step 2 side-finding F-j** | CONFIRMED | JOT's recorded "+21.9–28% reasoning" was a **cold-baseline artefact**; drift-free it is **+5.5% reasoning only** (−26% chat, −20% code). ~16% of the old number was the free K=1 switch JOT-faithful forces. |
| **Step 3** forward-remainder decomposition (**this session**, F-n) | CLOSED — NEGATIVE | engine-phase timers measured: forward **94.8%**, sampler 1.6%, selection 1.3%, loop-control 2.0%. **Selection-set fusion CLOSED** (2.9% < 8–10%); **§13.1 raw-Metal escape-hatch CLOSED with data** (2.0% < 10–15%). Committed on branch `step3-engine-phase-timers`. |
| **Step 7a** Credit Decoding | LARGELY CLOSED | premise was wrong — Credit is **default-OFF** in engine + server, nothing ships on it. At shipped params: chat +3.5%, reasoning −1.0%, code −5.6%, changes output on 8/12 prompts (never sheeted). **Recommendation: leave off, retire from the accepted-lever list.** |

**Accuracy levers already landed (not part of the kernel road, listed to answer "haven't we done these?"):** TSCV (WP-4a, default-on), ICE (WP-4b, served preset), Credit (WP-4d, default-off). Their *open work* is refinement/hygiene (Steps 6/7), not re-doing them.

## 3. The last gate before the kernel — Step 4: three cheap probes

All three sharpen or shrink the ~7.5 ms at near-zero cost. **Run them before the Instruments profile** — each can either shrink the prize or hand over the diagnosis for free.

- **4a — bump mlx-swift on the Studio, re-time.** Re-run `q-cached` baseline + `gatherQuantizedMM` timing on the newest mlx-swift. ml-explore actively iterates the quantized kernels; part of the 27% may already be fixed upstream — this is `[Inferred]`, not speculative: the 0.31.4→0.31.6 bump alone bought a uniform **+6–8%** wall-clock with byte-identical counters (pure kernel improvement). Writing a kernel against a stale MLX risks racing a shipped fix. **The P1 `.exact("0.31.6")` pin does NOT block this** — bump it *deliberately*, measure, keep or revert. The pin stops *silent per-host* drift; the rules are (1) both hosts move together, (2) the P4 toolchain/MLX-version field records which version produced each row.
- **4b — `sortedIndices: true`.** The resolved `gatherQuantizedMM` exposes the flag (`.build/checkouts/mlx-swift/Source/MLX/Ops.swift:1380`/`:1411`, `sortedIndices: Bool = false`). WP-6a already sorts indices adaptively at decode — **grep whether the decode forward passes the flag when indices *are* sorted**; if not, one arm measures the unsorted-path tax. `[Sourced]` the parameter exists; `[Speculative]` that it's unwired.
- **4c — dequant-cost isolation (diagnostics first, formats second).**
  - **(i) 8-bit experts** — M8 tested 6-bit for *quality* only; nobody tested 8-bit for *speed*. 8-bit skips nibble unpacking; ~16 GB fits the Studio trivially. If 8-bit at 2× bytes is ≈ or faster than 4-bit, **the mystery cost is unpacking**, not scheduling. Convert via `Tools/convert_weights_streaming.py`.
  - **(ii) `mode: .mxfp4`** — same op, different dequant inner loop (shared FP8 scale, no affine bias), same bytes (`Ops.swift` ~`:1138`, the `.affine`/`.mxfp4` mode arg). Needs a re-convert. Any format change would need its own **M8-style quality gate** before shipping — here it's purely a diagnostic.
- **Existing probe hooks in the bench** (don't rebuild these): `.moeFixedExperts` ablation (`Tools/diffusion-bench/Sources/LLaDABench.swift:643`), `--dequantize-experts` FP16 path (`:389`, `ExpertDequantization.dequantizeRoutedExperts` `:700`).
- **Expected outcome**: either the prize shrinks (upstream fix / a flag) or the diagnosis is handed over free (unpacking vs scheduling) — Step 5 becomes targeted.

## 4. The destination — Step 5: Instruments profile → custom `MLXFast.metalKernel`

This is where the hand-off ends and the fun begins. Entry state is "Step 4 done, ~7.5 ms question as sharp as free diagnostics can make it."

- **Profile first, the *sized* question.** Xcode **GUI** Metal debugger occupancy counters (**not** `xctrace`) on `gatherQuantizedMM` at **T=32**, asking: *what is the ~7.5 ms doing if it is not moving bytes?* Measure **occupancy, register pressure, memory-level parallelism — NOT ALU time.** Context in `gather_qmm_handoff.md` §3 (Case B open problem) + §2.3: at T=32 the kernel already hits **290.5 GB/s = 36.3% of peak**, so it is **not occupancy-bound** — the cause is subtler (leading `[Inferred]` hypothesis: register pressure from interleaved dequant+FMA capping memory-level parallelism).
- **Then route (b): a new inline-MSL kernel via `MLXFast.metalKernel`**, à la FlashBlock — Metal source as inline Swift strings compiled by MLX at runtime. **NO mlx-swift fork**: a fork needs an Xcode DerivedData rebuild of mlx-swift + reseed on *every edit* (`gather_qmm_handoff.md` ~§lines 205–235). The Alpha-MoE *ethos* (persistent kernel, threadgroup residency, fused dequant-into-GEMM) legitimately lands here `[Inferred]`; its Hopper-specific mechanisms do not (they're refuted — §3 of the plan).
- **Prize / expected outcome**: some fraction of ~27% of the forward. **Plan on +10–20% TPS if the profile finds a fixable cause; accept a recorded negative if it doesn't.** The 433 GB/s comparison rate is from a *different* kernel — **7.5 ms is a ceiling, not a promise** `[Sourced caveat]`.
- **Effort**: the largest item in the plan by far (~8–9/10 vs Step 3's ~5). Moving-target MLX + inline-MSL authoring + occupancy profiling. This is why it sits last.

## 5. Parallel, NON-gating work (does not block or touch the kernel)

- **Step 6 — TSCV upgrades (CPU-only, the best accuracy lever).** (i) dump per-prompt votes + McNemar the +6pp (close the significance debt); (ii) sweep `t_start` — the α sweep is inert, the cutoff is the only real hyperparameter and was never swept; (iii) generalize "semantic equivalence" beyond last-number matching so TSCV works off-math. Zero forward-pass cost.
- **Step 7b — ICE.** Deferred: needs a **GSM8K-500** suite for statistical power (only `gsm8k_100`/`gsm8k_val` are checked in). Generate via `Tools/download_gsm8k.py` (network + setup), then re-run + add ICE×TSCV composition. Entry condition unchanged.
- **Step 8 — conditional tail.** **LocalLeap**: run only if a step-reduction gap remains on dyn-τ's lever *with a hypothesis for beating the incumbent* — Step 1 gave reasoning-only dyn-τ, so this is at most a narrow reasoning probe, else closed by the Tier-3 rule. **FlashBlock**: park (ceiling 14.3%, `.reuseCache` never worked, currently GPU-hangs — OOB tile-read hypothesis; touch only as hygiene). **MultiTF**: zero-work watch item.

## 6. Methodology guardrails (the house rules that earned their keep — do not skip)

- **Interleaved, one process, per-arm overrides.** Cross-process drift is ±6–10% vs within-process CV ~0.33% — sub-10% effects are unresolvable across processes (F-j was a cold-baseline artefact hiding exactly here). K is process-global (can't interleave K=1 vs K=4) — use the per-arm-K override, single process.
- **Every attribution carries a control arm + the two arithmetic identities.** Control (unablated) must reproduce the served baseline; sanity gate = every term positive, sum ≤ 100%. These caught WP-6b's 161% and an lm_head 86%-DCE. **Difference every arm against the FULL forward, never against another ablation** (deltas are marginals — WP-6e's router read 13.2% vs 1.7%, a 7.7× error).
- **Behavioural validation, not timing, proves a diagnostic works** (F-m/Step 3): assert output *changes* (or is byte-identical, as appropriate) — never reason about the optimiser. Timing cannot tell "free" from "deleted".
- **A "failed" sanity gate is a hypothesis, not a number to bend** (Step 3 3d-bis): the forward-share gate expected 75%, measured 94.8% — reconciled via 26 ms ÷ 27.5 ms = 94.5%, not massaged.
- **Provenance per row**: host (`hw.model`), `toolchain.mlxCoreVersion` + `mlxSwiftPackage` (P4 — `Package.resolved` is not evidence of what linked), `envValid` handling, warmup excluded. Cross-host deterministic metrics transfer only *directionally* (§1a — it's hardware FP, not the MLX version).
- **Mechanics**: `Tools/seed-metallib.sh` after every build; run tests via redirect `swift test > log 2>&1; echo $?` — **never `| tail`** (a failing suite exits 0 through the pipe); read the real `Test Suite 'All tests'` line. Never partially delete `.build` (build.db poisoning).
- **Re-freeze the Studio baseline** after any preset change and again after Step 5 if the kernel lands.

## 7. Fast start for a fresh context

1. Read `final_optimisations_plans.md` §2 (Steps 4–5) + `gather_qmm_handoff.md` §2.3/§3 (Case B, why it's not occupancy-bound) + §6 side-findings F-k/F-l (K=1 is the served default now).
2. **Do Step 4 first** — 4a (bump MLX, re-time), 4b (grep the `sortedIndices` wiring at the decode forward), 4c (8-bit + mxfp4 diagnostics). Each is a short bench arm; all use existing hooks.
3. Re-size the ~7.5 ms with whatever Step 4 returns; if a flag/upstream fix already took it, **stop and record** — the kernel may no longer be worth it.
4. Only then Step 5: GUI occupancy profile at T=32 → inline `MLXFast.metalKernel` (no fork). Pre-register the hypothesis; plan +10–20%, accept a recorded negative.
5. Report per the CLAUDE.md three-section rule (What you should know / What you should do / Next steps), provenance-complete.

**One honest caveat to carry in**: the baseline has already collected most headline numbers (the Redundancy Rule — six confirmations, zero counter-examples). Treat Case B's ~27% as an *upper bound the baseline may have partly banked*, and let Step 4 tell you how much is really left before committing to the kernel.
