# Step 3 — Engine-Phase Decomposition of the Denoising Step (F-n)

**Family**: Forward/step attribution (continues WP-6c's in-situ ablation into the engine loop).
**Host**: Mac Studio M2 Ultra (Mac14,14, 192 GB) · 4-bit served artefact · MLX core 0.31.1 (mlx-swift 0.31.6) · Swift 6.3.3.
**Status**: **CLOSED — both gated items NEGATIVE.** Recorded as F-n in `final_optimisations_plans.md` §2 Step 3.
**Raw data**: `scratch/step3_phases.jsonl` (host-local; gitignored).

---

## 1. Objective

WP-6c decomposed the *forward pass* (gather_qmm 42.9%, attention 14.3%, router 1.7%, …, remainder ≈21%) by causal ablation. Two roadmap items were gated on splitting the part of the **denoising step** that lives *outside* the forward — the sampler, the Γ/Δ selection-set construction, and the K-step loop-control readback:

- **Selection-set fusion (phase-3 §4.3)** — build a fused Γ/Δ kernel **only if** sampler+selection is **≳ 8–10%** of the step.
- **§13.1 raw-Metal escape-hatch** — spec an ICB decode-loop port **only if** loop-control is **> 10–15%**.

Neither could be decided by forward-level ablation: **F-m** established that `ModuleAblation` reaches only inside the forward (attn/MoE/lm_head/norms), and that norms are numerically un-ablatable on real weights anyway (skipping RMSNorm across 20 layers → inf/NaN → the loop hangs). The sampler/selection/loop live in the engine's denoising loop *around* the forward, so `ms/forward` can never isolate them. This step adds **timers in the engine loop** instead.

## 2. What was built

Four `instrument`-gated wall-clock accumulators, each forcing an `eval()` at a true sub-graph boundary so the otherwise-lazy MLX graph splits where the phase does:

| timer | where | boundary |
|---|---|---|
| `forwardSeconds` | `windowStep` | around `forward(...)`, `eval(logits)` |
| `samplerSeconds` | `windowStep` | around softmax/argmax/x0p/creditDecode, `eval` of sampler outputs |
| `selectionSeconds` | `windowStep` | around Γ/Δ + JOT eval up to the post-update window, `eval(nextWindow)` |
| `loopControlSeconds` | `denoisePhase` | around the K-step batch readback (`eval(flags)` + `flags.asArray`) — already the one real sync |

Threaded `Metrics` → scheduler → `diffusion-bench` JSONL + a console `phase shares` line. **Residual** = `denoise − forward − sampler − selection − loopControl` (embed/positional/glue), derived at analysis time. Gated behind the existing `instrument` flag → **off on the served path** (no eval added there).

**Validated behaviourally, not by timing** (the F-m lesson: a mis-wired timer reads 0 and looks like "phase free"):
- `testInstrumentTimersAreDiagnosticOnly` — `instrument:false` output is byte-identical and its four timers are exactly 0.
- extended `testCachedMatchesUncachedWithInstrumentation` — all four timers > 0 and their sum ≤ `denoiseSeconds`.
- Full suite green: 121 tests, 20 skipped, 0 failures, 364 s.

## 3. Findings

Ran `diffusion-bench llada --arms q-cached --speculation-k 1 --no-early-stop --runs 3 --gen-length 128` on the Studio, all three suites. 35/36 rows valid (1 warmup excluded, 0 env-invalid); cross-run variance **0.5% — PASS**.

**Phase shares of `denoiseSeconds` (median over valid rows, identical across chat/reasoning/code):**

| phase | share | range |
|---|---|---|
| forward (whole model) | **94.8%** | 93.6–95.0 |
| sampler | 1.6% | 1.5–2.7 |
| selection | 1.3% | 1.2–1.4 |
| loop-control | 2.0% | 1.9–2.2 |
| residual | 0.3% | 0.2–0.3 |

- `[Sourced]` **Selection-set fusion — CLOSED.** sampler+selection = **2.9%**, far under the ≳8–10% build threshold. No fused Γ/Δ kernel.
- `[Sourced]` **§13.1 escape-hatch — CLOSED WITH DATA** (the pre-registered "likely" outcome). loop-control = **2.0%**, far under >10–15%. At the Studio's K=1 served default (F-l) the readback is one `.item()` per step; its cost is negligible. Keep the hatch shut.
- `[Inferred]` **Both verdicts are upper bounds.** Eval-inflation cuts *in our favour* here: each engine-loop sub-phase pays a fixed forced-`eval` sync that production never pays, so the *true* sampler/selection/loop shares are even smaller. The negatives only harden.

### 3.1 The sanity gate said "forward ≈ 75%"; measured 94.8% — reconciled, not massaged

The handoff §5 sanity gate expected `forward ≈ MoE+attn+lmHead+norms ≈ 75%` of the step and flagged anything else as "instrumentation mis-placed — stop and fix." Measured 94.8%. This is **not** a mis-placed timer — it is corroborated by two independent, pre-existing numbers:

> P9's full forward ≈ **26 ms** ÷ q-cached's ms/step ≈ **27.5 ms** = **94.5%** — matching the 94.8% timer to < 0.5 pt.

The §1 "~25% remainder" framing had **conflated two different remainders**: P9's *within-forward* remainder (norms + embed + lm_head ≈ 25% **of the forward**, which lives *inside* this `forward` bucket) with the engine-loop overhead. The actual engine loop is only ~5% of the *step*, of which the two addressable pieces are 2.9% and 2.0%. On the compute-bound Studio the 20-layer / 256-expert MoE forward simply dominates everything the CPU-side loop does.

## 4. Verdict & consequences

**CLOSED.** Selection-set fusion and the §13.1 raw-Metal escape-hatch are both **dead with data** — the engine loop they target is ~5% of the step and shrinking under scrutiny. This clears the last of the phase-3 "forward-budget" items; what remains live in the plan is LocalLeap (conditional on Step 1) and the phase-4 accuracy debts (ICE significance, Credit-Decoding global-default reconsideration, TSCV upgrades).

## 5. Lessons

- **The engine loop is a rounding error on a compute-bound host.** Everything that matters for latency here is inside the forward (WP-6c/6f territory: the 4-bit MoE gather). Kernel-fusing the CPU-side selection/loop machinery cannot pay when it is 5% of the step and hides behind GPU work.
- **A "failed" sanity gate is a hypothesis to test, not a number to bend.** The 75% expectation was a stale estimate; the honest move was to derive the expected forward share from two independent known quantities (26 ms / 27.5 ms) and confirm the 94.8% is real — the same control-arithmetic discipline that caught WP-6b's 161% and the lm_head 86% DCE.
- **Eval-inflation has a direction.** Forcing a per-phase `eval` over-counts the *cheap* phases (each pays one fixed sync); it never under-counts them. So an eval-inflated share is a safe **upper bound** for a "don't build" decision, and reporting it as such is stronger than pretending to production ms.
- **Behavioural guards, not timing, prove a diagnostic works** (F-m redux): the parity test asserting instrument-on ≡ instrument-off is what makes the numbers trustworthy.
