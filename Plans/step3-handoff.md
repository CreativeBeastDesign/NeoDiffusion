# Step 3 hand-off — decompose the ~21% forward "remainder" with engine-phase timers

**Status**: forward-level ablation path CLOSED (recorded negative, F-m); this doc is the continuation.
**Created**: 2026-07-17. **Prereq context**: `Plans/final_optimisations_plans.md` §2 Step 3 + F-m; `Plans/gather_qmm_handoff.md` §5 (the WP-6c ablation method this replaces for the remainder).
**Complexity**: ~5/10 (see §7). Diagnostic-only, no production risk, established pattern to mirror — the care is in *measurement interpretation*, not the code.

---

## 1. The one-paragraph goal

The Studio forward budget is **MoE 61.8%, attention 12.8%, remainder ~25%** (P9, control-validated 2026-07-17). The ~25% remainder is the second-largest bucket and is known **only by subtraction** — it lumps norms + the sampler + the Γ/Δ selection-set construction + the K-step loop-control readbacks + embed. This step splits it, and that split **decides two roadmap items with no further work**:

- **Selection-set fusion** (phase-3 §4.3): build a fused Γ/Δ selection kernel **only if** the sampler + selection tail is **≳ 8–10%** of the forward.
- **§13.1 raw-Metal escape-hatch**: spec the ICB decode-loop port **only if** loop-control overhead is **> 10–15%**; otherwise close the hatch *with data* and record why.

No implementation of either lands until this number exists.

## 2. Why the obvious approach (more `ModuleAblation` arms) does NOT work — read this first

Do **not** try to extend `ModuleAblation` for this. It was tried and closed (F-m, `final_optimisations_plans.md` §2):

1. **Norms are numerically un-ablatable on real weights.** `ModuleAblation.layerNorms` exists and is behaviourally correct on toy fixtures (the guard `ModuleAblationTests/testAblationActuallyBitesOnDefaultPath` passes), but on the 4-bit model, skipping RMSNorm across 20 layers compounds to inf/NaN and the denoising loop **hangs** (never clears a mask; killed after 64 min). RMSNorm is load-bearing for numerical stability. The case is kept as a recorded negative with **no bench arm** (it would hang a session).
2. **Sampler / selection / loop live OUTSIDE the forward.** `ModuleAblation` operates inside the model forward (attn/MoE/lm_head/norms). The sampler, Γ/Δ construction, and loop control run in the *engine's denoising loop around* the forward. `ms/forward = denoiseSeconds/forwardsEvaluated` can never isolate them.

So the remainder needs **timers in the engine loop**, gated behind the existing `instrument` flag — not ablation.

## 3. Where the phases are (verified file/line references, 2026-07-17)

All in `Packages/DiffusionGeneration/Sources/`:

- **The per-step body** is `windowStep(...)` in `DiffusionEngine+Step.swift` (private, ~line 364). Its sub-phases, in order:
  - **Forward**: `let logits = forward(window, A, ...)` — ~line 412.
  - **Sampler**: `softmax(logits)`, `x0`/`x0p`, credit application, argmax — ~lines 416–433.
  - **Selection-set (Γ/Δ)**: gamma/delta masks, threshold compare, τ_semi-gated top-1 fallback, write-token selection — ~lines 433–460+.
  - **Loop-control readback**: the single blocking sync of the stacked event flags — `eval(flags)` / `eval(confs)` at ~lines 170–172 (inside `denoisePhase`, which calls `windowStep` K times per batch).
- **The phase-timing pattern to mirror** is in `DiffusionEngine+Scheduler.swift` ~lines 200–212: `let denoiseStart = Date()` → run `denoisePhase` → `let phaseSeconds = Date().timeIntervalSince(denoiseStart)` → accumulate. This is exactly the shape the sub-phase timers take, one level deeper.
- **`Metrics`** struct: `DiffusionEngine.swift` ~line 118 (add the new `*Seconds` fields here, next to `singleActiveDenoiseSeconds`/`dualActiveDenoiseSeconds` at ~165–166); the mutable accumulators are constructed in the scheduler and passed into the `Metrics(...)` init at ~line 354.
- **Bench recording**: `Tools/diffusion-bench/Sources/LLaDABench.swift` — `LLaDARunResult` struct (~line 177) + the `appendResult` row build (~line 863) + the console/analysis. Add the new fields there so they land in JSONL.

## 4. The approach

Add four wall-clock accumulators, **gated behind `instrument`** (the eval overhead is only acceptable in diagnostic mode, exactly like the existing phase timers):

1. `forwardSeconds` — around the `forward(...)` call, with `eval(logits)` immediately after to force the graph.
2. `samplerSeconds` — around softmax/x0p/credit/argmax, `eval` the sampler outputs.
3. `selectionSeconds` — around Γ/Δ construction + threshold + fallback, `eval` the write masks.
4. `loopControlSeconds` — around the `eval(flags)` readback (this one is *already* a sync, so it's the cleanest to time).

Then: new phases summed and differenced against `denoiseSeconds` give the split. `denoiseSeconds − forward − sampler − selection − loopControl = residual` (embed/positional/glue).

## 5. Measurement protocol (and the ONE trap that matters)

- **Run at K=1.** Speculation batches K `windowStep`s before a sync; per-phase timing is cleanest when K=1 (one step per sync). Use the `spk-1` arm or `--speculation-k 1`. (The served default is now K=1 on the Studio anyway — F-l.)
- **`--no-early-stop`, ≥3 runs, interleaved (P2 default), envValid rows only, warmup excluded** — the standard discipline. Variance gate must PASS.
- **⚠️ The trap — eval-inflation (same lesson as `ModuleAblation`'s doc §"Why this exists").** Forcing `eval()` at each sub-phase boundary pays a sync per phase that production never pays, and that overhead is *host-dependent and phase-dependent*. So the per-phase **absolute ms are inflated** — treat them as **relative shares within the instrumented loop**, cross-checked against the un-instrumented `denoiseSeconds` total, **not** as production ms. This is the exact reason the microbench (`LLaDAMoEDispatchBench`) could not size modules on the Studio (§5 of the gather_qmm handoff). Report shares with this caveat stated, per the provenance rule.
- **Sanity gate** (like the ablation method): every phase positive; forward+sampler+selection+loopControl+residual ≈ denoiseSeconds. If the forward share doesn't roughly match P9's non-remainder budget (forward ≈ MoE+attn+lmHead+norms ≈ 75%), the instrumentation is mis-placed — stop and fix, do not massage.

## 6. Decision, once the number exists

| finding | action |
|---|---|
| sampler+selection ≳ 8–10% | build the fused Γ/Δ selection kernel (phase-3 §4.3) |
| sampler+selection < 8% | close selection-set fusion with data |
| loop-control > 10–15% | spec the §13.1 ICB decode-loop raw-Metal port |
| loop-control < 10% | **close the §13.1 escape-hatch with data** and record why (this is the likely outcome — the K-step readback is one `.item()` per step, and K=1 serving makes it one per step; but measure, don't assume) |

Record positive or negative in `final_optimisations_plans.md` §2 Step 3 + a wiki draft, either way.

## 7. Complexity: ~5/10

- **What makes it a 5, not a 3**: it instruments the hot per-step loop (careful `eval` placement so fused sub-graphs split at true boundaries), plumbs four fields through three files (Metrics → scheduler → bench), and — the real work — the results carry the eval-inflation interpretation caveat that needs the same paranoid discipline as the rest of this campaign.
- **What keeps it from an 8**: no new algorithms, no kernel, no numerics; the phase-timing pattern already exists to copy; it's diagnostic-only (behind `instrument`, off on the served path — the parity suite protects production); and the decision thresholds are pre-defined. For contrast, **Step 5 (the gather_qmm custom kernel) is ~8–9/10** — Instruments profiling + inline-MSL kernel authoring against a moving MLX target.

## 8. Guardrails (house rules that bit this session — do not skip)

- **Behavioural validation, not timing, proves a diagnostic works.** F-m's mis-wired overload read as "norms free" on ms/forward=0; only the output-difference guard caught it. Add an equivalent check that the timers are non-zero and move with load.
- **Never read a piped exit code / green-looking signal as a result** (CLAUDE.md reporting rule). Redirect, read the real summary.
- **Instrument-only**: the timers must not exist on the served path. Add/extend a parity assertion that `instrument:false` output is byte-identical (mirror `testNoneIsBitIdenticalToBaseline`).
- **Provenance**: every row carries `toolchain` (P4) + host; report host, MLX core version, envValid/warmup handling.
- **`Tools/seed-metallib.sh` after every build**; run tests via redirect (`swift test > log 2>&1; echo $?`), never `| tail`.

## 9. Fast start for a fresh context

1. Read `final_optimisations_plans.md` §0 (budget) + §2 Step 3 + F-m; skim `gather_qmm_handoff.md` §5 (why microbench/eval inflates).
2. Open `DiffusionEngine+Step.swift` `windowStep` and `DiffusionEngine+Scheduler.swift` ~200–212 (the timing pattern).
3. Add the four `instrument`-gated timers + Metrics fields + scheduler accumulation + bench fields.
4. Add the non-zero/parity guards (§8).
5. Run at K=1, `--no-early-stop`, 3 runs interleaved; apply §5 sanity gate + eval-inflation caveat; decide per §6.
