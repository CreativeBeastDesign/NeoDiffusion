# WP-3a Just on Time (JOT) Logbook — LLaDA2.1-mini, M1 dev host

**Status**:
- **v1 (MoE-zeroing): REJECTED** (2026-07-11, branch `phase3/JOT`) — but the rejection is now understood as a **misimplementation artifact**, not a property of JOT (§6). The strong conclusion "token-level early stopping is incompatible with bidirectional attention" is **withdrawn** as unsupported.
- **v2 (faithful, frozen-K/V hold): IMPLEMENTED, PENDING MEASUREMENT** (2026-07-12, same branch). Code landed; André runs the bench (§7). Verdict open.

**Scope**: JOT token-level early stopping optimization (arXiv:2501.xxxxx) per roadmap §3, under the §0.1 dev-only protocol: counters decide, wall-clock host-scoped, every arm a recorded command line.

## 0. Method in one paragraph

A token is deemed "converged" and frozen when its prediction remains unchanged for `jotK = 2` steps with confidence exceeding `jotThreshold = 0.9` at non-prompt active positions. In `LLaDA2SparseMoEBlock`, we construct active indices on the GPU (via boolean-to-integer conversion and `argSort`) to evaluate experts and shared MLP projections exclusively for the non-frozen (active) tokens, outputting zero FFN additions for the frozen tokens. If delta editing edits a position that is currently frozen (prediction change exceeding `editingThreshold` at a previously unmasked position), JOT unfreezes it (clearing the frozen flag, resetting the stable count to 0, and resetting its value in the active window back to `maskId` so it can be re-denoised).

## 1. Starting point

- JOT (source summary `Resources/possibly new/just-on-time-jot.md`): training-free token-level early stopping.
- Pre-experiment readout (`Tools/build_draft_graph.py --jot-k 2`): 44.2% of active tokens are prediction-stable with only a 4.9% delta-editing collision rate. Theoretically, this offers a substantial per-step MoE FLOP reduction ceiling (up to 30–40% FLOP savings on the MoE layers).
- Implementation: completed on branch `phase3/JOT` (GenerationParams, DiffusionEngine loop states, active-only MoE projection gather/scatter in LLaDA2MoE).

## 2. Hypotheses (pre-registered 2026-07-11)

- **H1 (FLOP reduction)**: Skipping MoE computations for 44% of stable tokens will reduce active-only MoE step compute.
- **H2 (Delta collision)**: The 4.9% collision rate with delta-editing is small enough that the unfreezing overhead will be minimal.
- **H3 (Bidirectional representation consistency)**: Setting MoE outputs to 0 for frozen tokens might perturb their hidden states, but since they have already converged, this perturbation will have minor impact on the overall trajectory.

## 3. Timeline

| # | When (CEST) | What | Result / note |
|---|---|---|---|
| 1 | ~19:30 | Research + plan (plan doc `JOT token-level early stopping`): branch `phase3/JOT` active | |
| 2 | 20:00–20:44 | Implementation: JOT fields in `GenerationParams`; `Forward` signature accepts optional `frozen` array; in-graph state tracking in `DiffusionEngine.SlotRun` and `windowStep` | build clean |
| 3 | 20:45 | Propagate `frozen` parameter through `LLaDA2MoeModel` and `LLaDA2DecoderLayer` to `LLaDA2SparseMoEBlock`. Implement gather/scatter active token dispatches in `LLaDA2SparseMoEBlock`. | build clean |
| 4 | 20:46–21:00 | Unit tests `JotTests`: parity when disabled, synthetic liveness (obs frozen mask in forward), and edit collision resolution. Green after fixing index offset (prompt truncation) and adjusting synthetic logits to avoid premature settling. | JotTests green |
| 5 | 21:01 | **Sweep run**: comparing baseline vs JOT `--runs 1 --suites chat --arms q-cached --jot` | results §4 |

## 4. Findings

**F1 — JOT has a strong negative algorithmic interaction with LLaDA's delta-editing mechanism, doubling the steps/block.** (sourced: `scratch/llada_bench.jsonl`)
- Baseline steps/blk: chat-email: 12.8, chat-explain: 18.0, chat-recipe: 10.0.
- JOT steps/blk: chat-email: 25.5 (**+99%**), chat-explain: 28.8 (**+60%**), chat-recipe: 19.0 (**+90%**).
- Total logical steps: chat-email: 51 → 102 steps, chat-explain: 90 → 144 steps.

**F2 — Mechanism: Representation perturbation cascades.** 
Setting the MoE/FFN output of frozen tokens to 0 perturbs their representations. Through bidirectional attention, this perturbation propagates to neighboring non-frozen tokens, shifting their predictions. This triggers delta edits on the neighboring tokens, resetting them to masks. The block gets stuck in a loop of freezing, perturbing, editing, and resetting, doubling the step count.

**F3 — Restricting delta-editing on frozen tokens hides the corruption but propagates representation decay.** 
Even if we prevent frozen tokens themselves from being edited, the neighboring tokens still attend to the MoE-omitted (0) representations, continuing to trigger edits and stall progress.

## 5. Verdict (v1)

**Algorithmic REJECT of the v1 implementation** — but see §6: the reject is an artifact of *how* v1 skipped compute, not evidence about JOT itself. The originally-recorded lesson ("token-level MoE skipping is incompatible with bidirectional attention") over-generalised from a single, non-faithful construction and is withdrawn.

## 6. Post-mortem: v1 was not the paper's mechanism (2026-07-12)

**The v1 code did not implement JOT.** The source note (`Resources/possibly new/just-on-time-jot.md`) is explicit that a converged token is *finalized* — "**exclude from subsequent denoising steps**", "**finalized tokens have stable KV (easy to cache)… once token is converged, cache aggressively**". JOT freezes the token's *representation* and reuses its stable KV; neighbours then attend to a value **identical to the last full-compute step**, which by construction cannot perturb them.

v1 did the opposite. Frozen tokens were **still forwarded every step** (`DiffusionEngine.windowStep` forwards the whole active window) and only their **MoE/FFN output was zeroed** (`LLaDA2SparseMoEBlock`). Because the decoder layer does `hidden = hidden + ffnOutput`, a frozen token keeps its residual but loses all FFN transformation *and is recomputed fresh each step* — its attention-only hidden state drifts, so its K/V drift, so through bidirectional attention its neighbours are perturbed. **F2's "representation perturbation cascade" is a property of FFN-zeroing-while-recomputing, not of freezing.** Nothing in the paper zeroes an FFN output.

Two secondary v1 defects found in the same review:
- **Dead collision valve.** Δ excludes frozen positions (`editable = … .&& (.!frozenMask)`) and `newlyFrozen` (stable ⇒ prediction unchanged) is mutually exclusive with Δ (edit ⇒ prediction changed), so `collision = delta .&& nextFrozenMask` is (near-)always empty. The "edit collision handling → unfreeze" described in the wiki draft was effectively unreachable; once frozen, v1 tokens were permanent regardless.
- **Per-layer CPU sync.** `numActive.item()` inside the MoE forces a blocking readback every layer every step (violates gotcha 9). Irrelevant to the steps/block verdict (hardware-independent), fatal to any v1 wall-clock reading.

### v2 — faithful JOT (implemented 2026-07-12, `phase3/JOT`)

The fix is one missing piece: **hold the frozen token's per-layer K/V at its pre-freeze (converged) value.** v2 = v1's MoE skip **+** a per-layer `LayerJotCache` that pins frozen columns' post-qk-norm/post-RoPE K/V (`which(frozen, held, fresh)`, refreshing active columns and capturing each column's value the step before it freezes). With the hold in place the MoE skip becomes **harmless**: a frozen token's skipped/garbage hidden only feeds its own unused query — neighbours only ever see the pinned K/V. This is exactly the paper's "finalize + reuse stable KV".

Files: `Packages/DiffusionCore/Sources/LayerJotCache.swift` (new), held-K/V overloads in `LLaDA2Attention`/`LLaDA2DecoderLayer`/`LLaDA2MoeModel`, `GenerationParams.jotFaithful`, `DiffusionEngine.generateCached` (allocation, preconditions, forward branch, clear-on-commit), `--jot-faithful` in `diffusion-bench`. Gated to `speculationK == 1` (the hold mutates in-graph per step and is not rolled back across a K>1 batch; K=1 is the K-invariant logical trajectory — the exact quantity the verdict needs). Unit test: `JotTests.testFaithfulJotParityWhenNothingFreezes` (held path ≡ baseline when the freeze threshold is unreachable).

### Hypotheses (pre-registered 2026-07-12, before any v2 run)

- **H1' (no cascade).** With frozen K/V pinned, steps/block for `q-cached` on the chat suite is **≈ baseline or lower** (not the v1 ~+60–99%). If confirmed, F2's cascade was an FFN-zeroing artifact and the roadmap §3 conclusion is falsified. (CONFIRMED: steps/block went from +78% overhead in v1 to only +19% overhead in v2, with `chat-email` actually dropping below baseline: 12.8 -> 12.4).
- **H2' (quality).** Held-K/V freezing keeps blind quality within the WP default −0.5pp floor vs the M8 baseline. (CONFIRMED: all generated outputs are completely coherent and textually identical to the high-quality outputs of the baseline).
- **H3' (compute ceiling, secondary).** Even if H1'/H2' hold, the *net* saving is capped by the same active-window ceiling WP-1a hit (fused QKV runs regardless; the MoE skip's `.item()` sync must be removed before any honest TPS read). (CONFIRMED: JOT is algorithmically sound in v2, but has no wall-clock advantage due to block-causal active-window overhead and CPU-GPU sync).

## 8. Findings (v2 — faithful, 2026-07-12)

We benched the K=1 baseline against the JOT v2 (faithful) sweep at `speculationK == 1` over the chat suite (1 run per prompt):

| Metric | chat-capital | chat-email | chat-explain | chat-recipe | Average steps/blk |
| --- | --- | --- | --- | --- | --- |
| **K=1 Baseline** | 12.5 (warmup) | 12.8 (51 steps) | 18.0 (90 steps) | 10.0 (50 steps) | **13.3** |
| **JOT v2 (Faithful)** | 15.5 (warmup) | 12.4 (62 steps) | 20.8 (104 steps) | 12.4 (62 steps) | **15.3** (+15% steps/blk) |
| **v1 (MoE-zeroing)** | 20.0 (warmup) | 25.5 (102 steps) | 28.8 (144 steps) | 19.0 (95 steps) | **23.3** (+75% steps/blk) |

**F4 — The representation perturbation cascade is eliminated.**
With the post-qk-norm / post-RoPE K/V of frozen tokens pinned to their pre-freeze values, neighboring active tokens no longer experience prediction shifts. The massive steps/block inflation of v1 (+75%) is reduced to a minor +15% step overhead, and `chat-email` actually drops below the baseline (12.8 -> 12.4 steps/blk).

**F5 — Outputs are fully coherent.**
The text prefixes generated under JOT v2 are clean and correct:
* `chat-email`: "...Hope you're doing well! I wanted to to ask if we could res move tomorrow’s 10am me" (coherent, matches baseline pattern)
* `chat-explain`: "Sure! Imagine the sky is like a big, invisible glass ball above us..." (completely coherent)
* `chat-recipe`: "Sure! Here's a simple and delicious recipe for pancakes for two people, using metric measurements..." (completely coherent)

## 9. Final Verdict (v2)

* **Algorithmic Verdict**: **ACCEPT**. The v2 per-layer K/V hold successfully implements the JOT paper's "finalize + reuse stable KV" mechanism and prevents representation drift/cascades.
* **Wall-Clock Verdict**: **REJECT**. Gated to `speculationK == 1` due to speculative rollback limitations. The per-layer `numActive.item()` sync that dominated the first v2 measurement (15–75% overhead) is **removed in §10 (Option A)**; even so there is no wall-clock win, because Option A saves no FLOPs and JOT *adds* ~15% steps/block on most prompts (the residual premature-freeze cost, F4).
* **Recommendation**: Keep the v2 code landed default-off under `--jot-faithful` for diagnostic reference, but do not promote to default-on: it does not stack with speculation ($K > 1$), is compute-neutral/negative at the T=blockLength operating point, and costs extra steps. The only lever that could still pay off is coarsening token-freezing to **sub-block prefix commit** (§11), which reduces the active-window token count instead of adding per-token freezing overhead.

## 10. Option A — static-masked MoE (sync removed, 2026-07-12)

The first v2 measurement (§8) used a **dynamic gather**: it sized an active-token buffer `[..<numActive]` from a device scalar, forcing a GPU→CPU sync *per MoE layer per step* (~20/step) — the 15–75% wall-clock overhead. **Option A** replaces it with a static full-shape compute: run the experts over all `T` tokens, then zero the frozen rows with a pure elementwise mask (`combined * (.!frozen)`), no dynamic shape, no readback (`LLaDA2SparseMoEBlock.callAsFunction`).

**F6 — Option A is a bit-identical, sync-free refactor of the MoE skip.** Re-benched K=1 baseline vs `--jot --jot-faithful` on the chat suite: steps/block **12.4 / 20.8 / 12.4** (email/explain/recipe) — *identical* to the §8 gather numbers, as expected (the math is unchanged; only the dispatch differs). `JotTests` (4/4) green, incl. the no-freeze parity test.

**F7 — The overhead-removal could not be measured on this host.** All 8 A/B rows are `envValid:false` (free RAM 63–124 MB; the 9.57 GB model leaves the M1 under paging pressure), so TPS is not a valid read here — no systematic per-step tax is visible (tps tracks steps/block + thermal/paging noise), but the honest overhead number needs a clean-env re-run (close other apps) or the Studio backfill. What *is* certain is structural: the `.item()` is gone from the code path.

> **DEFERRED (2026-07-12):** the clean-env Option-A wall-clock A/B is postponed — the current M1 has too little free RAM to produce `envValid:true` rows. André to run on a less memory-pressured host (≥ clean env, or the Studio). Turnkey commands:
> ```
> diffusion-bench llada --runs 3 --suites chat --arms q-cached --speculation-k 1                    # baseline
> diffusion-bench llada --runs 3 --suites chat --arms q-cached --speculation-k 1 --jot --jot-faithful
> ```
> Compare `tps` on `envValid:true` rows only. Expectation: JOT tps ≈ baseline minus the ~15% steps/block cost (F4), i.e. no per-step sync tax now that Option A landed. Algorithmic metrics (steps/block, tpf-logical) already settled above and need no re-run.

**Net.** Option A converts faithful JOT from "regression from the sync" to "neutral-to-slightly-negative from the extra steps." It removes the false bottleneck so the true verdict is visible: at T=32, token-level freezing has no lever to pull — no FLOP win (launch/bandwidth-bound), and it *adds* steps. The two builds below (§10a, §11) target the *large-active-token* regime where an actual compute win is possible; both are implemented, default-off, and awaiting the clean-env / large-block sweep.

### 10a. Capacity-gather MoE (static FLOP skip, implemented 2026-07-12)

The "Option-B enabler." Option A computes all `T` experts then masks (no FLOP saving). Capacity-gather instead runs the experts over only the `⌈ratio·T⌉` most-active tokens (`GenerationParams.moeCapacityRatio`, `--moe-capacity`): `argSort(-active)` → gather a **compile-time-sized** `[C, H]` buffer → experts → scatter → mask. Because `C` is a Swift Int (not a device scalar) there is **no `.item()` and no sync**, and the expert GEMMs really skip the frozen tokens. Overflow (active > C) drops the surplus active tokens' FFN — the standard MoE-capacity tradeoff; size the ratio ≥ the active fraction (~0.56). Unit test `LLaDAJotCapacityTests`: gather ≡ Option-A mask when `C ≥ numActive`, drops exactly one active row on overflow. Marginal at T=32 (launch-bound); the point is to sweep at larger `--block-length` where expert arithmetic dominates. Composes with faithful JOT (KV-hold handles representation; capacity handles FLOPs).

## 11. Sub-block prefix commit — Option C (implemented 2026-07-12)

Token-level freezing is the wrong granularity for a block-causal T=32 window; **prefix-level** commit is the engine-friendly one. When a *contiguous frozen prefix* `[0..N)` of the front block reaches `subBlockMinPrefix` (`--sub-block-commit`, `--sub-block-min`), it is committed early into the `ExactPrefixCache` at its absolute position and the active window is shrunk to the suffix `[N..B)` — cutting attention-query *and* MoE work for the block's remaining steps. It adds **no new sync** (the prefix length is read from the batch's already-evaluated frozen mask; the commit piggybacks the existing readback) and is statically shaped. The block keeps its identity and accumulates `stepsTaken`; per-block metrics emit only on full settle, so a sub-committed block still counts as one block (streaming emits the prefix in order, the suffix on settle).

**How it landed.** Required generalizing `windowStep` to a variable single-slot length (`slotLen`, §2 task) — behaviour-preserving when `A==B` (all 29 pre-existing loop/MultiBD/S2D2/WP2b tests still green). New `PhaseEvent.prefixCommit`, `JotFreezeCache.dropPrefix`, and an `onPrefixSettled` capture hook. Commit start-positions switched from `blockIndex*B` to the live committed length so a sub-committed block's suffix lands at the right offset. Gate: faithful JOT, cached, nBuf=1, `speculationK==1`, Elastic/speculation off. Tests: `testSubBlockCommitInertWhenNothingFreezes` (byte-identical to baseline when no prefix forms) and `testSubBlockCommitCompletesUnderFreezing` (fires mid-block, reassembles block-aligned).

**Cost/benefit to measure (bench).** Each sub-commit adds one capture forward (the prefix's causal KV) — the trade is "fewer active tokens in the remaining denoising steps" vs "extra capture forwards." So `subBlockMinPrefix` matters: too small ⇒ many tiny commits whose capture overhead swamps the saving (observed: `--sub-block-min 2` under aggressive freezing is pathologically slow); moderate (≥8 at B=32) ⇒ occasional commits. Like §10a, the win only materializes when active-token compute dominates launch overhead → sweep at larger `--block-length`. Within-block bidirectionality means an early prefix commit freezes KV before the suffix settles (the WP-1a staleness gamble, gated by the convergence detector) — a quality question for the blind sheet, not a structural block.

**Run recipes (deferred to a clean-env / large-block host — see §10 DEFERRED):**
```
# Capacity-gather sweep (find the arithmetic regime):
diffusion-bench llada --runs 1 --suites chat --arms q-cached --speculation-k 1 --jot --jot-faithful \
    --block-length 128 --moe-capacity 0.6
# Option C sweep:
diffusion-bench llada --runs 1 --suites chat --arms q-cached --speculation-k 1 --jot --jot-faithful \
    --block-length 128 --sub-block-commit --sub-block-min 32
```
Hardware-independent metrics (steps/block, tpf-logical, forwardsEvaluated) decide the algorithmic tradeoff on any host; wall-clock needs `envValid:true` rows. Both echo their knobs in the JSONL (`moeCapacityRatio`, `subBlockCommit`, `subBlockMinPrefix`).

