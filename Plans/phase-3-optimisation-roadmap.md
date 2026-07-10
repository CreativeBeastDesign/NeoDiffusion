# NeoDiffusion Phase 3 — Optimisation Roadmap

**Status**: draft
**Last updated**: 2026-07-04
**Prerequisite**: Phase 2 milestones M0–M8 complete; specifically **M6's frozen baseline** (TPS, TPF, steps/block, post-steps/block, sync-count, peak memory, per-phase wall-clock on both machines) is the reference every work package (WP) measures against. No WP lands without a bench delta.
**Priority (confirmed 2026-07-04)**: Elastic-Cache ≈ MultiBD (co-first) ≥ Spiffy ≈ Streaming-dLLM > d²Cache > LocalLeap ≈ FreeDave > Sparse-dLLM. MoE-kernel fusion is a cross-cutting track driven by profiling, not a list position.

---

## 0. Ground rules

- **Three levers, keep them separate in analysis**: (i) *fewer steps per block* (threshold calibration, early exit, editing discipline), (ii) *cheaper steps* (caching, suffix pruning, kernel fusion), (iii) *more useful tokens per forward* (MultiBD, Spiffy-style speculation, parallel decoding). A WP claims one primary lever; cross-lever interactions go in the composability matrix (§5).
- **Every WP produces a wiki experiment page** (`05-Experiments/`) linking the proposal/concepts it validates or refutes; negative results are recorded, not retried without a new hypothesis (wiki-rules discipline).
- **Quality gates**: each WP defines its own accuracy floor relative to the M8-selected quantized baseline on the bench's scored set. Default floor: −0.5pp unless the WP argues otherwise up front.
- **One-variable discipline**: WPs land sequentially onto `main`; a WP benches against the current accumulated stack *and* against baseline, so both marginal and cumulative gains are known.
- All numbers quoted below are from source papers on non-Apple hardware — treat as direction and ceiling, not prediction (**inferred** portability throughout).

### 0.1 Dev-only protocol (added 2026-07-10 — Phase 3 may proceed on the M1 without the Studio)

Phase 3 can start and produce durable verdicts on the dev M1 alone, **if every WP separates its evidence into two classes** so the later Studio runs extend rather than invalidate it:

1. **Hardware-independent metrics decide algorithmic questions**: steps/block, post-steps/block, forwards-evaluated (TPF), tokens/step, sync-point count, selection-set sizes, and all quality scores are properties of the *algorithm + weights*, not the chip — measured on M1, they transfer. A WP whose lever is "fewer steps" or "more tokens/forward" (MultiBD, early exit, threshold work, Spiffy) can reach its accept/reject verdict entirely on the M1.
2. **Wall-clock claims are host-scoped**: TPS/latency/kernel rankings are labelled with the host (every bench JSONL row now carries `host`, env snapshot, thermal state, `envValid`, and warmup classification — m6-logbook §5 rules). M1 wall-clock A/Bs are valid *within-host* (within-process ≤1% repeatability; cross-process ±6–10% thermal drift, m6-logbook Finding 7). Kernel/dispatch choices additionally cite their scope assumptions and re-run trigger list (Finding 3 pattern) — e.g. the `gather_qmm` ranking must be re-measured on the M2 Ultra before Phase-3 kernel work assumes it there.
3. **Correctness gates stay hardware-independent**: WPs touching the loop/caches gate against the M5 toy-fixture parity suite (config-driven, runs anywhere) — this is orthogonal to the still-open Studio BF16 parity debt, which validates the *forward* stack, not the scheduling Phase 3 changes. If the BF16 run later surfaces a forward-level discrepancy, Phase-3 scheduling verdicts survive; only absolute quality numbers would need re-anchoring.
4. **Studio backfill is a re-run, not a redo**: every arm is a recorded command line (logbook discipline); the Studio session re-executes the same commands → host-tagged rows land in the same JSONL next to the M1 rows; the M6 dev baseline gets a Studio twin (re-freeze) and each landed WP re-benches marginal + cumulative per §6's existing cadence. Budget one Studio session per landed tier, not per WP.
5. **Baseline for M1-side A/Bs**: the frozen M6 dev baseline + the accumulated-stack rule from §0 apply unchanged on the dev host; the M8-selected quality floor (accuracy relative to the scored set) is hardware-independent and ports as-is.

## 1. Tier 1 (co-first)

### WP-1a — Elastic-Cache pipeline (lever: cheaper steps)

Implements the wiki proposal [[elastic-cache-metal-kernel]] unchanged, against the **ActiveBlockCache only** (the exact prefix tier is never policy-managed):
- Phase A: full validated pipeline — sliding window β=16 over active MASK set, most-attended-token drift test (γ=0.9 starting point, re-tune on Metal), depth boundary ℓ★ recompute-deep/reuse-shallow. Establishes the staleness-ablation harness.
- Phase B: swap in token-stability per-token signal (θ 0.95–0.99), window and ℓ★ held fixed.
- Phase C: OR/AND combined-signal ablation.
Source ceiling: 45.1× on long sequences (v1/v2 discrepancy unresolved — see proposal caveats); expect far less at our short-context chat workload where prefix recompute is already eliminated by ExactPrefixCache. Acceptance per the proposal's own criteria; γ, ℓ★, β re-tuned on M2 Ultra.
**Note on scope** (**inferred**): our engine already has ExactPrefixCache, which the CUDA baselines lacked — Elastic-Cache's headline numbers partly include savings we already banked in Phase 2. The honest measurement is marginal gain over M6, not reproduction of 45×.

#### STATUS: CLOSED — NEGATIVE RESULT (2026-07-10, record: `Plans/elastic-cache-logbook.md`, wiki draft `Plans/wiki-drafts/wp-1a-elastic-cache-active-kv-reuse.md`)

Phase A rejected on the dev host under §0.1; Phases B/C closed without running (same empty ceiling). The rejection is **architectural**: on block-causal LLaDA2.x, ExactPrefixCache absorbs the paper's decoded-token refresh tier exactly, and the Block-Buffer loop never computes distant MASKs (the paper's other tier) — the only cacheable object left is the active window itself, whose staleness inflates steps/block monotonically in reuse depth (12.6 → up to 30.6) while the as-built reuse skips ~0% compute (fused QKV + MoE run regardless; perfect-implementation ceiling ≈ 4–5%). Surviving direction: per-position selective recompute (d²Cache/vicinity slot, Tier 3). Serving-path drift-instrumentation leak fixed on `main` (660a9a7).

*Superseded* — the previously listed "Future Experiment Directions" (finer static sweeps, lighter similarity metrics, adaptive block scaling) are withdrawn: the first two cannot change the sign of a zero-ceiling optimisation; the third is a different lever (block sizing, not caching).


### WP-1b — Training-free MultiBD (lever: more tokens/forward)

Raise `N_buf` 1→2 in the Block-Buffer loop; add τ_add (activation) and τ_semi (semi-completion fallback) to DecodingPolicy; per-slot ActiveBlockCache instances.
- Source numbers on LLaDA2.1-Mini: TPF 4.50→6.50, −0.59pp (math, τ_add=0.10); code needs τ_add=0.90 (**sourced**, [[mbd-lms]]).
- First task: τ_add sweep on *chat/reasoning* prompts (uncharted in the source) — sweep {0.1, 0.3, 0.5, 0.7, 0.9} on the bench suite before trusting any setting.
- Verify TPF→TPS conversion on M2 Ultra: measure step-latency multiplier at N_buf=2 (H100 saw 1.24×); if the M2 Ultra step is more compute-bound, the multiplier will be larger and the net TPS gain smaller — record η_tok and the roofline terms per the source's model.
- Interaction with LLaDA2.1 editing: post-step (`max_post_steps`) refinement runs per active block; confirm edit behavior with two concurrently active blocks matches single-block quality (open question in [[mbd-lms]]).
**Accept**: net TPS gain ≥15% on chat suite at ≤0.5pp quality cost, else record negative result with roofline attribution (compute-bound vs efficiency-bound).

### WP-1 order

1a-Phase-A and 1b are independent (different levers, different code regions: cache policy vs loop control). Build **1b first** (smaller: loop is already buffer-shaped; no new kernels), then 1a while 1b's τ_add sweep runs. 1a's staleness ablations (B/C) come after both.

## 2. Tier 2

### WP-2a — Auto-speculation: Spiffy, benchmarked against S2D2 (lever: more tokens/forward)

Two single-model speculation candidates; implement behind one `SpeculationPolicy` interface and let the bench decide:
- **Spiffy** ([[auto-speculative-decoding]], [[directed-draft-graph]], [[offline-draft-calibration]]): offline-calibrated draft graph (<50 samples), batched draft states (batch dim paid for in Phase 2), lossless verification — *exactly* lossless on block-causal models like LLaDA2.x (**sourced**, [[spiffy]]), but verify under Metal fp16/bf16 numerics.
- **S2D2** ([[self-speculative-decoding-dlm]]): the concept evaluation's strongest single recommendation — the only pre-MBD concept with a number measured on LLaDA2.1-Mini itself.
Head-to-head is an open question in both wiki source notes; we get to answer it.
**Accept**: better of the two lands if ≥20% TPS over the then-current stack; comparison result goes to the wiki either way.

### WP-2b — Streaming-dLLM cluster (levers: cheaper steps + fewer steps)

Three independent sub-items, ablated separately (source Table 3 shows each contributes):
1. Suffix window + final-token positional cue ([[attenuation-guided-suffix-modeling]]) — swaps in behind the suffix-representation interface from Phase 1 §4.2. Interaction warning: with MultiBD, the "suffix" beyond the running-set shrinks in importance; measure with N_buf=2 active.
2. Dynamic confidence threshold τ(t) = τ0·(1 − α(1 − r_mask)) ([[dynamic-confidence-aware-decoding]]) — replaces static τ_mask inside DecodingPolicy; α≈0.6 starting point.
3. Early exit on high-confidence EOS ([[early-exit-block-diffusion]]) — extends the M5 EOS handling to terminate remaining blocks incl. active MultiBD slots.
Source ceiling 68–225× is dominated by long-generation suffix savings; our ≤4k chat profile will see a fraction. Sub-item 2 doubles as the threshold-calibration consolidation from `Optimisations.md` ("Convergence" section) — if it wins, OSDT-style one-shot calibration is only needed to *set τ0/α per domain*, not per-step logic.

## 3. Tier 3 (conditional / later)

- **d²Cache, LocalLeap, FreeDave** — enter only if Tier 1/2 leave a measured gap on their lever (d²Cache competes with WP-1a's winner for the "when to refresh" slot; LocalLeap's bounded-neighborhood decoding competes with WP-2b-2; FreeDave's draft-verification competes with WP-2a). Each is a *replacement* candidate for an occupied slot — run only with a hypothesis for why the incumbent underperforms.
- **Sparse-dLLM / MaskKV eviction** — activate when coding workloads push context >8k; irrelevant at chat scale (KV is ~40 KB/token; §3 of Phase 1).
- **Model scheduling / sandwich schedule** — needs a light-model checkpoint; revisit if a LLaDA2.1 small variant appears.
- **MultiTF checkpoint adoption** — if a MultiTF-post-trained LLaDA2.1 checkpoint is released, swap it in (engine-identical; **sourced**, [[multi-block-teacher-forcing]]); removes WP-1b's residual accuracy cost.

## 4. Cross-cutting track: kernel fusion (lever: cheaper steps)

Profiling-driven, runs alongside tiers; targets in expected-payoff order (**inferred** from architecture, confirm with Instruments/Metal captures on M6 baseline):
1. **Fused MoE dispatch**: router (fp32 sigmoid + group-limited top-k) + gather + expert GEMM + scatter + shared-expert add in minimal dispatches. The [[alpha-moe-megakernel]] notes are the direction; its CUDA-specific claims are unvalidated on Metal.
2. **Attention epilogue fusion**: qk-norm + partial RoPE + SDPA + (WP-1a's drift-test cosine, once landed) in one custom MLX kernel.
3. **Selection-set fusion**: confidence extraction + Γ/Δ construction + state update as one kernel — shrinks the per-step tail of small ops.
4. **Step-loop consolidation**: after 1–3, re-measure loop-control overhead (graph eval + K-step flag readbacks). **This is the §13.1 escape-hatch gate**: if loop overhead >10–15% of step time on the M2 Ultra, spec the raw-Metal ICB decode-loop port; otherwise close the escape hatch and record why.

## 5. Composability matrix (maintain as WPs land)

| | 1a Elastic | 1b MultiBD | 2a Spec | 2b-1 Suffix | 2b-2 Dyn-τ | 2b-3 Early-exit |
|---|---|---|---|---|---|---|
| **1a Elastic** | — | per-slot instances; staleness signals untested with 2 active blocks (**open**) | cache staleness × draft verification untested (**open**, flagged in [[spiffy]] note re eviction) | both shrink attention working set; likely additive (**speculative**) | orthogonal | orthogonal |
| **1b MultiBD** | | — | drafts within one slot: OK; across slots: **open** ([[mbd-lms]] Q) | overlapping intent — suffix beyond running-set only (**open**) | additive (per-slot τ(t)) | composes (source's own early exit covers multi-slot) |
| **2a Spec** | | | — | orthogonal | interacts: threshold changes acceptance dynamics → recalibrate draft graph (**inferred**) | orthogonal |
| **2b-1/2/3** | | | | — | additive (source ablation) | additive (source ablation) |

Rule: a cell marked **open** requires its own mini-ablation before both WPs ship enabled together by default.

## 6. Milestone cadence and reporting

- Each WP: branch → implement → bench (marginal + cumulative) → wiki experiment page → land or record negative.
- Re-freeze the baseline after each tier (M6 re-run) so later WPs measure against reality.
- Quarterly (or per-tier) roll-up into [[neodiffusion-inference-engine]] on the wiki: cumulative TPS/TPF/quality vs the Phase 2 baseline, plus the composability matrix state.
- End state for Phase 3: the escape-hatch gate (§4.4) decided with data, and a served default configuration (mode presets: Q/S × chat/code) documented in the server README.
