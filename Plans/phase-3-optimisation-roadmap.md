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

**STATUS (2026-07-11, `Plans/wp1b-logbook.md`): implemented + M1-swept on branch `wp-1b-multibd` — provisional algorithmic ACCEPT.** TPF-logical +23.3% chat (τ_add=0.5; cliff in (0.6, 0.7)) / +22.3% reasoning (τ_add 0.1–0.3) at gen-128, +29.5% reasoning at gen-256 (runway-limited at 128; both suites converge on τ_add≈0.5 at 256). τ_add/τ_semi semantics sourced from the paper (arXiv:2606.29215 Alg. 5 / Table 4; τ_semi gates the trailing block's top-1 fallback). M1 wall-clock does not convert (dual-step multiplier ~1.75× — compute-bound), so the net-TPS gate above is decided by the Studio backfill; served default stays nBuf=1 meanwhile. Quality: blind-scored by André (2026-07-11): 7/8 ties, 1/8 baseline win, 0 MultiBD wins — smoke-clean, consistent with the source's −0.59pp; formal ≤0.5pp gate goes to the Studio scored set.

### WP-1 order

1a-Phase-A and 1b are independent (different levers, different code regions: cache policy vs loop control). Build **1b first** (smaller: loop is already buffer-shaped; no new kernels), then 1a while 1b's τ_add sweep runs. 1a's staleness ablations (B/C) come after both.

## 2. Tier 2

### WP-2a — Auto-speculation: Spiffy, benchmarked against S2D2 (lever: more tokens/forward)

Two single-model speculation candidates; implement behind one `SpeculationPolicy` interface and let the bench decide:
- **Spiffy** ([[auto-speculative-decoding]], [[directed-draft-graph]], [[offline-draft-calibration]]): offline-calibrated draft graph (<50 samples), batched draft states (batch dim paid for in Phase 2), lossless verification — *exactly* lossless on block-causal models like LLaDA2.x (**sourced**, [[spiffy]]), but verify under Metal fp16/bf16 numerics.
- **S2D2** ([[self-speculative-decoding-dlm]]): the concept evaluation's strongest single recommendation — the only pre-MBD concept with a number measured on LLaDA2.1-Mini itself.
Head-to-head is an open question in both wiki source notes; we get to answer it.
**Accept**: better of the two lands if ≥20% TPS over the then-current stack; comparison result goes to the wiki either way.

**STATUS (2026-07-11, `Plans/wp2a-logbook.md`, wiki draft `Plans/wiki-drafts/wp-2a-auto-speculation.md`): closed on the dev host — gate NOT met vs Q-mode, mechanistically.** S2D2 fully implemented (verifier ≡ sequential-AR anchor test; acceptance 4–10 tok/verified step) and landed **default-off**; the break-even identity (acc/vstep > 2× baseline tokens/step) shows Q-mode's aggressive thresholding already banks the parallelism — causally confirmed by the τ=0.95 arms, where S2D2 converts (+15–19% width-corrected). Surviving candidates for the Studio session: (a) the **verified-conservative preset** (τ=0.95 + S2D2; quality sheet `scratch/wp2a_blind/` pending André), (b) **Spiffy runtime** — calibrated ceiling readout says 27–35% forward-count savings at D=3–8 (`scratch/draft_graph.json`), unaffordable at (1+D)× width on the M1, decided by the Studio's wide-forward multiplier. Third instance of the baseline-relativity lesson (WP-1a: prefix cache; here: threshold decoding).

### WP-2b — Streaming-dLLM cluster (levers: cheaper steps + fewer steps)

Three independent sub-items, ablated separately (source Table 3 shows each contributes):
1. Suffix window + final-token positional cue ([[attenuation-guided-suffix-modeling]]) — swaps in behind the suffix-representation interface from Phase 1 §4.2. Interaction warning: with MultiBD, the "suffix" beyond the running-set shrinks in importance; measure with N_buf=2 active.
2. Dynamic confidence threshold τ(t) = τ0·(1 − α(1 − r_mask)) ([[dynamic-confidence-aware-decoding]]) — replaces static τ_mask inside DecodingPolicy; α≈0.6 starting point.
3. Early exit on high-confidence EOS ([[early-exit-block-diffusion]]) — extends the M5 EOS handling to terminate remaining blocks incl. active MultiBD slots.
Source ceiling 68–225× is dominated by long-generation suffix savings; our ≤4k chat profile will see a fraction. Sub-item 2 doubles as the threshold-calibration consolidation from `Optimisations.md` ("Convergence" section) — if it wins, OSDT-style one-shot calibration is only needed to *set τ0/α per domain*, not per-step logic.

**STATUS (2026-07-11, `Plans/wp2b-logbook.md`, wiki draft `Plans/wiki-drafts/wp-2b-streaming-dllm.md`): closed on the dev host — one accept, one N/A, one null.** Sub-item 1 CLOSED N/A without implementation: block-causal LLaDA2.x never computes a suffix (the reference forwards `x[:, :current_window_end]`; verified) — nothing to prune, no signal for the positional cue (fourth baseline-relativity instance; the Phase-1 §4.2 "suffix representation interface" anticipated a reference that forwards suffix). Sub-item 2 **provisional algorithmic ACCEPT at α=0.6** (landed default-off as `dynamicTauAlpha`): logical steps −14.8% chat / −11.3% reasoning at Q mode; mechanism = the late-block fallback tail (the first Phase-3 lever the baseline had NOT banked); additive with MultiBD (nBuf=2 τ_add=0.5 + α=0.6: cumulative TPF-logical +41%/+40%); the τ0=0.9 probe is dominated — consolidation verdict: per-step τ(t) beats any single recalibrated τ0, one-shot calibration only sets (τ0, α). Blind sheet `scratch/wp2b_blind/` pending André; Studio decides quality gate + wall-clock + default-on. Sub-item 3 implemented + tested but a **mechanistic NULL** (byte-identical rows, 0/8 text changes): post-EOS padding clears τ_mask early while the boundary EOS settles last — threshold decoding pre-banks the paper's saving (fifth instance); landed default-off, no τ_eos knob (André's condition: text-change rate was zero). τ_edit stays static; dynamic-τ_edit follow-up dropped (entry condition unmet, post-steps/block 1.16 at the winner).

## 3. Tier 3 (conditional / later)

- **d²Cache, LocalLeap, FreeDave** — enter only if Tier 1/2 leave a measured gap on their lever (d²Cache competes with WP-1a's winner for the "when to refresh" slot; LocalLeap's bounded-neighborhood decoding competes with WP-2b-2; FreeDave's draft-verification competes with WP-2a). Each is a *replacement* candidate for an occupied slot — run only with a hypothesis for why the incumbent underperforms.
- **Sparse-dLLM / MaskKV eviction** — activate when coding workloads push context >8k; irrelevant at chat scale (KV is ~40 KB/token; §3 of Phase 1).
- **Model scheduling / sandwich schedule** — needs a light-model checkpoint; revisit if a LLaDA2.1 small variant appears.
- **MultiTF checkpoint adoption** — if a MultiTF-post-trained LLaDA2.1 checkpoint is released, swap it in (engine-identical; **sourced**, [[multi-block-teacher-forcing]]); removes WP-1b's residual accuracy cost.
- **JOT token-level early stopping** (deferred 2026-07-11, André; source summary `Resources/possibly new/just-on-time-jot.md`) — freeze converged tokens and skip their per-token compute. Unlike the closed WP-1a, this has a **real ceiling on our stack**: the MoE is per-token, so ~70% of layer FLOPs are skippable for frozen positions (vs Elastic-Cache's ~5% KV-projection share). Known conflict: freezing vs the Δ-editing quality mechanism. **Entry condition (cheap, run before any implementation)**: from WP-2a's `--dump-traces` dumps, measure (a) the fraction of positions prediction-stable k steps before their block settles and (b) how often Δ later edits a would-have-been-frozen position. High (a) + low (b) ⇒ promote to a WP next to d²Cache with a real hypothesis; otherwise record and drop. **MEASURED (2026-07-11, 50-prompt calibration traces, `Tools/build_draft_graph.py --jot-k`)**: k=2 → 44.2% stable / 4.9% collisions; k=4 → 15.0% / 4.8%. Verdict: entry condition met at shallow windows — a scoped JOT WP (freeze-after-2-stable-steps, per-position MoE skip, ~30–40% per-token FLOP ceiling × 44% coverage) is justified when a Tier-3 slot opens. Note: the other 2026-07-11 additions to `Resources/possibly new/` (streaming-dllm, OSDT, LocalLeap, MaskKV) map to existing roadmap entries (WP-2b, WP-2b-2, Tier-3) — no roadmap change.

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
| **1b MultiBD** | | — | drafts within one slot: OK; across slots: **open** ([[mbd-lms]] Q) | moot — 2b-1 closed N/A (WP-2b §2) | additive (**measured**, WP-2b F6: −14.7% steps on top of MultiBD) | null × anything = null (**measured**, WP-2b F5/F6) |
| **2a Spec** | | | — | orthogonal | interacts: threshold changes acceptance dynamics → recalibrate draft graph (**inferred**; combination forbidden by engine precondition, untested) | orthogonal |
| **2b-1/2/3** | | | | — | additive (source ablation) | additive (source ablation) |

Rule: a cell marked **open** requires its own mini-ablation before both WPs ship enabled together by default.

## 6. Milestone cadence and reporting

- Each WP: branch → implement → bench (marginal + cumulative) → wiki experiment page → land or record negative.
- Re-freeze the baseline after each tier (M6 re-run) so later WPs measure against reality.
- Quarterly (or per-tier) roll-up into [[neodiffusion-inference-engine]] on the wiki: cumulative TPS/TPF/quality vs the Phase 2 baseline, plus the composability matrix state.
- End state for Phase 3: the escape-hatch gate (§4.4) decided with data, and a served default configuration (mode presets: Q/S × chat/code) documented in the server README.
