# WP-1b MultiBD Logbook — LLaDA2.1-mini, M1 dev host

**Status**: in progress (overnight session 2026-07-10/11, Claude; branch `wp-1b-multibd`)
**Scope**: training-free MultiBD (N_buf 1→2) per roadmap §1 WP-1b, under the §0.1 dev-only protocol: hardware-independent metrics decide; wall-clock host-scoped; every arm a recorded command line for Studio backfill.

## 0. Method in one paragraph

Implement the source paper's Block-Buffer MultiBD (arXiv:2606.29215, Algorithm 5 — **the actual mbd-lms source, deep-read this session**; τ_add/τ_semi semantics are now *sourced*, not inferred) in the existing Block-Buffer-shaped loop: a slot scheduler with up to two concurrently active blocks in one 64-token cached forward under a block-causal active-window mask; activation gated by τ_add on the newest block's decoded fraction; the top-1 fallback of a trailing block gated by the preceding block's τ_semi semi-completion; front-block in-order commit preserving streaming. Gate with internal-consistency parity tests (no nBuf=2 reference trace exists), then sweep τ_add ∈ {0.1,…,0.9} on chat+reasoning (uncharted in the source, which covers math/code only).

## 1. Starting point (sourced)

- Paper, LLaDA2.1-Mini training-free MultiBD (N_buf=2, block 32): TPF 4.50→6.50 (+44%), accuracy 87.22→86.63 (−0.59pp); hyperparameters (Table 4): τ_add 0.10 math / 0.90 code, τ_semi 0.90, τ_M2T 0.70, τ_T2T 0.50 — the M2T/T2T values are exactly our Q-mode thresholds (cross-validation of the reference alignment).
- Paper, LLaDA2-Mini roofline (H100, TP=2): step-latency multiplier 1.24× at N_buf=2; predicted TPS gain TPFgain/multiplier ≈ measured 1.44×. Our M1 multiplier is expected larger (more compute-bound at small batch) — roadmap requires measuring it.
- Algorithm 5 key semantics (verbatim-extracted, see `scratchpad` extraction; recorded here for the wiki):
  - activation: "if b_last satisfies progress > τ_add and stability > τ_stable then activate the first trailing dummy slot" (τ_stable "—"/not applicable for LLaDA2.1-Mini);
  - fallback: "Accept masked positions with confidence > τ_M2T; if no masked position is accepted and the preceding active block is semi-complete then accept the highest-confidence masked position";
  - commit: "if b is complete and all preceding resident blocks are cached or ready-to-cache then mark b as to-cache"; front commits pop in order.
- M6 frozen dev baseline + envValid discipline (`Plans/m6-logbook.md`); elastic-cache campaign provenance lessons (`Plans/elastic-cache-logbook.md` F7) — adopted here as: **JSONL records engine effective echoes, never CLI values**, and hypotheses are pre-registered below *before* the sweep ran.

## 2. Hypotheses (pre-registered 2026-07-11, before the τ_add sweep)

- **H1**: best-τ_add TPF-logical gain on chat+reasoning ≥ +30% vs the nBuf=1 re-freeze (source direction: +44% on math). Reject WP if < +15% with no quality path.
- **H2**: the τ_add optimum is domain-dependent (source: 0.10 math vs 0.90 code); chat is expected to tolerate lower τ_add than code because its continuations are less brittle — **speculative**.
- **H3**: the M1 dual-phase step-latency multiplier lands in 1.3–2.0× (worse than H100's 1.24× — small-batch compute-bound). If multiplier ≥ TPF gain, the M1 TPS verdict is "deferred to Studio", per §0.1 wall-clock scoping.
- **H4**: dual-active editing (Δ over both blocks) does not inflate the front block's steps (the mbd-lms open question); front steps/block at nBuf=2 ≈ nBuf=1 within noise.
- **H5**: trailing starvation (τ_semi gate) is non-zero at low τ_add and shrinks as τ_add grows.

## 3. Timeline

| # | When (CEST) | What | How tested / result |
|---|---|---|---|
| 1 | 23:20 | Paper deep-read: fetched arXiv:2606.29215 PDF, extracted Algorithm 4/5 + Appendix C.3/C.4 + Table 4 via pypdf | τ_add/τ_semi predicates now **sourced**; τ_semi ≠ the roadmap's "semi-completion fallback relaxes thresholds" guess — it gates the trailing block's top-1 fallback |
| 2 | 23:25–05:20 | Implementation on branch `wp-1b-multibd`: `BlockBuffer` nBuf≤2 + `cancel` + front-only settle; `GenerationParams` nBuf/tauAdd/tauSemi; `BlockDiffusionMask.activeWindowMask`; `mask:` threaded through the plain cached attention chain; `DiffusionEngine` rewritten — `run()` slot scheduler, `denoisePhase` (batch-break events, `[2K]` stacked readback), `windowStep` (S∈{1,2}: per-slot Γ with τ_semi-gated fallback, window-wide Δ, per-slot posts, front-only break, τ_add activation flag, `[4S]` stats) | `swift build` clean; commit `wp-1b-multibd` |
| 3 | 05:25 | Unit gates: `testActiveWindowMask` (mask == full-window strict slice, element-checked), single-block degeneration, `BlockBuffer` two-active/cancel/in-order | 3/3 pass |
| 4 | 05:26–05:29 | Parity gates: `swift test --filter "DenoisingLoopParityTests\|MultiBDParityTests"` | **13/13 pass**: 6 existing M5/WP-1a gates unchanged-green (nBuf=1 byte-identical after the rewrite); disabled parity (nBuf=2+τ_add=2.0 ≡ nBuf=1, 16 cases, both paths); **cached==uncached with 88 real activation events** (mask correctness proven against the full-window strict anchor); K-invariance incl. activation timing; sync budget with events; in-order streaming; EOS-cancel; metrics invariant. One test-bug fixed en route ("noeos".contains("eos") filter made the metrics test vacuous — re-ran real, passes) |
| 5 | 05:36 | Full suite | 80 tests, 0 failures (15 skipped, the usual env-gated ones) |
| 6 | 05:40 | Bench: `--n-buf/--tau-add/--tau-semi`; JSONL fields `nBuf/tauAdd/tauSemi/dualActiveSteps/activationSteps/trailingStarvedStepsPerBlock/single+dualActiveDenoiseSeconds`, all **effective engine echoes**; `logicalSteps`/`tpfLogical` switched to the global step counter (block-sum double-counts dual steps); console TPF fixed the same way | smoke run below |
| 7 | 05:45 | Smoke, real 4-bit weights: `diffusion-bench llada --runs 1 --cooldown 0 --suites chat --gen-length 128 --arm nbuf2-smoke --arm-mode q --n-buf 2 --tau-add 0.3` | end-to-end good; peak 9.64 GB (baseline 9.57 — no paging blowup, risk 2 clear); accounting self-consistent (capital: blockSum 29 − logical 21 = dual 8 ✓); early signal: TPF-logical capital 1.86 vs 1.52 baseline, email 2.46 vs 2.06 |
| 8 | 05:50 | **τ_add sweep launched**: `diffusion-bench llada --runs 3 --cooldown 30 --suites chat,reasoning --gen-length 128 --arm {nbuf1-refreeze, nbuf2-tadd01/03/05/07/09} --arm-mode q [--n-buf 2 --tau-add τ] --json scratch/wp1b_sweep.jsonl` (driver: 6 arms sequential, 30 s inter-arm) | log `scratch/wp1b_sweep.log`; results §4/§5 |
| 9 | 06:25 | **Provenance rule catches a driver bug**: the tadd01 arm's rows echoed `nBuf 1, tauAdd 2` — the zsh driver's unquoted `$2` was *not* word-split (zsh default), so the MultiBD flags never reached the binary and 25 rows were baseline duplicates mislabeled as nbuf2 arms. Sweep killed; 25 rows scrubbed by effective-echo filter (`nbuf2* && nBuf==1`); driver fixed (`${=2}`); nbuf2 arms relaunched detached. First re-run row verified: echo `nBuf 2, tauAdd 0.1`, activation at step 0 | **the F7 rule works**: without the echoes this would have been a silently-void sweep, indistinguishable from "MultiBD has zero effect" — the elastic campaign's mislabeled-K failure mode, prevented this time |

## 4. Findings (sweep: 6 arms × chat+reasoning × 4 prompts × 3 runs, gen-128, Q mode, K=4; all counters deterministic across runs; envValid 0/144 — memory-pressured night, wall-clock is indicative only, counters decide per §0.1)

**F1 — TPF-logical gain clears the reject line but lands below the source's math number (H1: partial).** (sourced: `scratch/wp1b_sweep.jsonl`) Best arms: chat **+23.3%** at τ_add=0.5 (TPF 1.79→2.21), reasoning **+22.3%** at τ_add∈{0.1,0.3} (4.77→5.83). Above the ≥+15% floor on both suites; below the paper's +44% (math, 4096-token budget). Likely runway-limited at gen-128 (blocks 4–5, early EOS) — the gen-256 probe (§5a) tests this.

**F2 — The τ_add optimum is domain-dependent, and chat/reasoning differ from both of the paper's domains (H2: confirmed).** (sourced) Chat: flat +21% at 0.1–0.3, peak +23.3% at 0.5, **cliff to +3% at 0.7** (activation arrives too late to matter once the front block is nearly settled — dual% falls 80→35%). Reasoning: best early (0.1–0.3, +22%), gentle decay to +10% from 0.5. The paper's math=0.10 / code=0.90 settings bracket ours; per-mode presets are warranted (serving: chat 0.5, reasoning 0.3 — pending quality).

**F3 — M1 wall-clock is net-negative at the winner; TPS verdict deferred to Studio (H3: confirmed, defer branch).** (sourced, host-scoped) Dual/single per-step latency multiplier (within-run ratio, robust to ambient swap): ~1.75× at chat τ_add=0.5 (paper H100: 1.24×). Weighted: steps ×0.80, cost per dual step ×1.75 over 52% of steps → ≈ ×1.11 net wall-clock on M1 — the M1 is too compute-bound at 64-token windows for the TPF gain to convert. Roofline attribution: compute-bound (multiplier ≥ TPF gain). The roadmap's ≥15% net-TPS accept gate is **formally undecided until the Studio backfill**; expected favorable there (M2 Ultra headroom; H100 analogue converted 1.78×TPF/1.24× → 1.44× TPS).

**F4 — Dual-block interference is real at aggressive activation and vanishes at the winner (H4: refuted at low τ_add, holds at ≥0.5).** (sourced) Front-block steps vs nBuf=1, per-block: median +2.0 at τ_add=0.1, +1.0 at 0.3, **+0.0 at ≥0.5**. Chat edits double (4→8 total) under dual activity — the mbd-lms open question resolves as "editing keeps working; interference shows up as front-block steps only when the trailing block activates very early".

**F5 — Trailing starvation behaves as designed and is negligible at the winner (H5: confirmed).** (sourced) τ_semi-gated fallback denial: 13 steps at τ_add=0.1 → 4 at 0.5 → 0 at ≥0.7 (chat). A τ_semi sweep is **not warranted** at the winning τ_add (≤4 starved steps ≈ ≤2% of steps — nothing to recover; recorded as the C.5(a) skip rationale).

**F6 — TPF-honest diverges from TPF-logical as event density rises; overshoot is the cost of K=4 with events.** (sourced) Reasoning at τ_add=0.1: logical 4.77→5.83 but honest 3.21→2.67 — activation/commit events split batches and every break discards up to K−1 evaluated forwards, and short reasoning blocks make events dense. On hosts where forwards ≈ wall-clock (M1), overshoot eats the logical gain — same compute-bound story as F3. Future direction (not run tonight): event-aware K (drop to K=1–2 when activation/settle is predicted near) — files under the §4 kernel/loop track, not WP-1b.

**F7 — The effective-echo provenance rule caught a silently-void sweep on its first outing** (timeline 9): zsh's unsplit `$2` dropped all MultiBD flags; rows echoed `nBuf 1, tauAdd 2.0` under nbuf2 arm names; 25 rows scrubbed by echo filter and re-run. Without the echoes the sweep would have read as "MultiBD has zero effect".

## 5. Results summary

| Arm (Q, gen-128) | chat tpfLog (Δ) | reasoning tpfLog (Δ) | chat dual% | front-Δsteps (med) | starved |
|---|---|---|---|---|---|
| nbuf1-refreeze | 1.79 (—) | 4.77 (—) | 0 | — | 0 |
| τ_add=0.1 | 2.16 (+20.7%) | 5.83 (+22.3%) | 80% | +2.0 | 13 |
| τ_add=0.3 | 2.16 (+20.7%) | 5.83 (+22.3%) | 70% | +1.0 | 10 |
| **τ_add=0.5** | **2.21 (+23.3%)** | 5.24 (+10.0%) | 52% | +0.0 | 4 |
| τ_add=0.7 | 1.84 (+3.1%) | 5.24 (+10.0%) | 35% | +0.0 | 2 |
| τ_add=0.9 | 1.85 (+3.4%) | 5.23 (+9.6%) | 14% | +0.0 | 0 |

**Overnight verdict (dev-only, §0.1): algorithmic ACCEPT, provisional.** The hardware-independent deciders clear the floor (+23%/+22% TPF-logical, deterministic, parity-gated); M1 wall-clock is net-negative (F3) so the roadmap's net-TPS gate defers to the Studio backfill; quality awaits André's blind scores (`scratch/wp1b_blind/`). Recommended state: **land with `nBuf=1` as the served default** and per-mode presets (chat τ_add 0.5, reasoning 0.3) enabled only after Studio TPS + quality confirm. §5a below adds the improvement-loop probes.

### 5a. Improvement-loop probes (C.5)

*(pending: τ_add=0.6 cliff localization; gen-256 length sensitivity on baseline/0.3/0.5; τ_semi probe skipped per F5; K=2-dual skipped — no paging observed, peak 9.64 GB)*

## 6. Deviations from the source algorithm (recorded per house rules)

1. **Budget-break reverts the whole window**: when the front block exits via the post-steps budget, the trailing block's same-step write is discarded too (simplest K-invariant rule; the reference's "no update on the break iteration" extended window-wide). Trailing re-derives the write in later steps; tokens unaffected in the settle path.
2. **Progress denominators are over generated positions only** (prompt-tail positions are never masked; counting them would make τ_add mean different things in block 0 vs later blocks). The paper does not specify the denominator.
3. **τ_stable not implemented** — "—" (not applicable) for LLaDA2.1-Mini in Table 4.
4. **Activation is EOS-gated only under `eos_early_stop`** (Alg. 4's "EOS has not appeared"): without early stop the engine generates past EOS by design, so activation continues too.
5. **Trailing promotion carries its post-steps counter** (a block's mask-free iterations count while trailing; the reference's per-block budget semantics extended, not reset).
6. **Elastic-Cache × MultiBD**: mutually exclusive by precondition (roadmap §5 cell stays **open**); per-slot ActiveBlockCache instances are *not* plumbed — moot while WP-1a is closed as a negative result.
