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

## 4. Findings

*(to be filled from the sweep — hypotheses H1–H5 above decide)*

## 5. Results summary

*(pending sweep completion)*

## 6. Deviations from the source algorithm (recorded per house rules)

1. **Budget-break reverts the whole window**: when the front block exits via the post-steps budget, the trailing block's same-step write is discarded too (simplest K-invariant rule; the reference's "no update on the break iteration" extended window-wide). Trailing re-derives the write in later steps; tokens unaffected in the settle path.
2. **Progress denominators are over generated positions only** (prompt-tail positions are never masked; counting them would make τ_add mean different things in block 0 vs later blocks). The paper does not specify the denominator.
3. **τ_stable not implemented** — "—" (not applicable) for LLaDA2.1-Mini in Table 4.
4. **Activation is EOS-gated only under `eos_early_stop`** (Alg. 4's "EOS has not appeared"): without early stop the engine generates past EOS by design, so activation continues too.
5. **Trailing promotion carries its post-steps counter** (a block's mask-free iterations count while trailing; the reference's per-block budget semantics extended, not reset).
6. **Elastic-Cache × MultiBD**: mutually exclusive by precondition (roadmap §5 cell stays **open**); per-slot ActiveBlockCache instances are *not* plumbed — moot while WP-1a is closed as a negative result.
