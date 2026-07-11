# Overnight Report — 2026-07-10/11 (Claude)

Three tracks per the approved plan. All verdicts from hardware-independent counters (§0.1); the M1 was memory-pressured all night (envValid ~0), so no wall-clock claim below is load-bearing except the explicitly host-scoped multiplier.

## Verdicts, one line each

1. **Serving-path fix — DONE, on `main`** (660a9a7): `generateCached` no longer runs drift instrumentation when elastic is off; parity suite green; bench sanity unchanged counters.
2. **WP-1a Elastic-Cache — CLOSED, negative, now from two independent angles**: the salvage experiment (paper-faithful same-token drift test, branch `wp1a-salvage`) produced output **token-identical to baseline** at γ=0.9 and 0.98 — the faithful signal never clears γ, i.e. the paper's own trigger says "always stale, never reuse". Combined with the ~4–5% FLOP ceiling: no operating point exists. Records: `Plans/elastic-cache-logbook.md` (F1–F9), wiki draft `Plans/wiki-drafts/wp-1a-elastic-cache-active-kv-reuse.md`.
3. **WP-1b MultiBD — IMPLEMENTED + SWEPT, provisional algorithmic ACCEPT** (branch `wp-1b-multibd`): TPF-logical **+23.3% chat (τ_add=0.5) / +22.3% reasoning (τ_add=0.1–0.3)** at gen-128, **+29.5% reasoning at gen-256**; 13/13 parity gates green (incl. cached==uncached with 88 activation events — the dual-block mask is proven); M1 wall-clock does NOT convert (dual-step multiplier ~1.75× > TPF gain — compute-bound), so the roadmap's ≥15% **net-TPS gate defers to the Studio backfill**. Served default stays `nBuf=1`. Records: `Plans/wp1b-logbook.md`, wiki draft `Plans/wiki-drafts/wp-1b-multibd-training-free.md`.

## What needs you

1. ~~Score the WP-1b blind sheet~~ **DONE (2026-07-11 morning): 7/8 ties, 1/8 baseline win, 0 MultiBD wins — smoke-clean.** (The M8 strict-vs-referenceBias sheet `scratch/m8_blind/sheet.md` is still pending.)
2. **Review/merge branches**: `wp-1b-multibd` (the WP; 5 commits, tests green, ready for review) and `wp1a-salvage` (reference implementation of the faithful drift test; recommend keep-unmerged, it's dead code for serving). `main` already has the serving fix + elastic closure docs.
3. **Wiki import**: move the two drafts from `Plans/wiki-drafts/` into `05-Experiments/`; the WP-1b page links [[mbd-lms]] — note the paper is arXiv:2606.29215 and my Algorithm-5 extraction is in the wp1b logbook §1 if the vault note needs updating.
4. **Studio backfill list** (one session, per §0.1.4): re-run `scratch/wp1b_sweep.log`'s recorded arms + the probe commands (logbook §5a) → decides the net-TPS accept gate and the serving presets (proposal: chat τ_add 0.5, reasoning 0.3 at gen-128; single τ_add 0.5 for long-form).

## Notable process events

- **The effective-echo provenance rule caught a silently-void sweep** on its first outing: a zsh word-splitting bug dropped all MultiBD flags; rows echoed `nBuf 1` under nbuf2 arm names; 25 rows scrubbed, driver fixed, re-run. Without the echoes this would have read as "MultiBD has zero effect" (wp1b logbook timeline 9 / F7).
- τ_add/τ_semi semantics are **sourced, not inferred**: I found and deep-read the MBD paper's PDF (Algorithm 5, Appendix C.4, Table 4). τ_semi is *not* a threshold relax — it gates the trailing block's top-1 fallback. The paper's LLaDA2.1-Mini row uses exactly our Q-mode thresholds (τ_M2T 0.70 / τ_T2T 0.50).
- Deviations from the source algorithm are recorded in wp1b logbook §6 (budget-break whole-window revert, generated-position denominators, τ_stable omitted, EOS-gated activation, post-carry on promotion, elastic×MultiBD preconditioned off).
- Improvement loop: cliff localized to (0.6, 0.7) on chat; τ_semi and K=2 probes skipped with recorded rationale; "event-aware K" filed as a future §4-track item (overshoot at K=4 eats honest TPF when events are dense — F6).

## Branch/commit map

- `main`: serving fix + elastic logbook/wiki/roadmap/CLAUDE.md closure (2 commits tonight).
- `wp1a-salvage` (off main): faithful drift test + salvage verdict in logbook §6 (1 commit).
- `wp-1b-multibd` (off main): implementation, tests, bench, logbook, wiki draft, this report (6 commits).
