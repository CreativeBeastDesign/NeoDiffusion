# B-1 — Is K=4 loop speculation still worth it on the dev M1? (pre-registration)

**Registered**: 2026-10-18, before any run. Results go in §5 only; §1–§4 are frozen.

## 1. Why
The server picks `speculationK = 4` on hosts with < 64 GB (CLAUDE.md, final-plan F-l) on the belief that
K>1 "can still pay on the sync-bound M1". Cross-process data from the cache campaign contradict this:
for the same 291 logical steps, K=4 evaluated 374 forwards vs 337 at K=1, at the same ms/forward
(189.0 vs 186.5) `[Sourced: paper/facts/runs R-001 q-cached vs R-003 spk-1; cross-process, so only suggestive]`.
Speculation is output-invariant, so if K=1 is faster on the M1 the default can change without a quality gate.

## 2. Hypothesis
H: on `MacBookPro17,1`, K=1 has higher aggregate TPS than K=4 for the served Q mode with the current
default engine (clean-commit on). Falsified if aggregate K=1 TPS ≤ K=4 TPS, or if the difference is
smaller than the arm-order spread (|Δ| < 3 %), in which case the default stays.

## 3. Procedure
- Build: release `diffusion-bench` from the committed tree (row `gitDirtyFileCount` must be 0 for code files).
- Command (one process, prompt-major interleaving, arm order rotating):
  `Tools/compute-gate.sh b1-spk ./.build/release/diffusion-bench llada --arms spk-1-clean,spk-4-clean --interleave --suites chat,reasoning,code --runs 2 --gen-length 128 --warmup-once --json scratch/2026-10-18/b1_spk_k1_vs_k4.jsonl`
  (`spk-*-clean` = q-cached with clean-commit at fixed K; added to the bench for this run.)
- Exclusions: warmup row; `envValid=false` rows (listed, not silently dropped).
- Output-invariance check: `finalSequence`/text identical between arms for every prompt (if not, that is a bug report, not a result).

## 4. Metrics and decision
Primary: aggregate TPS = Σtokens / Σseconds over valid rows, per arm, plus per-prompt paired ratio
(median and min/max). Secondary: forwards, ms/forward, sync points. Decision: change the M1 default to
K=1 only if H holds on both runs; record either way in `Plans/final_optimisations_plans.md` style.

## 5. Results
(appended after the run)
