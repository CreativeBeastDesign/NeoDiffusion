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

## 5. Results (appended 2026-10-18; §1–§4 unchanged)

`[Sourced: scratch/2026-10-18/real/b1_spk_k1_vs_k4.jsonl, ingested as paper/facts/data/2026-10-18-real/b1_spk_k1_vs_k4.jsonl
sha256 cfd2b2e2…; run by André with the Claude app quit, 18:55–19:02 CEST]`
Host `MacBookPro17,1`; mlxCoreVersion 0.31.1 / mlx-swift 0.31.6; **committed tree** `ea40e92`, dirty = 0 (§3 met);
`schedule = interleaved`; 48 non-warmup rows (12 prompts × 2 arms × 2 runs), warmup row excluded.

| Arm | Agg. TPS | Tokens | Seconds | Forwards | Logical steps | ms/forward |
|---|---|---|---|---|---|---|
| spk-1-clean (K=1) | **16.52** | 2,666 | 161.3 | 876 | 852 | 184.2 |
| spk-4-clean (K=4) | 14.45 | 2,666 | 184.5 | 1,000 | 852 | 184.5 |

Paired per prompt (24 pairs): K1/K4 TPS ratio median **1.122**, min 1.044, max 1.368; **K=1 faster on 24/24**;
text identical on 24/24 (output invariance holds). Aggregate **+14.3 %** for K=1. Mechanism: same steps (852) and
same ms/forward; K=4 evaluates 124 extra (overshoot) forwards (+14.2 %).

**Validity problem — reported before the verdict**: **0 of 48 rows are `envValid`** under the frozen rule. Every
row fails only the free-page floor (free 79–653 MB < 1,024 MB); swap growth ≤ 256 MB on all rows (1,077 → 1,390 MB
over the whole run) and thermal `nominal` throughout. ms/forward (184 ms) matches the unloaded M6 figure
(~0.19 s), i.e. no paging pathology (which shows as ~50 s/forward). With the 9.6 GB model resident, free pages on
this 16 GB host sit below 1 GB by construction, so the free-floor rule cannot be met here at all.

**Verdict under the registered rule: NO DECISION** (no valid rows). The evidence (24/24 paired wins, interleaved,
identical output, unchanged ms/forward, flat swap) is strong, but changing the M1 default is André's call — see
the decision list in the session report: either (a) amend the validity rule for the M1 (e.g. swap growth + an
"available incl. reclaimable cache" floor instead of free pages) and re-judge, or (b) accept this run as an
explicit exception.

## 6. Decision and amendments (2026-10-18, André)

- **Decision**: option A — amend the validity rule, re-judge, and set **K=1 on every host** (server
  `hostAwareDefaultSpeculationK` → 1; bench default `--speculation-k` → 1).
- **envValid v2** (`.agents/AGENTS.md`, `EnvSnapshot.memoryFloorMet`): on hosts ≤ 32 GB the 1/16-RAM floor
  applies to *available* memory (free + inactive + speculative + purgeable); larger hosts keep the free-page
  floor. Rows now carry `env.availableMemoryMBBefore`, `envValid` (v2) and `envValidV1`.
- **Re-judgement of this run under v2**: the B-1 rows predate the `availableMemoryMBBefore` field, so the v2
  memory floor cannot be evaluated for them; the remaining v2 conditions (swap growth ≤ 256 MB, thermal
  nominal/fair) hold on **48/48** rows. Re-judged as **valid-partial (v2 without the memory floor)**. André's
  decision ("amend, re-judge, set K=1") was taken with this limitation stated in the report; the next
  run with v2 rows (E-001, all arms at K=1) is the first fully v2-valid evidence.
- **Correction (found while preparing E-001)**: the script passed `--text-out`, which was not a bench flag
  (the real one is `--dump-text`) and was silently ignored, so **no full-text dump exists** for this run. The
  "text identical on 24/24" above compares `textPrefix` (first 160 characters) only. K is output-invariant by
  construction and the BF16/FP32 parity tests cover it, so the verdict stands, but the 24/24 claim is a
  prefix check. Fixes: `--text-out` is now an alias, and the bench refuses unknown flags
  (`rejectUnknownFlags`, override `--allow-unknown-flags`).
