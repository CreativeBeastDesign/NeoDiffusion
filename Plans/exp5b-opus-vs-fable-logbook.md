# Exp-5B — Session Prefix Cache, implemented twice: Opus 5.5 vs Fable 5.1

**Status**: PRE-REGISTERED 2026-10-18 (this section written before either agent started).
**Orchestrator**: Claude Opus 5.5 main session (also the evaluator — see §6 for the conflict-of-interest controls).
**Host**: `MacBookPro17,1` (Apple M1, 16 GB), mlx-swift 0.31.6 / MLX core 0.31.1, Swift 6.3.

## 1. Question

Two models get the identical brief (§3) and build the same feature independently. Which produces the
better implementation, at what cost? This is an n = 1 comparison per model on one task: it is a case
study, not a model benchmark, and the write-up must say so.

## 2. Feature (engine track B-2)

Cross-turn reuse of committed 32-token prefix blocks (`SessionPrefixCache`): turn N+1 skips the prefill
forwards for every leading block whose tokens equal a block already committed in turn N.

Correction to `Plans/next-phase-plan-2026-10-18.md` B-2 (made while writing this brief, after reading
`DiffusionEngine+Entry.swift` prefill): prefill commits the prompt **block by block**, each a 32-token
capture forward against the committed prefix — the same computation a generated block's commit performs.
So reuse should be **bit-exact even on the 4-bit artefact**, not merely "mathematically exact". My earlier
"prefill tile shapes differ" caveat came from the *uncached* path and was wrong for this one `[Inferred:
code reading; the hidden real-model test checks it]`.

## 3. Brief (verbatim copy of what both agents receive)

See §8 — the brief text is appended there unchanged at launch, so the exact words are on record.

## 4. Procedure and fairness controls

- Same `subagent_type` (general-purpose), same brief, no effort override, launched in the same message.
- Each runs in its own git worktree created from the same commit (recorded in §7).
- **Compute serialisation**: every `swift build` / `swift test` / bench in either worktree must go through
  `Tools/compute-gate.sh <label> …` (machine-wide lock + ≥ 5000 MB available, retry every 60 s). Gate log
  `/tmp/neodiffusion-compute-gate.log` is the audit trail for queueing time per label (`opus-*`, `fable-*`).
- Neither agent may load the real 9.5 GB model; real-model measurements are run by the orchestrator,
  centrally and identically, after both finish.
- The **hidden evaluation** (`HiddenSessionCacheEvalTests.swift`) was written before launch and kept
  outside the repository (orchestrator scratchpad); it is copied into each worktree only at evaluation
  time. Its sha256 is recorded in §7 before launch.

## 5. Metrics (decided before launch)

| # | Metric | Source | Direction |
|---|---|---|---|
| M1 | Tokens consumed by the agent | task notification `subagent_tokens` | lower better |
| M2 | Tool calls, wall-clock duration | task notification | lower better (secondary) |
| M3 | Gate wait / run time | gate log, per label prefix | descriptive |
| M4 | Hidden toy tests passed (A1–A5) | `swift test --filter HiddenSessionCacheEvalTests` in each worktree, run by orchestrator | higher better — **primary** |
| M5 | Full regression suite in the worktree | `swift test` (orchestrator, gated) — any new failure vs the baseline run is a defect | must be 0 regressions |
| M6 | Real-model bit-exactness (B1) | hidden B1, 3 reps | must be identical |
| M7 | Real-model speed: turn-2 total seconds with vs without session; forwards | hidden B1 JSONL | higher speedup better |
| M8 | Blind code review score | a third agent (Sonnet) sees diffs labelled X/Y only, scores correctness, simplicity, fit with codebase idiom, test quality, docs/provenance; label key opened after scoring | higher better |
| M9 | Diff size (lines added/removed, files) | `git diff --stat` | descriptive |
| M10 | Own tests written + whether they pass | agent's tests | descriptive |

Note on M7: `prefillSeconds` is only meaningful with `instrument: true` (lazy evaluation shifts work),
so the primary speed number is total generation wall-clock and forward count, not `prefillSeconds`.

**Decision rule**: M4 + M5 + M6 decide "correct or not". Among correct implementations, M8 then M7 then
M1 rank them. If both fail M4/M5/M6 somewhere, report failures side by side; no winner is declared.

## 6. Conflict of interest

The orchestrator is Opus 5.5, i.e. the same model family as one contestant. Controls: hidden tests
written before launch; blind review by a third model; all raw outputs (diffs, test logs, JSONL) kept so
André can re-judge. The orchestrator's own opinion is reported separately and labelled as such.

## 7. Launch record

(filled at launch: base commit, worktree paths, hidden-test sha256, launch time)

## 8. Brief (verbatim)

(appended at launch)

## 9. Results

(appended after evaluation; never edits §1–§6)
