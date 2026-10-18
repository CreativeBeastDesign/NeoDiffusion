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

- Launch time: 2026-10-18T13:17Z (both agents launched in one message).
- Base commit: `5f05bf9` (clean-up commit; baseline suite 156 executed / 24 skipped / 0 failures on the tree it was made from).
- Label assignment by coin flip (`secrets.randbelow(2)` = 0 → rule "0: Opus = impl-x"): **impl-x = Opus 5.5 (`claude-opus-5-5`), impl-y = Fable 5.1 (`claude-fable-5-1`)**. The blind reviewer sees only "X"/"Y" diffs with notes-file names neutralised.
- Hidden evaluation `HiddenSessionCacheEvalTests.swift` sha256 `76e64fe39953ac59219dcefc65fac636328d4b9d6f453ef1bd00ee3476ffce74` (kept in the orchestrator scratchpad; copied to `Plans/exp5b-artifacts/` after evaluation).
- Brief sha256 `2f42915df9c9db6f4f56d1b7f224538910e7eb7341369cf7cc06155b0a1e87b8` (template; `{LABEL}` substituted with impl-x / impl-y).
- Worktree paths: recorded when the agents report.

## 8. Brief (verbatim)

````markdown
You are implementing one engine feature in the Swift/MLX repository NeoDiffusion (block-diffusion LLM inference on Apple Silicon). You are working in your OWN git worktree (your current working directory). Your implementation label is **{LABEL}** — use it wherever this brief says {LABEL}. Another agent may be implementing the same feature elsewhere; you will not see its work and must not look for it.

## Read first
- `CLAUDE.md` (project rules, gotchas) and `.agents/AGENTS.md` (provenance tags sourced/inferred/speculative, paranoid logging).
- `Packages/DiffusionGeneration/Sources/DiffusionEngine+Entry.swift` (`generateCached`, prefill loop, `captureAndCommit`, `onSettled`), `ExactPrefixCache.swift`, `Packages/DiffusionCore/Sources/LayerKVCache.swift`, `KVCacheCompression.swift`, `GenerationParams.swift` (note `cleanCommitSkipEligible`), `DiffusionEngine.swift` (`Output`, `Metrics`, `RunStats`), `Tools/diffusion-server/Sources/main.swift`.

## HARD machine rules (16 GB laptop, shared with other work)
1. **Every** `swift build`, `swift test`, `Tools/seed-metallib.sh` and any other heavy command must be wrapped: `Tools/compute-gate.sh {LABEL}-<short-purpose> <command…>` (e.g. `Tools/compute-gate.sh {LABEL}-build swift build --build-tests`). The gate serialises heavy work machine-wide and may make you wait minutes — that is expected; do not bypass it, do not edit it, do not run heavy commands in the background to dodge it.
2. Never load the real model (`models/…`, ~9.5 GB). Toy FP32 fixtures only. The real-model evaluation is done by someone else afterwards.
3. Your worktree has no `scratch/` (git-ignored). Test fixtures live in the main checkout: export `NEODIFFUSION_LOOP_FIXTURE_DIR=/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/loop_fixtures` for tests (the loop-parity fixtures). Do not write into the main checkout.
4. After building, run `Tools/compute-gate.sh {LABEL}-seed Tools/seed-metallib.sh` once (this toolchain's `swift build` does not produce the MLX metallib; tests crash without it). Never partially delete `.build`.
5. Never read a piped exit code as a result: `cmd > log 2>&1; echo $?`, then grep the log for `Test Suite 'All tests'` / `Executed N tests, with M failures`.
6. Run the full `swift test` at most twice (it takes ~6–8 min plus gate waits); iterate with `--filter`.

## Feature: cross-turn session prefix cache
Today `generateCached` builds a fresh `ExactPrefixCache` per call and prefills the prompt block by block (one 32-token capture forward per full prompt block). In multi-turn chat, turn N+1's prompt starts with the same tokens as turn N's prompt + answer, so those prefill forwards are wasted. Implement reuse of committed blocks across calls.

### Required public API (exact — an external test suite will compile against it)
```swift
// in DiffusionGeneration
public final class SessionPrefixCache {
    public init()                                   // you may add other inits/options with defaults
    public var cachedTokenCount: Int { get }       // tokens whose K/V are currently held
    public func reset()                            // drop everything
}
extension DiffusionEngine {
    // New overload / parameter. Calls without `session` must behave exactly as today.
    public func generateCached(prompt: [Int], params: GenerationParams,
                               iceTemplate: [Int]? = nil,
                               streamBlock: (([Int]) -> Void)? = nil,
                               session: SessionPrefixCache?) -> Output
}
// DiffusionEngine.Output.Metrics gains:
public let sessionReusedTokens: Int   // prompt tokens whose prefill was skipped this call (0 without a session)
```

### Required behaviour
- **Exactness**: with a session, `finalSequence` must be identical to the same call without a session — in FP32 and on the 4-bit artefact. Reason it out from the code: a reused block's K/V must be exactly what this call's prefill would have computed.
- Reuse only whole `blockLength`-token blocks, only the longest leading run whose tokens equal the session's committed tokens, and never beyond the new prompt's full blocks (`prompt.count / B` blocks — the prompt tail shares a block with generation and is not prefilled).
- After a call, the session holds this call's committed blocks (prompt blocks + generated blocks, as committed), so turn N+2 can reuse turn N+1's answer if the client resends it.
- Never reuse across incompatible configurations: at least block length, KV quantization mode, and anything else you find that changes committed K/V (think: mask semantics, the model instance, features that commit partial blocks such as sub-block commit, nBuf≠1/MultiBD commits). When incompatible, start the session afresh (or bypass it) — decide, document, test.
- Truncation of a diverged tail must work for every storage form `LayerKVCache` can hold (raw FP16, pool, `CompressedKVTensor` incl. the per-channel `kivi4` layout).
- Fixed-size, sync-free hot path rules from CLAUDE.md still apply (no new per-step CPU readbacks). Any added host readback must be per call, not per step, and stated.
- Memory: say what a session costs (bytes/token) and give it a bound or explain why none is needed.
- `Metrics.sessionReusedTokens` must be an *effective* echo (what actually happened).

### Integration
- `Tools/diffusion-server`: opt-in `--session-cache` (+ env `NEODIFFUSION_SESSION_CACHE=1`), default OFF. One session for the single-request server; log per request how many tokens were reused. Keep the server behaviour unchanged when the flag is off.
- Optional: a `diffusion-bench` hook is NOT required.

### Tests (yours)
Add XCTests under `Tests/DiffusionGenerationTests/` using the toy loop fixtures (pattern: `DenoisingLoopParityTests.setUpWithError`). Cover at least: multi-turn exactness at K=1 and K=4, partial/diverged prefixes, incompatible params, quantized KV (k8v4 and kivi4), reset. Then make sure the existing suite still passes.

### Notes (paranoid)
Write `Plans/exp5b-notes-{LABEL}.md`: design, every decision with its reason, provenance tags, what you tested (exact commands + the real "Executed N tests, with M failures" lines), known limitations/risks, and anything in the brief you think is wrong. Do not edit other Plans docs.

### Finish
Commit your work on your worktree's branch (clear message; end with `Co-Authored-By: <your model name> <noreply@anthropic.com>`). Do not push. Final report (≤ 400 words): files changed, API as implemented, test results (exact lines), limitations, and your own estimate of anything left undone.
````

## 9. Results

### 9.1 Deviations from the registered procedure (recorded before reading any evaluation output)

- **D1 — hidden tests A3/A4 changed (v1 → v2).** The toy fixture has `head_dim = 32`; MLX rejects g64
  quantization on it, so `k8v4`/`kivi4` cannot run end-to-end on the toy model (found by impl-y, confirmed
  by the shape). v1 (sha256 `76e64fe3…`) would have crashed the test process for both implementations.
  v2 (sha256 `92bd5ac0a99a74cabad81b08b13dda6b12aaa4c295c6303f212817a0fa08a0c5`) uses `fp4g32` in A3/A4 and
  adds **B2**: a real-model `k8v4` two-turn exactness check. The change was made after both agents had
  finished but before any evaluation run, and applies identically to both. v1 is kept as
  `HiddenSessionCacheEvalTests.v1.swift` in the artefacts.
- **D2 — worktree base.** The Agent tool cut both worktrees from `main` (`16794ec`), not from the
  pre-registered base `5f05bf9` (branch `kernel`). Both agents noticed independently, could not `reset`
  (sandbox), and created a fresh branch at `5f05bf9`: impl-x → `exp5b-impl-x` @ `7f125f0`;
  impl-y → `impl-y-session-prefix-cache` @ `5c47d50`. Both evaluated from those commits.
- **D3 — information exposure (impl-x).** To find the right base, impl-x read this logbook in the main
  checkout, which at that time contained §7 (label mapping and the v1 hidden-test sha256 — not its
  contents). impl-y reports deriving the base from the main checkout's branch (`kernel`); it does not
  report reading this file. No evidence either agent read the hidden test file (scratchpad, outside repo).
- **D4 — shared scratchpad.** Both agents wrote logs to the same orchestrator scratchpad directory
  (`full1.log` from impl-x, `logs/build1.log` from impl-y), so cross-reading was possible in principle.
  Neither report mentions the other's files.

### 9.2 Process metrics (M1–M3) `[Sourced: task notifications; Tools/analysis/compute_gate_report.py]`

| | impl-x (Opus 5.5) | impl-y (Fable 5.1) |
|---|---|---|
| Agent tokens (`subagent_tokens`, final notification) | **250,637** | **271,708** |
| Tool calls | 64 | 99 |
| Wall-clock duration | 1,601 s (26.7 min) | 2,834 s (47.2 min; includes a lingering `tail -f` monitor it left running, killed by the orchestrator) |
| Gated commands (n, total ran, total waited) | 8, 1,062 s, 0 s | 6, 880 s, 540 s |
| Gated commands with nonzero rc | 4 (build error, test failures while iterating, an intentional mutation run, full suite with 1 env failure) | 2 (first quantized test crash on head_dim 32, full suite with 1 env failure) |
| Diff vs base (files, +/−) | 10 files, +960 / −9 | 12 files, +1,108 / −19 |

Notes: impl-y's 540 s of gate waiting was spent behind impl-x's full test run (it built later because it
wrote more code before its first build); wall-clock therefore partly reflects queueing, not only model speed.
Token counts include reading the same repository context; they are single samples.


### 9.3 Blind review (M8) `[Sourced: Sonnet reviewer report, 2026-10-18; reviewer saw anonymised diffs A/B only]`

Assignment by a second coin flip (`secrets.randbelow(2)` = 0 → **A = impl-x (Opus), B = impl-y (Fable)**),
key written to the scratchpad before the reviewer started and opened only after its report arrived.
Model names and labels were scrubbed from the diffs (`sed`; residual-mention count 0). The reviewer did not
build or run anything and skimmed the server hunks and test bodies.

| Criterion (1–5) | A = impl-x | B = impl-y |
|---|---|---|
| Correctness | 4 | 3.5 |
| Completeness vs spec | 4 | 4.5 |
| Simplicity | 3 | 4 |
| Idiom fit | 4 | 4 |
| Test quality | 5 | 3.5 |
| Docs / provenance | 4 | 4.5 |
| **Overall** | **tie leaning A, confidence 2/5** | |

Reviewer's deciding reasons: A's tests drive k8v4 **and** kivi4 end-to-end through `generateCached` (it built a
seeded head_dim-64 toy for that) and include a mutation check; A is thread-safe by construction (lock +
checkout). B mirrors the spec's incompatibility list more literally (bypasses nBuf≠1, FlashBlock, dual,
Elastic, ablation, ICE via `GenerationParams.sessionReuseEligible`), has a default memory bound and is simpler,
but its quantized coverage is fp4g32-only end-to-end and its bound drops the *whole* session when exceeded
(medium defect: a prefix could be kept). A allows FlashBlock/dual/Elastic/nBuf=2 with a session, argued only
for nBuf=2 (medium). Both full suites fail only `testTokenizerParity` (missing fixture in worktrees).

### 9.4 Evaluation incidents (2026-10-18, 16:05–17:25 CEST) — recorded before any hidden-test verdict

1. **Hidden A1 hung for BOTH implementations** (impl-x killed after ~36 min; impl-y timed out at 600 s).
   Diagnosis `[Sourced: scratch/2026-10-18/diagA1.log, diagA1-nosession.log, repro-hang.log, repro-hang-sample.txt]`:
   every call in A1 takes < 1 s until case `p20_s_noeos` (S mode, τ_edit = 0.0), turn 2. That turn-2 prompt
   (88 tokens) **also hangs with no session at all**, and it **reproduces on the unmodified base engine**
   (main checkout, `DenoisingLoopParityTests/testReproSModeHang`, env-gated, 2 min 19 s at 100 % CPU before kill,
   expected < 1 s). So the hang is a **pre-existing engine bug**, not an implementation defect.
   Mechanism `[Inferred from code + reference]`: the per-block loop only counts post-steps on mask-free
   iterations, and nothing prevents the sampler from emitting `mask_id` as x0 — so a fallback "transfer" can
   write `[MASK]` back into a masked slot, and with τ_edit = 0.0 an edit can re-mask a decoded slot. The block
   is then never mask-free, `max_post_steps` never starts counting, and there is no global step cap. The
   reference `generate()` (models/llada2-1-mini/modeling_llada2_moe.py:1352–1425) has the same structure
   (`while True`, no mask-logit suppression), so this is **reference-faithful latent non-termination**,
   triggered here by the random-weight toy model; on the real model it needs `[MASK]` to win an argmax, which is
   unlikely but not excluded. → New engine item (André's decision, see report): a parity-neutral safety cap
   (e.g. steps/block ≤ 4·B + max_post_steps, logged when hit) and/or suppressing `mask_id` in x0.
   **Consequence for this experiment (deviation D5)**: A1 is re-run on Q-mode cases only (S-mode cases with
   τ_edit = 0.0 can hang the base engine for any implementation). Applied identically to both.
2. **impl-y real-model run invalidated** after 7.5 min: macOS paging (swap 970 MB → 5.4 GB, ~60 MB free) —
   the documented CLAUDE.md pathology. Killed; no rows used. The gate's 5 GB floor is too low for a 9.5 GB
   model load on this host while the Claude desktop app holds ~2 GB; real-model gated jobs need a higher floor.

### 9.5 Correctness results (M4, M5, M10) `[Sourced: scratch/2026-10-18/exp5b-eval-summary.log + per-test logs]`

Hidden toy tests, hidden file **v3** (sha256 `5978cdf3cc77d360ee68caa3084e91f956284231b6e6c727bbb59b9a8453c49c`; A2–A5
byte-identical to v2; only A1 restricted to Q-mode, deviation D5). Each test run separately under the gate with
a 600 s timeout.

| Hidden test | impl-x (Opus) | impl-y (Fable) |
|---|---|---|
| A1 two/three-turn exactness + reuse, K=1 & K=4 (Q-mode cases) | pass (35.2 s) | pass (36.1 s) |
| A2 divergence + unrelated prompt | pass (38.5 s) | pass (39.7 s, run under v2 — identical test) |
| A3 incompatible params → no reuse | pass | pass |
| A4 quantized (fp4g32) session exactness | pass | pass |
| A5 reset + one forward saved per reused block | pass | pass |
| **Total** | **5/5** | **5/5** |

Full regression suite in each worktree (orchestrator-run, gated, `scratch/` linked to the main checkout so
fixture tests run; baseline before the feature: 156 executed / 24 skipped / 0 failures):

| | impl-x | impl-y |
|---|---|---|
| Kernels / Generation / Core | 1 / 105 (11 skip) / 58 (13 skip) | 1 / 114 (11 skip) / 58 (13 skip) |
| Total executed, skipped, failures | **164, 24, 0** | **173, 24, 0** |
| New tests vs baseline | +8 (`SessionPrefixCacheTests`) | +17 (`SessionPrefixCacheTests`) |

Incident: impl-y's first orchestrator regression run reported 58 skips because impl-y had created its own
`scratch/` holding only the tokenizer fixture, so the driver did not link the full fixture set; that run was
discarded as unequal coverage and repeated with the full link (above). Both agents' *own* full-suite runs had
reduced coverage for the same reason (26 + 32 skips) and the same single environmental failure
(`testTokenizerParity`, fixture missing) — neither agent's self-reported suite was equivalent to the baseline.

### 9.6 Real-model results (M6, M7) — DEFERRED

Not run: a 9.6 GB model load next to the Claude desktop app (~2.5 GB resident) pages on this 16 GB host
(§9.4 item 2). Scripted for André to run with the app quit: `caffeinate -i Tools/run-real-model-evals.sh`
(gate floor 10.5 GB; also runs B-1). Results will be appended as §9.7 from `scratch/2026-10-18/real/`.

### 9.7 Interim verdict (toy-level only; real-model gates M6/M7 still open)

- **Correctness**: tie — both 5/5 hidden, both 0 regressions at equal coverage.
- **Blind review (M8)**: tie leaning impl-x (Opus), reviewer confidence 2/5.
- **Cost (M1)**: impl-x 250,637 tokens / 64 tool calls / 26.7 min; impl-y 271,708 / 99 / 47.2 min
  (≈ 8 % more tokens, ≈ 55 % more tool calls; wall-clock inflated by 9 min of gate queueing behind impl-x).
- **Differences that matter for merging** `[Inferred: reviewer + orchestrator reading]`:
  impl-x — stronger quantized end-to-end tests (built a head_dim-64 toy for k8v4/kivi4), lock-based
  thread safety, mutation check; but allows sessions with FlashBlock/dual/Elastic/nBuf=2 with only nBuf=2
  argued, and has no default memory cap. impl-y — spec-literal bypass allowlist
  (`GenerationParams.sessionReuseEligible`, mirrors `cleanCommitSkipEligible`), default 8192-token cap,
  sync-point test; but drops the whole session when the cap is exceeded, and quantized k8v4/kivi4 only at
  storage level.
- **Orchestrator's own opinion (labelled; same model family as impl-x — weigh accordingly)**: functionally
  equivalent on everything measured. For merging I would take impl-y's eligibility allowlist + cap and
  impl-x's head_dim-64 quantized E2E tests and checkout/lock design. Neither is clearly better; n = 1.
- **Process note worth more than the verdict**: the hidden test found a *pre-existing* engine
  non-termination bug (§9.4) that neither implementation's own tests could have revealed, and both agents
  independently caught the toy head_dim limitation (impl-y reported it explicitly).

### 9.8 Real-model results (M6, M7) — run by André, 2026-10-18 18:46–18:54 CEST, Claude app quit
`[Sourced: scratch/2026-10-18/real/SUMMARY.txt, impl-{x,y}-real.log; rows ingested as paper/facts/data/2026-10-18-real/exp5b_session_perf.jsonl]`
Host `MacBookPro17,1`, 4-bit artefact, K=1, Q mode, clean-commit on, gen 64; turn-1 prompt 411 tokens
(repeated paragraph), turn 2 = turn 1 + answer + follow-up; 3 reps per implementation, arm order alternated.

| | impl-x (Opus) | impl-y (Fable) |
|---|---|---|
| `testB1` (XCTest) | passed (192.3 s) | passed (186.7 s) |
| Turn-2 output identical with vs without session (FP16 KV), 3/3 reps | **yes** | **yes** |
| B2: k8v4 two-turn identical with vs without session | **yes** (reused 384) | **yes** (reused 384) |
| Tokens reused of turn-2 prompt | 384 / 411 (12 of 12 full blocks) | 384 / 411 |
| Forwards turn 2, with / without | 33 / 45 | 33 / 45 |
| Turn-2 total seconds with / without (mean of 3) | 7.62 / 9.84 | 7.69 / 9.88 |
| Swap across the step | 702 → 1245 MB (**+543, env INVALID** — timings descriptive only) | 1221 → 1207 MB (valid) |

Reading `[Inferred]`: both implementations are **bit-exact on the served 4-bit artefact** (M6 met — confirms the
§2 correction that prefill reuse is exact, not just mathematically equivalent). Saving 12 prefill forwards
at ~0.185 s each predicts ≈ 2.2 s; measured ≈ 2.2 s (−22.5 % turn-2 wall-clock at this prompt length) — the
two implementations are indistinguishable on M7. `prefillSeconds` (≈ 0.33 s "without") understates prefill
because, uninstrumented, lazy evaluation pushes the work into the first denoise step — as pre-registered,
total time and forwards are the primary numbers. Unexplained: `peakGB 17.505` (MLX peak allocator) exceeds
the 16 GB of physical RAM; it is cumulative over 4 model generations plus weights and needs a look before
anyone quotes it.

### 9.9 Final verdict

All pre-registered correctness gates (M4 hidden 5/5, M5 0 regressions, M6 real-model bit-exactness) pass for
**both**. M7 (speed) is a tie by construction — both skip exactly the same forwards. M8 blind review: tie leaning
impl-x (confidence 2/5). M1 cost: impl-x ~8 % fewer tokens, ~35 % fewer tool calls, ~40 % less wall-clock
(part of impl-y's wall-clock was gate queueing). **No winner on output quality; impl-x (Opus) was somewhat
cheaper and faster to the same result on this one task.** n = 1 task — a case study, not a model ranking.
