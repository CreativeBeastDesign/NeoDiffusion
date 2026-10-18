# Exp-5B notes — impl-x: cross-turn session prefix cache

**Author**: impl-x (Claude Opus 5.5), 2026-10-18. **Host**: dev M1 16 GB, toy FP32 fixtures only (no real model loaded).
**Branch**: `exp5b-impl-x` in worktree `.claude/worktrees/agent-ae8403ea4fa3150dc`, based on `5f05bf9`.

## 0. Setup deviation (read first)

The worktree I was given was checked out at `16794ec` (`main`, GPU-control commits). That tree has no
`KVCacheCompression.swift`, `KVCachePool.swift`, `cleanCommitSkipEligible` or `Tools/compute-gate.sh` — the
brief cannot be followed on it. The pre-registered base (`exp5b-opus-vs-fable-logbook.md` §7) is `5f05bf9`
(`kernel` branch). A `git reset --hard` was refused by the sandbox, so I created branch **`exp5b-impl-x` at
`5f05bf9`** (the original worktree branch `worktree-agent-ae8403ea4fa3150dc` is untouched at `16794ec`). All
work and the commit are on `exp5b-impl-x`. `[Sourced: git log; diff -rq against the main checkout showed the
source trees then matched, except the orchestrator's uncommitted LLaDABench.swift]`.

## 1. Design

- **`SessionPrefixCache`** (new, `Packages/DiffusionGeneration/Sources/SessionPrefixCache.swift`) owns the last
  call's `ExactPrefixCache` *storage* + the ids it was computed for + a compatibility key.
- **Checkout** (start of `generateCached(…, session:)`, after all preconditions): compare the prompt with the held
  ids block by block; reuse `r` = longest leading run of whole equal blocks, capped at `prompt.count / B` and at
  the held length. The held layers are truncated to `r` tokens and their storage is **moved** (not shared) into
  the call's fresh `ExactPrefixCache` (`LayerKVCache.adoptCommittedStorage`). The prefill loop starts at block
  `r / B`. The session is empty for the duration of the call.
- **Checkin** (after `run`): the session takes the call's cache and `Output.finalSequence` (== exactly the
  committed ids — `run`'s `committedIds`; checked against every layer's `committedLength`, mismatch ⇒ session
  dropped). Per-call transient state (pending K/V, FlashBlock accumulators/runner/tables) is cleared first.
- **Storage truncation** for every form `LayerKVCache` holds: raw (slice axis 2), pool (`KVCachePool.truncate` —
  logical length only, tail overwritten in place on next commit), `CompressedKVTensor.truncated(toLength:)`
  token-major (slice axis 2) and per-channel kivi4 Keys (slice axis 3 to `n·bits/32` packed / `n/g` groups;
  precondition `n % groupSize == 0`). Length 0 frees the storage.
- `Metrics.sessionReusedTokens` (+ `RunStats.sessionReusedTokens`) is set from what was actually adopted;
  `forwardsEvaluated` drops by exactly `sessionReusedTokens / B` (asserted in every test turn).
- Old `generateCached(prompt:params:iceTemplate:streamBlock:)` now forwards to the new overload with
  `session: nil`; that path skips every session branch (no behaviour change).

## 2. Exactness argument `[Inferred: code reading; verified FP32 toy + quantized-KV toy, not 4-bit weights]`

Block b's committed K/V is a deterministic function of (model weights, B, prefix storage form, ids of blocks
0…b). Prefill computes it with `captureAndCommit`: one B-wide plain cached forward (`model(ids, positionIds:,
caches:)`, mask nil, frozen nil, no router reuse/ablation) at absolute positions against the committed prefix.
Every commit the session can hold came from that same computation:
1. prompt blocks — the same `captureAndCommit`;
2. generated blocks — `captureAndCommit` over the final ids, or the clean-commit skip, which
   `cleanCommitSkipEligible` (allowlist) permits only when the settle step was that same plain fresh forward
   over exactly the committed ids (same overload; `frozen` is nil unless JOT/SureLock, both excluded);
3. MultiBD (nBuf=2) commits still go through `onSettled` → B-wide `captureAndCommit` against the prefix only
   (skip ineligible at nBuf≠1), so they are canonical too.
By induction over blocks (block b sees blocks < b only through the same held storage), a held block is
bit-identical to what this call's prefill computes, **given MLX determinism for identical op sequences at
identical shapes/dtypes** `[Inferred]`. The one exception, **sub-block commit** (Option C: n<B-token capture
forwards — different within-block attention and non-block-aligned lengths), bypasses the session.

Residual risk for the 4-bit artefact `[Speculative]`: the clean-commit skip's K/V comes from a step forward that
was built inside a K-step speculative graph and also produced logits; I rely on MLX evaluating each op with the
same kernel regardless of graph context (no `compile` is used on this path). The hidden real-model B1 test is
the authority here; the FP32 toy tests at K=1 and K=4 with skip on and off pass.

## 3. Decisions

| Decision | Reason |
|---|---|
| Compatibility key = model instance (weak ref + `ObjectIdentifier`), layerCount, `blockLength`, `kvQuantization`, `kvPoolEnabled`, mask semantics (`.strict` constant) | These change the K/V function or its storage. Pool is included conservatively: the pool view is a strided slice of a capacity-sized buffer, and I did not want to depend on attention kernels being stride-invariant bit-for-bit. Mask semantics is constant today (cached path is `.strict`-only); it is in the key so a future parameter cannot silently mix. |
| Incompatible key ⇒ **start fresh** (drop the store, run without reuse, then hold this call's blocks) | The new configuration is what the client is using now; keeping the stale store would never be reused. Counted in `incompatibleResetCount`. |
| `subBlockCommit` ⇒ **bypass** (no read, no write, session untouched; `bypassCount`) | Its commits are not canonical B-wide K/V; the existing store is still valid for compatible later calls. |
| NOT in the key: thresholds/mode, nBuf, speculation policy, `speculationK`, dyn-τ, JOT/FlashBlock/CSA2/Elastic/router reuse/capacity/ICE/credit, `skipRedundantCommitForward` | They change which tokens are generated, not the K/V of given tokens (commit is always a plain capture or an allowlisted skip). nBuf=2→1 mixing is tested. |
| Move storage instead of sharing | `KVCachePool` uses `mlx_slice_update`; with a second reference the buffer could not be donated (silent copy per commit). Moving also makes a concurrent call see an empty session instead of racing. |
| Weak model reference | A session must never keep a 9.5 GB model alive; a deallocated model ⇒ incompatible. |
| `@unchecked Sendable` + `NSLock` around hand-over | The server is single-request (RequestQueue) but handlers run in tasks; the lock makes misuse safe (no reuse) rather than racy. |
| Reuse capped at `prompt.count / B` even when held ids match further | Brief + engine semantics: the prompt tail shares its block with generation and is never prefilled. |
| Optional `maxCachedTokens` (default nil) | Hard cap, rounded down to whole blocks after each call. |

## 4. Cost

- **Host / sync**: per call, one `[Int]` prefix compare (≤ prompt.count elements) and lazy slices. **No new GPU
  readbacks, none per step.** The denoising loop is unchanged.
- **Memory** `[Sourced: KVCacheQuantization doc for bytes/token]`: the session keeps the *last* call's prefix
  store alive between calls — bytes the call already allocated. FP16: `2·layers·nKV·headDim·2` = **40,960 B/token**
  on LLaDA2.1-mini (≈ 40 MB per 1k tokens); k8v4 16,640; kivi4 12,160; fp8 21,760 B/token. With the pool, the
  footprint is the pool capacity (geometric growth, truncation does not shrink it). Bound: one call's committed
  length (≤ prompt + genLength rounded up to B); `maxCachedTokens` gives a hard cap. Truncated raw/compressed
  slices keep the old buffer alive only until the next commit concatenates.

## 5. Server

`Tools/diffusion-server`: `--session-cache` or `NEODIFFUSION_SESSION_CACHE=1` (default OFF; startup prints
`Session prefix cache: ON|off`). One `SessionPrefixCache` for the process; blocking and streaming handlers pass it
and print `[Session] <id>: reused R/P prompt tokens (forwardsEvaluated=N), session now holds H tokens (…)` from
the engine's effective echo. Flag off ⇒ the exact pre-existing `generateCached` calls. Not run (needs the model).

## 6. Tests

`Tests/DiffusionGenerationTests/SessionPrefixCacheTests.swift` (8 tests). Every session turn is compared against
the same call without a session: `finalSequence`, `tokens`, `blockCommits`, `stepsPerBlock` identical; reuse echo
equals an independent re-statement of the rule; `forwardsEvaluated` lower by exactly reused/B; the session holds
the call's committed ids.
- `testMultiTurnExactnessK1AndK4` — 4 fixture seeds (Q/S, eos on/off) × K∈{1,4} × clean-commit skip on/off × 3 turns.
- `testMultiBDCompatibleAndExact` — nBuf=2 turn then nBuf=1 turn reusing it.
- `testDivergedAndPartialPrefixes` — divergence inside block 1, re-convergence, divergence at token 0, prompt a
  strict prefix of held ids, block-aligned prompt, sub-block prompt.
- `testIncompatibleParamsStartFresh` — B 16→8, pool toggle, reuse resumes, other model instance.
- `testSubBlockCommitBypassesSession` — bypass, session untouched, later compatible reuse.
- `testResetAndCap` — reset, `maxCachedTokens`.
- `testQuantizedKVAndPoolMultiTurnExactness` — seeded random toy (head_dim 64, B=32) for k8v4, kivi4 (per-channel
  Keys), pool (capacity 32, forced growth) × K∈{1,4}: 3-turn chats + divergence truncation; quant-mode change ⇒ fresh.
- `testTruncateAndAdoptEveryStorageForm` — raw / pool / k8v4 / kivi4 / fp4g32: commit 4 blocks, truncate to
  {0,1,2,4} blocks, move, append; storage (incl. packed + scales) bit-equal to a cache built from the kept blocks.

**Mutation check**: with the token comparison disabled (reuse every block regardless of ids),
`testDivergedAndPartialPrefixes` failed with 18 failures, including `finalSequence differs` — the identity
assertions detect wrong K/V, not just a wrong echo. Reverted.

Commands (all through `Tools/compute-gate.sh impl-x-*`, `NEODIFFUSION_LOOP_FIXTURE_DIR=…/scratch/loop_fixtures`):
- `swift build --build-tests` → `Build complete!`; `Tools/seed-metallib.sh` once.
- `swift test --filter SessionPrefixCacheTests` → first run `Executed 8 tests, with 1 failure` (my helper did
  not apply the `maxCachedTokens` cap to its expectation — test bug, fixed); after the fix
  `swift test --filter SessionPrefixCacheTests/testResetAndCap` → `Executed 1 test, with 0 failures`.
- Full suite: see §6a.

### 6a. Full suite

`Tools/compute-gate.sh impl-x-fulltest swift test > full1.log 2>&1` (exit 1), the three per-bundle
`Test Suite 'All tests'` lines:
- DiffusionKernelsTests: `Executed 1 test, with 0 failures (0 unexpected)`
- DiffusionGenerationTests: `Executed 105 tests, with 26 tests skipped and 1 failure (0 unexpected)`
- DiffusionCoreTests: `Executed 58 tests, with 32 tests skipped and 0 failures (0 unexpected)`

Total 164 executed (baseline 156 + my 8), 58 skipped, **1 failure**: `LLaDA2MoeTokenizerTests.testTokenizerParity`
— `Fixture file not found at ./scratch/tokenizer_test_fixtures.json`. That test hard-codes a relative path to a
git-ignored fixture that does not exist in a worktree (rule 3: no `scratch/` here); it touches no code I changed.
It is environmental, but I could not prove it green in this worktree. The higher skip count vs the baseline
(24) has the same cause (fixture-gated tests using `./scratch/...` paths). All 8 SessionPrefixCacheTests
passed in this run. Not re-run (budget: at most two full runs).

## 7. Limitations / risks

- **4-bit weights not exercised** (rule 2). Bit-exactness there rests on §2's argument; the clean-commit-skip
  path is the part I would watch in the hidden B1 test.
- `prefillSeconds` is lazy-blurred unless `instrument: true` (unchanged behaviour); use forwards/total wall-clock.
- Session holds only the *last* conversation (one branch). A client alternating between two conversations gets
  no reuse; a multi-entry (radix/LRU) store is out of scope.
- The server session is process-wide and unauthenticated: requests from different clients share it. Exactness
  makes that safe for outputs, but timing reveals whether a prefix was cached (a side channel) — fine for the
  local single-user server, not for a multi-tenant deployment.
- Pool truncation does not shrink capacity.

## 8. Where I think the brief is off

- "nBuf≠1/MultiBD commits" is listed as an incompatibility to think about; by code reading MultiBD commits are
  canonical B-wide capture forwards, so I kept them compatible (tested).
- The brief's worktree was not at the pre-registered base (§0).
