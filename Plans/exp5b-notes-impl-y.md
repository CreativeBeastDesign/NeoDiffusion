# exp-5b — cross-turn session prefix cache — implementation notes (impl-y)

Status: implemented, dev-only (toy FP32 fixtures). Real-model evaluation is somebody else's step.
Branch `impl-y-session-prefix-cache` in worktree `.claude/worktrees/agent-ac87f8dedec5e3cb7`, based on
`kernel` @ `5f05bf9` (see §0).

Provenance tags follow `.agents/AGENTS.md`: **[sourced]** = reproducible by a named command/test/file,
**[inferred]** = follows from sourced facts, **[speculative]** = not measured.

## 0. Base-branch note (read first)

The worktree was created from `main` (`16794ec`, GPU-control T1.5). The brief names
`KVCacheCompression.swift`, `CompressedKVTensor`, `kivi4` and `GenerationParams.cleanCommitSkipEligible`,
none of which exist on `main` — they live on `kernel` (`5f05bf9`, "Cache campaign clean-up"), which is
what the main checkout has checked out and what its working tree matches file-for-file **[sourced:
`diff -q` of the ten engine/server files against the main checkout, all "same"]**. I therefore created
`impl-y-session-prefix-cache` from `kernel` and worked there. The original worktree branch
`worktree-agent-ac87f8dedec5e3cb7` is untouched. `main`'s five GPU-control commits are **not** on my
branch; merging them is a separate (conflict-free as far as I can see — they touch
`DiffusionEngine+Scheduler` selection code, I touch one line of its `Metrics(...)` call) decision.

## 1. Design

### 1.1 What is reused, and why it is exact

`generateCached` commits every block into `ExactPrefixCache` either through the dedicated capture
forward (`captureAndCommit`: plain `model(ids, positionIds:, caches:)` over the block's final tokens
at its absolute positions) or, when `cleanCommitSkipEligible`, from the settle step's pending K/V —
already proven output-identical to the capture path (`testSkipRedundantCommitForwardParity`, 16/16
cases × K∈{1,4}) **[sourced]**. Under the `.strict` block mask (the only semantics `generateCached`
runs) a committed block's K/V is a function of `(model weights, tokens of this and all earlier
blocks, absolute positions, KV storage mode)` and nothing else — that is the M5(b) exactness argument
(phase-2 §5 deviation 1) **[sourced: `testCacheEnabledIdentity`]**. A later call's prefill forward
over the same leading tokens is the *same* computation with the *same* inputs, so adopting the stored
K/V instead of recomputing it is identical, not approximate **[inferred]**. With quantized storage the
compressed representation is built block-wise (`CompressedKVTensor.append` quantizes only the new
block; per-token groups are along head-dim; kivi4's per-channel groups are 32 tokens = one block), so
a truncated store equals a freshly built store over the retained blocks **[sourced:
`testCompressedTruncationMatchesFreshCompression`, bitwise on packed/scales/biases]**.

Reuse rule (implemented in `SessionPrefixCache.prepare`):
`reusedBlocks = min(LCP(sessionTokens, prompt) / B, prompt.count / B, sessionTokens.count / B)`.
Only whole blocks; only the longest leading run of equal tokens; never past the prompt's full blocks
(the prompt tail shares a block with generation and is never prefilled — `run` builds it into the
first active slot).

### 1.2 Types and flow

- `SessionPrefixCache` (new, `Packages/DiffusionGeneration/Sources/SessionPrefixCache.swift`): holds
  `committedTokens: [Int]`, the per-layer `[LayerKVCache]`, a `Key`, a `maxCachedTokens` bound and
  three diagnostics counters (`lastReusedTokens`, `incompatibleResets`, `boundEvictions`).
  Public API as required: `init()` (with `maxCachedTokens: Int = 8192`), `cachedTokenCount`, `reset()`;
  extras: `committedTokens`, `key`, `cachedBytes`.
- `prepare(prompt:key:layerCount:initialPoolCapacity:)` (internal) → compares keys (reset on mismatch),
  computes the reusable prefix, **truncates** the held layers to it (`LayerKVCache.truncateCommitted`),
  and returns an `ExactPrefixCache(adopting:)` — or a fresh cache when nothing matches. While the call
  runs the session holds **no** layers (they are the live cache's).
- `store(cache:committed:)` (internal) → after `run`, re-checks `cache.committedLength ==
  finalSequence.count` on every layer and block alignment, applies the bound, then takes the layers.
- `DiffusionEngine.generateCached(prompt:params:iceTemplate:streamBlock:session:)` (new public
  overload; the old 4-argument signature is a thin wrapper passing `session: nil`, so existing calls
  are byte-for-byte unchanged). The prefill loop starts at `reusedTokens / B`;
  `stats.sessionReusedTokens` is set right there — the echo is the loop's own start index, not the
  request's wish.
- `Output.Metrics.sessionReusedTokens` (new field; the only `Metrics(...)` constructor is in
  `DiffusionEngine+Scheduler.swift`).
- `GenerationParams.sessionReuseEligible` (new): the bypass allowlist (§2.3).
- `LayerKVCache.truncateCommitted(to:)` + `resetTransientState()`, `KVCachePool.truncate(to:)`,
  `CompressedKVTensor.truncated(toTokens:)`: the storage primitives.
- Server (`Tools/diffusion-server`): `--session-cache` / `NEODIFFUSION_SESSION_CACHE=1` (default OFF),
  optional `--session-max-tokens` / `NEODIFFUSION_SESSION_MAX_TOKENS`; one session per process;
  per-request `[Session] …` log line with the engine's effective echo. With the flag off `session`
  is `nil` and both handlers call the new overload with `nil` — identical to before.

## 2. Decisions (each with its reason)

1. **Reset vs bypass are two different things.** Storage-shape mismatch (`Key`: block length, KV
   quantization mode, pool flag, model instance) → the held bytes are unusable → *reset the session*
   and let the new call rebuild it (the next turn under the new config benefits). Feature sets that
   change *what a commit means* → *bypass* (session neither read nor written), so an occasional
   experimental request does not destroy a good session. Both are tested
   (`testIncompatibleKeyResetsSession`, `testIneligibleParamsBypassSession`).
2. **`kvPoolEnabled` is in the key** although pool and raw storage hold the same values: a
   `LayerKVCache` cannot switch storage form mid-life (`commitPending` with `usePool` on a raw store
   would create an empty pool and drop `rawKeys` — data loss) **[sourced: `LayerKVCache.commitPending`
   lines 145–152]**. Keying is cheaper than making the switch safe.
3. **Model identity by `ObjectIdentifier(model)`.** Two instances of the same weights are treated as
   different (`testDifferentModelInstanceResetsSession`). Conservative on purpose: there is no cheap
   weight fingerprint, and the server has exactly one model.
4. **Bypass list** (`sessionReuseEligible`): `nBuf != 1`, `subBlockCommit`, `flashBlockEnabled`,
   `dualCachingEnabled`, `elasticCacheEnabled`, `moduleAblation != .none`, `iceEnabled`, plus a
   per-call `iceTemplate != nil`. Reasons in the doc comment. Notable *inclusions*: CSA2, JOT/SureLock,
   S2D2, router reuse, dynamic-τ, credit decoding, temporal voting — all commit through the plain
   capture forward (`cleanCommitSkipEligible` is false for them) so the committed K/V is exact
   **[inferred from `onSettled`; not separately tested — see §5]**.
5. **Truncation keeps buffers.** Pool: only `committedLength` moves; the stale tail is overwritten by
   the next `commit` and never observed (the live view is a `0..<committedLength` slice). Raw/compressed:
   lazy slices. No `eval`, no readback.
6. **Transient FlashBlock state is wiped** on every hand-over (`resetTransientState`) even though
   FlashBlock calls bypass the session — belt and braces: only committed K/V may cross a call.
7. **Bound = all-or-nothing.** Positions are absolute (RoPE at absolute index, block-causal mask), so a
   leading prefix cannot be evicted. If `finalSequence.count > maxCachedTokens` the session is emptied
   (`boundEvictions += 1`). Default 8192 tokens.
8. **Defensive `store`.** If any layer's `committedLength != finalSequence.count` (should be impossible
   with the bypass list — every commit grows both in lockstep) the session is reset and
   `assertionFailure` fires in debug; serving never stores an unprovable cache.
9. **No `diffusion-bench` hook** (brief: not required).
10. **Server concurrency**: `extension SessionPrefixCache: @unchecked Sendable` in the server only
    (same pattern as `DiffusionEngine`), justified by `RequestQueue` serialising every generation.
    The library type stays honest (not Sendable).

## 3. Cost

Per cached token, summed over layers: `layers × nKV × headDim × 2 (K+V) × bytes/elem`
**[inferred from config.json + KVCacheQuantization doc]**:
- LLaDA2.1-mini FP16 (`.none`): 20 × 4 × 128 × 2 × 2 B = **40,960 B/token** → 8192 tokens ≈ 335 MB.
- `k8v4`: 16,640 B/token; `kivi4`: 12,160 B/token (doc figures incl. scales/biases).
- Pool: `allocatedBytes` up to ~2× logical (geometric growth), reported by `cachedBytes` as logical.
Bound: `maxCachedTokens` (default 8192), see decision 7. On the 16 GB M1 with the 4-bit model at
~9.6 GB peak, 335 MB is affordable but not free; `--session-max-tokens` exists to tune it
**[speculative: not measured on the real model]**.

## 4. Sync / hot-path

- No new per-step work. The reuse decision compares two Swift `[Int]` arrays once per call; truncation
  is lazy slicing; `store` moves references. **Zero added GPU→CPU readbacks**, per step or per call
  **[sourced: `testSessionAddsNoSyncPoints` — `syncPoints` equal with and without a session]**.
- A reused prefix is *not* evaluated eagerly; MLX materialises it on the first denoising forward as it
  would have for a fresh prefill. Under `instrument: true`, prefill phase time therefore drops to
  the un-reused blocks only (correct attribution).

## 5. What I tested — exact commands and result lines

Fixtures: `NEODIFFUSION_LOOP_FIXTURE_DIR=/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/loop_fixtures`
(toy FP32, block 16, 2 layers, 16 cases). Every heavy command went through
`/Users/andrebarlocher/Documents/Swift/NeoDiffusion/Tools/compute-gate.sh impl-y-<purpose> …`
(the gate script is uncommitted in the main checkout, so it is invoked by absolute path).

Build: `compute-gate.sh impl-y-build swift build --build-tests` → `Build complete! (97.03 Sek.)`, 0 errors
(cold build after a 540 s gate wait; 20 warnings, all pre-existing, none in touched files); then
`compute-gate.sh impl-y-seed Tools/seed-metallib.sh`.

New test class alone — `compute-gate.sh impl-y-test-session2 swift test --filter SessionPrefixCacheTests`
(scratchpad `logs/test-session2.log`):
`Executed 17 tests, with 0 failures (0 unexpected) in 53.994 (53.996) seconds` — all 17 passed.
(First attempt, `impl-y-test-session`, crashed on the g64/head_dim-32 constraint described below; fixed.)

Full suite, run 1 of 2 — `compute-gate.sh impl-y-fulltest swift test` (scratchpad `logs/fulltest1.log`,
gate: `start waited=0s avail=6963MB`, `end rc=1 ran=645s`):
- DiffusionKernelsTests: `Executed 3 tests, with 0 failures`, `Test Suite 'All tests' passed`.
- DiffusionGenerationTests: `Executed 114 tests, with 26 tests skipped and 1 failure (0 unexpected) in 639.376 (639.389) seconds`,
  `Test Suite 'All tests' failed at 2026-10-18 16:01:18.433`.
- DiffusionCoreTests: `Executed 58 tests, with 32 tests skipped and 0 failures`, `Test Suite 'All tests' passed`.
- The one failure: `LLaDA2MoeTokenizerTests testTokenizerParity` — `Fixture file not found at
  ./scratch/tokenizer_test_fixtures.json` (that test uses `XCTFail`, not `XCTSkip`, on a missing
  fixture, and this worktree has no `scratch/`). **Environmental, not a code regression.** Verified
  by symlinking the main checkout's fixture into this worktree's git-ignored `scratch/` and re-running
  `compute-gate.sh impl-y-test-tokenizer swift test --filter LLaDA2MoeTokenizerTests` →
  `Test Case '-[DiffusionGenerationTests.LLaDA2MoeTokenizerTests testTokenizerParity]' passed (2.314 seconds)`.
  Every other suite in the run passed (`DenoisingLoopParityTests` 9/9, `MultiBDParityTests` 7/7,
  `JotTests` 10/10, `S2D2SpeculationTests` 7/7, `WP2bTests` 9/9, `SessionPrefixCacheTests` 17/17 …).
  Baseline for comparison: CLAUDE.md records 120 tests / 0 failures on `main` (2026-07-17); the `kernel`
  branch total here is 3 + 114 + 58 = 175 tests incl. my 17 (so 158 pre-existing), 58 skips all
  `XCTSkipUnless` fixture gates. The second allowed full run was **not** spent — the single failure is
  explained and individually re-verified above, and a re-run would only re-test the symlink.

`SessionPrefixCacheTests` (new, `Tests/DiffusionGenerationTests/SessionPrefixCacheTests.swift`):
- `testMultiTurnExactnessAndReuse` — 3 turns × 3 fixture cases × K∈{1,4}; each turn: session output
  == session-less output (`finalSequence`, `tokens`), `sessionReusedTokens` == expected,
  `forwardsEvaluated` difference == reused blocks, session holds the call's committed tokens.
- `testContinuationBeyondSessionIsPrefilled`, `testDivergedPrefixReusesOnlyLeadingBlocks` (K=1,4),
  `testDivergenceInFirstBlockReusesNothing`, `testShorterPromptCapsReuseAtPromptFullBlocks`,
  `testSubBlockPromptReusesNothing` — partial/diverged/shorter prompts.
- `testIncompatibleKeyResetsSession` (kvQuantization, kvPool, blockLength=8),
  `testDifferentModelInstanceResetsSession`, `testIneligibleParamsBypassSession` (nBuf=2, iceTemplate),
  `testSessionReuseEligibilityAllowlist`.
- `testQuantizedKVSessionExactness` (**fp4g32** × K∈{1,4}, incl. a diverged turn),
  `testPooledKVSessionExactness`. **Why not k8v4/kivi4 end-to-end** — the toy model's `head_dim`
  is 32 and MLX refuses group-64 quantization on a 32-wide axis (first run crashed:
  `Fatal error: [quantize] The last dimension of the matrix needs to be divisible by the quantization
  group size 64 … shape (1,4,96,32)` **[sourced: scratchpad `logs/test-session1.log`]**). `k8v4` and
  `kivi4` both quantize Values at g64, so *no* toy-fixture E2E test of those modes is possible (none
  exists in the repo today either — `DualCachingTests` works on synthetic tensors). `fp4g32` (K and V
  g32) is the only compressed mode the toy shapes admit and is representative of the per-token
  layout. The brief's "quantized KV (k8v4 and kivi4)" is therefore met at storage level only (next
  two tests, real-model head_dim 128) — a real-model multi-turn A/B under `k8v4`/`kivi4` is on the
  evaluator's list, not provable here.
- `testCompressedTruncationMatchesFreshCompression` — per-token int4/int8 (g64) and **per-channel
  kivi4 (group 32)** on a synthetic `[1,4,96,128]` f16 tensor: truncation == fresh compression,
  bitwise on packed/scales/biases, and append-after-truncate == full. (At B=16 `compress` would fall
  back to the per-token layout for kivi4 — `x.dim(2) % 32 != 0` — so the per-channel path is only
  reachable with 32-token blocks, as in production.)
- `testLayerKVCacheTruncateAllStorageForms` — raw / pool / k8v4 / kivi4 `LayerKVCache` with
  `[1,4,32,128]` f16 blocks: truncate + recommit equals never-truncated, bitwise; truncate to 0 empties.
- `testResetDropsEverything`, `testMaxCachedTokensBound`, `testSessionAddsNoSyncPoints`.

## 6. Known limitations / risks

- **Not run on the real model.** Exactness on the 4-bit artefact follows from the same argument
  (same code path, f16 storage) but is **[inferred]** until someone runs a multi-turn A/B there. The
  CLAUDE.md rule applies: the toy heuristics prove identity of the mechanism, not quality.
- **Chat-template boundary.** Reuse only helps if the tokenizer renders turn N's prompt+answer as a
  byte-identical *prefix* of turn N+1's prompt. With the LLaDA2 chat template the assistant turn is
  closed by `<|im_end|>`/EOS and the new user turn appended, so the reused run ends at the last full
  block before the new user message **[speculative — verify on the real tokenizer; the server log line
  shows the effective count]**. The generated answer's *trailing padding* (EOS-filled positions after
  the first EOS in the last block) is part of the committed sequence but **not** of what a client
  resends as text, so the last generated block is normally *not* reusable — expected loss ≤ 1 block.
- **Divergent sampling at temperature > 0** is irrelevant to exactness (K/V depend on tokens, not on
  how they were sampled) but the *reused tokens* of course are whatever the client resends.
- `nBuf == 2`, S2D2, JOT, CSA2 compositions: S2D2/JOT/CSA2 are *included* on the argument in §2.4 but
  have no dedicated session test; `nBuf == 2` is excluded. If a composition test is wanted, add the
  config to `testQuantizedKVSessionExactness`'s pattern.
- The session is single-threaded by contract; a multi-request server would need one session per
  conversation plus locking. Out of scope (single-request server).
- `assertionFailure` in `store` is debug-only; release builds silently reset. Intentional.

## 7. Anything in the brief I think is wrong or worth flagging

- The brief says the worktree branch is "yours" and names `kernel`-only symbols; the worktree was cut
  from `main`. See §0 — I rebased by creating a branch from `kernel` rather than resetting the given
  branch (a `git reset --hard` was refused by the sandbox, and a merge would have mixed the two
  campaigns' unmerged work).
- "Reuse … only the longest leading run whose tokens equal the session's committed tokens" — correct,
  and note it is strictly stronger than "the prompt starts with the previous prompt": a client that
  edits history mid-way still gets the leading blocks (tested).
- `Tools/compute-gate.sh` is not in the repository (uncommitted in the main checkout). Not added here
  (it is not mine), invoked by absolute path.
