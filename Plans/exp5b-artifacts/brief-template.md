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
