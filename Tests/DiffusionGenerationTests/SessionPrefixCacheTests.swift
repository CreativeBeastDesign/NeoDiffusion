import XCTest
import MLX
import MLXNN
import DiffusionCore
@testable import DiffusionModel
@testable import DiffusionGeneration

/// exp-5B (impl-x): cross-turn ``SessionPrefixCache``. The contract under test is **exactness**:
/// a call with a session must produce exactly the output of the same call without one, while
/// skipping `sessionReusedTokens / B` prefill forwards. FP32 toy fixtures (loop-parity weights,
/// B=16) for the uncompressed / pool paths; a seeded random toy with head_dim 64 and B=32 for the
/// quantized-KV paths (the fixture's head_dim 32 is below the group-64 quantizer's minimum, and
/// kivi4's per-channel Key layout needs B to be a multiple of its 32-token group).
final class SessionPrefixCacheTests: XCTestCase {

    var traces: DenoisingLoopParityTests.Traces!
    var model: LLaDA2MoeModel!
    var weightsURL: URL!
    var configJSON: [String: Any]!

    override func setUpWithError() throws {
        let dir = DenoisingLoopParityTests.fixtureDir
        weightsURL = dir.appendingPathComponent("weights.safetensors")
        let tracesURL = dir.appendingPathComponent("traces.json")
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: tracesURL.path),
            "Loop fixtures missing at \(dir.path) — run: python3 Tools/generate_loop_fixtures.py")
        let manifest = try JSONDecoder().decode(
            DenoisingLoopParityTests.Manifest.self, from: Data(contentsOf: manifestURL))
        traces = try JSONDecoder().decode(
            DenoisingLoopParityTests.Traces.self, from: Data(contentsOf: tracesURL))
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
        configJSON = raw["config"] as? [String: Any]
        let dm = DiffusionModel(config: manifest.config)
        try dm.loadWeights(from: weightsURL)
        model = dm.model
    }

    // MARK: - Helpers

    /// Independent re-statement of the reuse rule: longest leading run of whole blocks equal to
    /// the held ids, capped at the prompt's full blocks.
    func expectedReuse(prompt: [Int], held: [Int], B: Int) -> Int {
        let maxBlocks = min(prompt.count / B, held.count / B)
        var n = 0
        while n < maxBlocks, prompt[(n * B) ..< ((n + 1) * B)] == held[(n * B) ..< ((n + 1) * B)] {
            n += 1
        }
        return n * B
    }

    func assertIdentical(_ a: DiffusionEngine.Output, _ b: DiffusionEngine.Output, _ what: String,
                         file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(a.finalSequence, b.finalSequence, "\(what): finalSequence differs", file: file, line: line)
        XCTAssertEqual(a.tokens, b.tokens, "\(what): tokens differ", file: file, line: line)
        XCTAssertEqual(a.blockCommits, b.blockCommits, "\(what): blockCommits differ", file: file, line: line)
        XCTAssertEqual(a.stepsPerBlock, b.stepsPerBlock, "\(what): stepsPerBlock differ", file: file, line: line)
    }

    /// Runs one session call and the same call without a session; asserts identity, the reuse
    /// echo, the forward saving, and what the session holds afterwards. Returns the output.
    @discardableResult
    func checkTurn(engine: DiffusionEngine, session: SessionPrefixCache, prompt: [Int],
                   params: GenerationParams, expectReuse: Int? = nil, _ what: String,
                   file: StaticString = #filePath, line: UInt = #line) -> DiffusionEngine.Output {
        let B = params.blockLength
        let held = session.cachedTokens
        let predicted = expectReuse ?? expectedReuse(prompt: prompt, held: held, B: B)
        let withS = engine.generateCached(prompt: prompt, params: params, session: session)
        let without = engine.generateCached(prompt: prompt, params: params)
        assertIdentical(withS, without, what, file: file, line: line)
        XCTAssertEqual(withS.metrics.sessionReusedTokens, predicted, "\(what): reuse echo", file: file, line: line)
        XCTAssertEqual(without.metrics.sessionReusedTokens, 0, "\(what): no-session echo must be 0", file: file, line: line)
        XCTAssertEqual(withS.metrics.forwardsEvaluated,
                       without.metrics.forwardsEvaluated - predicted / B,
                       "\(what): exactly reused/B prefill forwards must be skipped", file: file, line: line)
        XCTAssertEqual(withS.metrics.committedTokensCount, without.metrics.committedTokensCount,
                       "\(what): committed length", file: file, line: line)
        let keep = session.maxCachedTokens.map { ($0 / B) * B } ?? Int.max
        XCTAssertEqual(session.cachedTokens, Array(withS.finalSequence.prefix(keep)),
                       "\(what): session must hold this call's committed ids", file: file, line: line)
        return withS
    }

    /// A 3-turn chat: turn k+1 = turn k prompt + answer + new user tokens.
    func runConversation(engine: DiffusionEngine, params: GenerationParams, firstPrompt: [Int],
                         userTurns: [[Int]], _ tag: String) {
        let session = SessionPrefixCache()
        var prompt = firstPrompt
        var totalReused = 0
        for (t, user) in ([[]] + userTurns).enumerated() {
            prompt += user
            let out = checkTurn(engine: engine, session: session, prompt: prompt, params: params,
                                "\(tag) turn \(t)")
            totalReused += out.metrics.sessionReusedTokens
            prompt += out.tokens
        }
        XCTAssertGreaterThan(totalReused, 0, "\(tag): the conversation never reused a block (test is vacuous)")
    }

    let userTurns: [[Int]] = [Array(100 ..< 113), Array(200 ..< 221)]

    // MARK: - Multi-turn exactness (FP32 fixtures)

    func testMultiTurnExactnessK1AndK4() throws {
        let seeds = traces.cases.filter { ["p20_q_noeos", "p33_s_eos", "p33_q_eos", "p16_s_noeos"].contains($0.name) }
        XCTAssertEqual(seeds.count, 4)
        for k in [1, 4] {
            let engine = DiffusionEngine(model: model, speculationK: k)
            for c in seeds {
                // Clean-commit skip on (default) and off (legacy capture on every commit).
                for skip in [true, false] {
                    var p = c.params.toGenerationParams()
                    p.skipRedundantCommitForward = skip
                    runConversation(engine: engine, params: p, firstPrompt: c.prompt,
                                    userTurns: userTurns, "\(c.name) K=\(k) skip=\(skip)")
                }
            }
        }
    }

    /// Merged design (2026-10-18): MultiBD (nBuf=2) is *ineligible* (allowlist from impl-y) — the call
    /// bypasses the session, leaves it untouched, and output equals the session-less run. (impl-x's
    /// original version of this test asserted reuse was exact under nBuf=2 — passed at 7f125f0 —
    /// which is the evidence for relaxing the exclusion once measured on the real model.)
    func testMultiBDBypassesSession() throws {
        let c = try XCTUnwrap(traces.cases.first { $0.name == "p20_q_noeos" })
        let engine = DiffusionEngine(model: model, speculationK: 4)
        let session = SessionPrefixCache()
        let o1 = checkTurn(engine: engine, session: session, prompt: c.prompt,
                           params: c.params.toGenerationParams(), "nBuf=1 turn 0")
        let held = session.cachedTokens
        var p2 = c.params.toGenerationParams()
        p2.nBuf = 2
        p2.tauAdd = 0.1
        XCTAssertNotNil(SessionPrefixCache.bypassReason(p2))
        let prompt2 = c.prompt + o1.tokens + userTurns[0]
        let withS = engine.generateCached(prompt: prompt2, params: p2, session: session)
        let without = engine.generateCached(prompt: prompt2, params: p2)
        assertIdentical(withS, without, "nBuf=2 bypass")
        XCTAssertEqual(withS.metrics.sessionReusedTokens, 0)
        XCTAssertEqual(session.cachedTokens, held, "bypass must leave the session untouched")
    }

    /// Ported from impl-y: the eligibility allowlist itself.
    func testSessionReuseEligibilityAllowlist() throws {
        let base = traces.cases[0].params.toGenerationParams()
        XCTAssertTrue(base.sessionReuseEligible)
        var k8 = base; k8.kvQuantization = .k8v4
        XCTAssertTrue(k8.sessionReuseEligible, "storage mode is a key, not an eligibility exclusion")
        let disablers: [(String, (inout GenerationParams) -> Void)] = [
            ("nBuf2", { $0.nBuf = 2 }), ("subBlockCommit", { $0.subBlockCommit = true }),
            ("flashBlock", { $0.flashBlockEnabled = true }), ("dualCaching", { $0.dualCachingEnabled = true }),
            ("elastic", { $0.elasticCacheEnabled = true }), ("ice", { $0.iceEnabled = true }),
            ("ablation", { $0.moduleAblation = .lmHead }),
        ]
        for (name, apply) in disablers {
            var p = base; apply(&p)
            XCTAssertFalse(p.sessionReuseEligible, "\(name) must bypass the session")
            XCTAssertNotNil(SessionPrefixCache.bypassReason(p), name)
        }
    }

    /// Ported from impl-y: reuse adds no GPU→CPU sync points (CLAUDE.md gotcha 9).
    func testSessionAddsNoSyncPoints() throws {
        let c = try XCTUnwrap(traces.cases.first { $0.name == "p16_q_noeos" })
        let p = c.params.toGenerationParams()
        let session = SessionPrefixCache()
        let engine = DiffusionEngine(model: model, speculationK: 4)
        let t1 = engine.generateCached(prompt: c.prompt, params: p, session: session)
        let prompt2 = t1.finalSequence + userTurns[0]
        let fresh = engine.generateCached(prompt: prompt2, params: p)
        let reused = engine.generateCached(prompt: prompt2, params: p, session: session)
        XCTAssertGreaterThan(reused.metrics.sessionReusedTokens, 0)
        XCTAssertEqual(reused.syncPoints, fresh.syncPoints)
    }

    // MARK: - Partial / diverged prefixes

    func testDivergedAndPartialPrefixes() throws {
        let c = try XCTUnwrap(traces.cases.first { $0.name == "p33_q_noeos" })
        let p = c.params.toGenerationParams()
        let B = p.blockLength
        let engine = DiffusionEngine(model: model, speculationK: 4)
        let session = SessionPrefixCache()
        let o1 = checkTurn(engine: engine, session: session, prompt: c.prompt, params: p,
                           expectReuse: 0, "turn 0 (empty session)")
        let base = c.prompt + o1.tokens + userTurns[0]

        // Diverge inside block 1 → only block 0 reusable (truncation of a longer held store).
        var div = base
        div[B + 3] = (div[B + 3] + 1) % 900
        checkTurn(engine: engine, session: session, prompt: div, params: p, expectReuse: B,
                  "diverged in block 1")

        // Re-send the original continuation: session now holds `div`'s run, so blocks equal up to
        // block 1 only → reuse B again, and the held tail is truncated + regrown.
        checkTurn(engine: engine, session: session, prompt: base, params: p, expectReuse: B,
                  "back to base after divergence")

        // Diverge at token 0 → nothing reusable.
        var div0 = base
        div0[0] = (div0[0] + 1) % 900
        checkTurn(engine: engine, session: session, prompt: div0, params: p, expectReuse: 0,
                  "diverged at token 0")

        // Prompt is a strict prefix of the held ids (regenerate): reuse capped at the prompt's
        // full blocks, never into the block the prompt tail shares with generation.
        let regen = Array(div0.prefix(2 * B + 5))
        checkTurn(engine: engine, session: session, prompt: regen, params: p,
                  expectReuse: 2 * B, "strict prefix of held ids")

        // Exactly block-aligned prompt equal to held ids: all its blocks reused.
        let aligned = Array(regen.prefix(2 * B))
        checkTurn(engine: engine, session: session, prompt: aligned, params: p,
                  expectReuse: 2 * B, "block-aligned prompt")

        // Prompt shorter than one block → no full prompt block → reuse 0.
        checkTurn(engine: engine, session: session, prompt: Array(aligned.prefix(B - 1)), params: p,
                  expectReuse: 0, "sub-block prompt")
    }

    // MARK: - Incompatible configurations

    func testIncompatibleParamsStartFresh() throws {
        let c = try XCTUnwrap(traces.cases.first { $0.name == "p33_s_noeos" })
        let engine = DiffusionEngine(model: model, speculationK: 4)
        let session = SessionPrefixCache()
        let p16 = c.params.toGenerationParams()
        let o1 = checkTurn(engine: engine, session: session, prompt: c.prompt, params: p16, "B=16")
        let prompt2 = c.prompt + o1.tokens + userTurns[0]

        // Block length change → fresh start (would otherwise match plenty of 8-token blocks).
        var p8 = p16
        p8.blockLength = 8
        checkTurn(engine: engine, session: session, prompt: prompt2, params: p8, expectReuse: 0, "B=8 after B=16")
        XCTAssertEqual(session.incompatibleResetCount, 1)

        // Pool storage toggle → fresh start.
        var pPool = p8
        pPool.kvPoolEnabled = true
        checkTurn(engine: engine, session: session, prompt: prompt2, params: pPool, expectReuse: 0, "pool after raw")
        XCTAssertEqual(session.incompatibleResetCount, 2)

        // Same config again → reuse resumes under the new key.
        let o4 = checkTurn(engine: engine, session: session, prompt: prompt2, params: pPool, "pool again")
        XCTAssertGreaterThan(o4.metrics.sessionReusedTokens, 0)

        // Different model instance (same weights) → fresh start.
        let dm2 = DiffusionModel(config: model.config)
        try dm2.loadWeights(from: weightsURL)
        let engine2 = DiffusionEngine(model: dm2.model, speculationK: 4)
        checkTurn(engine: engine2, session: session, prompt: prompt2, params: pPool, expectReuse: 0,
                  "other model instance")
        XCTAssertEqual(session.incompatibleResetCount, 3)
    }

    /// Sub-block commit (Option C) commits n<B-token segments → bypass: no reuse, session untouched.
    func testSubBlockCommitBypassesSession() throws {
        let c = try XCTUnwrap(traces.cases.first { $0.name == "p33_q_noeos" })
        let engine = DiffusionEngine(model: model, speculationK: 1)
        let session = SessionPrefixCache()
        let base = c.params.toGenerationParams()
        let o1 = checkTurn(engine: engine, session: session, prompt: c.prompt, params: base, "plain turn")
        let heldBefore = session.cachedTokens

        var p = base
        p.jotEnabled = true
        p.jotFaithful = true
        p.subBlockCommit = true
        p.subBlockMinPrefix = 4
        p.jotK = 1
        p.jotThreshold = 0.5
        XCTAssertNotNil(SessionPrefixCache.bypassReason(p))
        let prompt2 = c.prompt + o1.tokens + userTurns[0]
        let withS = engine.generateCached(prompt: prompt2, params: p, session: session)
        let without = engine.generateCached(prompt: prompt2, params: p)
        assertIdentical(withS, without, "subBlockCommit bypass")
        XCTAssertEqual(withS.metrics.sessionReusedTokens, 0)
        XCTAssertEqual(session.bypassCount, 1)
        XCTAssertEqual(session.cachedTokens, heldBefore, "bypass must leave the session untouched")

        // The untouched session still serves a later compatible call.
        let o3 = checkTurn(engine: engine, session: session, prompt: prompt2, params: base, "after bypass")
        XCTAssertGreaterThan(o3.metrics.sessionReusedTokens, 0)
    }

    // MARK: - Reset / cap / no-session

    func testResetAndCap() throws {
        let c = try XCTUnwrap(traces.cases.first { $0.name == "p20_s_noeos" })
        let p = c.params.toGenerationParams()
        let engine = DiffusionEngine(model: model, speculationK: 4)
        let session = SessionPrefixCache()
        XCTAssertEqual(session.cachedTokenCount, 0)
        let o1 = checkTurn(engine: engine, session: session, prompt: c.prompt, params: p, expectReuse: 0, "turn 0")
        XCTAssertEqual(session.cachedTokenCount, o1.finalSequence.count)
        session.reset()
        XCTAssertEqual(session.cachedTokenCount, 0)
        XCTAssertTrue(session.cachedTokens.isEmpty)
        let prompt2 = c.prompt + o1.tokens + userTurns[0]
        checkTurn(engine: engine, session: session, prompt: prompt2, params: p, expectReuse: 0, "after reset")
        XCTAssertGreaterThan(session.cachedTokenCount, 0)

        // Hard cap: 40 tokens → 2 whole blocks of 16 kept.
        let capped = SessionPrefixCache(maxCachedTokens: 40)
        let oc = engine.generateCached(prompt: prompt2, params: p, session: capped)
        XCTAssertEqual(capped.cachedTokenCount, 32)
        XCTAssertEqual(capped.cachedTokens, Array(oc.finalSequence.prefix(32)))
        checkTurn(engine: engine, session: capped, prompt: prompt2 + oc.tokens, params: p,
                  expectReuse: 32, "capped session")
    }

    // MARK: - Quantized KV + pool (seeded random toy, head_dim 64, B = 32)

    /// Seeded random toy with head_dim 64 (fixture topology otherwise). Weights are N(0, σ²) with
    /// norms around 1 — the point is a deterministic model whose argmax is sensitive to tiny K/V
    /// differences, not plausible text.
    func randomHeadDim64Model() throws -> LLaDA2MoeModel {
        var cfg = configJSON!
        cfg["head_dim"] = 64
        let data = try JSONSerialization.data(withJSONObject: cfg)
        let config = try JSONDecoder().decode(LLaDA2MoeConfig.self, from: data)
        let dm = DiffusionModel(config: config)
        var flat: [(String, MLXArray)] = []
        for (i, (name, a)) in dm.model.parameters().flattened().enumerated() {
            let key = MLXRandom.key(UInt64(1000 + i))
            let noise = MLXRandom.normal(a.shape, key: key).asType(a.dtype)
            if name.contains("norm") {
                flat.append((name, MLXArray.ones(a.shape, dtype: a.dtype) + 0.1 * noise))
            } else {
                flat.append((name, 0.08 * noise))
            }
        }
        dm.model.update(parameters: ModuleParameters.unflattened(flat))
        eval(dm.model.parameters())
        return dm.model
    }

    func testQuantizedKVAndPoolMultiTurnExactness() throws {
        let m = try randomHeadDim64Model()
        let prompt1 = (0 ..< 70).map { ($0 * 37 + 11) % 900 }
        for (mode, pool) in [(KVCacheQuantization.k8v4, false), (.kivi4, false), (.none, true)] {
            for k in [1, 4] {
                var p = GenerationParams.mode(.q, blockLength: 32, genLength: 64,
                                              maskId: traces.maskId, eosId: traces.eosId)
                p.kvQuantization = mode
                p.kvPoolEnabled = pool
                p.kvPoolCapacity = 32   // force geometric growth across turns
                let engine = DiffusionEngine(model: m, speculationK: k)
                let tag = "\(mode.rawValue) pool=\(pool) K=\(k)"
                runConversation(engine: engine, params: p, firstPrompt: prompt1,
                                userTurns: [Array(300 ..< 337), Array(400 ..< 410)], tag)

                // Divergence truncation on this storage form: hold a long run, then diverge in
                // block 1 → keep exactly block 0.
                let session = SessionPrefixCache()
                let o = checkTurn(engine: engine, session: session, prompt: prompt1, params: p,
                                  expectReuse: 0, "\(tag) div setup")
                var div = prompt1 + o.tokens + Array(300 ..< 310)
                div[40] = (div[40] + 1) % 900
                checkTurn(engine: engine, session: session, prompt: div, params: p, expectReuse: 32,
                          "\(tag) diverged in block 1")
            }
        }
        // Quantization mode change → fresh start.
        let session = SessionPrefixCache()
        var p = GenerationParams.mode(.s, blockLength: 32, genLength: 32,
                                      maskId: traces.maskId, eosId: traces.eosId)
        let engine = DiffusionEngine(model: m, speculationK: 4)
        let o = checkTurn(engine: engine, session: session, prompt: prompt1, params: p, "fp16")
        p.kvQuantization = .k8v4
        checkTurn(engine: engine, session: session, prompt: prompt1 + o.tokens, params: p,
                  expectReuse: 0, "k8v4 after fp16")
        XCTAssertEqual(session.incompatibleResetCount, 1)
    }

    // MARK: - Storage-level truncation (every LayerKVCache form)

    func testTruncateAndAdoptEveryStorageForm() throws {
        let nKV = 2, D = 64, Bk = 32
        func block(_ seed: UInt64) -> MLXArray {
            MLXRandom.normal([1, nKV, Bk, D], key: MLXRandom.key(seed)).asType(.float16)
        }
        var blocks: [(MLXArray, MLXArray)] = []
        for i in 0 ..< 4 {
            let seed = UInt64(10 + 2 * i)
            blocks.append((block(seed), block(seed + 1)))
        }
        let extra = (block(90), block(91))
        func same(_ a: MLXArray?, _ b: MLXArray?) -> Bool {
            switch (a, b) {
            case (nil, nil): return true
            case let (x?, y?):
                guard x.shape == y.shape else { return false }
                return arrayEqual(x, y).item(Bool.self)
            default: return false
            }
        }
        func commit(_ c: LayerKVCache, _ kv: (MLXArray, MLXArray), pool: Bool, q: KVCacheQuantization) {
            c.pendingKeys = kv.0
            c.pendingValues = kv.1
            c.commitPending(usePool: pool, quantization: q)
        }
        let forms: [(String, Bool, KVCacheQuantization)] = [
            ("raw", false, .none), ("pool", true, .none), ("k8v4", false, .k8v4),
            ("kivi4", false, .kivi4), ("fp4g32", false, .fp4g32),
        ]
        for (name, pool, q) in forms {
            for keep in [0, 1, 2, 4] {
                let a = LayerKVCache(usePool: pool, initialPoolCapacity: 32, quantization: q)
                for kv in blocks { commit(a, kv, pool: pool, q: q) }
                if q == .kivi4 { XCTAssertTrue(a.compressedKeys!.perChannel, "kivi4 must use per-channel Keys") }
                a.truncateCommitted(to: keep * Bk)
                XCTAssertEqual(a.committedLength, keep * Bk, "\(name) keep=\(keep)")

                // Move into a fresh layer (the session hand-over), then grow both.
                let moved = LayerKVCache(usePool: pool, initialPoolCapacity: 32, quantization: q)
                moved.adoptCommittedStorage(from: a)
                XCTAssertEqual(a.committedLength, 0, "\(name): source must be empty after move")
                XCTAssertEqual(moved.committedLength, keep * Bk)
                commit(moved, extra, pool: pool, q: q)

                let ref = LayerKVCache(usePool: pool, initialPoolCapacity: 32, quantization: q)
                for kv in blocks.prefix(keep) { commit(ref, kv, pool: pool, q: q) }
                commit(ref, extra, pool: pool, q: q)

                XCTAssertEqual(moved.committedLength, ref.committedLength, "\(name) keep=\(keep)")
                XCTAssertTrue(same(moved.keys, ref.keys), "\(name) keep=\(keep): keys differ after truncate+append")
                XCTAssertTrue(same(moved.values, ref.values), "\(name) keep=\(keep): values differ after truncate+append")
                if q != .none {
                    XCTAssertTrue(same(moved.compressedKeys?.packed, ref.compressedKeys?.packed), "\(name): packed K")
                    XCTAssertTrue(same(moved.compressedKeys?.scales, ref.compressedKeys?.scales), "\(name): K scales")
                    XCTAssertTrue(same(moved.compressedValues?.packed, ref.compressedValues?.packed), "\(name): packed V")
                }
            }
        }
    }
}
