import XCTest
import MLX
import DiffusionCore
@testable import DiffusionModel
@testable import DiffusionGeneration

/// WP-2b (Streaming-dLLM cluster) acceptance.
///
/// There is no reference trace for either sub-item (the reference implements static τ and
/// commit-time eos_early_stop only), so these are internal-consistency gates on the M5 loop
/// fixtures plus a synthetic-forward behavioral test for the EOS early exit (toy random
/// weights never emit EOS — handoff §4 — but the engine's `Forward` is injectable):
///   • disabled parity is covered by the untouched existing suites (both knobs default-off);
///   • dynamic τ (2b-2): cached==uncached identity at α>0, K-invariance, knob liveness;
///   • EOS early exit (2b-3): settles the block early with trim-identical output, K-invariant;
///   • combined + nBuf=2 graph-path exercise (dual-width window under both knobs).
final class WP2bTests: XCTestCase {

    var manifest: DenoisingLoopParityTests.Manifest!
    var traces: DenoisingLoopParityTests.Traces!
    var model: LLaDA2MoeModel!

    override func setUpWithError() throws {
        let dir = DenoisingLoopParityTests.fixtureDir
        let tracesURL = dir.appendingPathComponent("traces.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: tracesURL.path),
            "Loop fixtures missing at \(dir.path) — run: python3 Tools/generate_loop_fixtures.py")
        manifest = try JSONDecoder().decode(
            DenoisingLoopParityTests.Manifest.self,
            from: Data(contentsOf: dir.appendingPathComponent("manifest.json")))
        traces = try JSONDecoder().decode(
            DenoisingLoopParityTests.Traces.self, from: Data(contentsOf: tracesURL))
        let dm = DiffusionModel(config: manifest.config)
        try dm.loadWeights(from: dir.appendingPathComponent("weights.safetensors"))
        model = dm.model
    }

    // MARK: - 2b-2 dynamic τ

    /// Mask/graph-correctness proof at α>0: the cached path must equal the uncached path
    /// (the `.strict` full-window semantics anchor). Note: on toy random weights α=0.6 does
    /// not change trajectories (toy confidences fall outside the τ0→τ0(1−α) window), so knob
    /// liveness is proven by the deterministic synthetic test below, not here.
    func testDynamicTauCachedUncachedIdentity() throws {
        var failures: [String] = []
        for c in traces.cases {
            var p = c.params.toGenerationParams()
            p.dynamicTauAlpha = 0.6
            let un = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p)
            let ca = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p)
            if ca.finalSequence != un.finalSequence {
                failures.append("\(c.name): cached != uncached at α=0.6")
            }
            XCTAssertEqual(un.metrics.effectiveDynamicTauAlpha, 0.6, "\(c.name): echo missing")
        }
        XCTAssertTrue(failures.isEmpty,
            "WP-2b-2 cached/uncached identity failures (\(failures.count)):\n"
            + failures.joined(separator: "\n"))
        print("[WP-2b-2] cached==uncached at α=0.6 on \(traces.cases.count) cases")
    }

    /// Deterministic liveness proof: every position predicts token 7 at conf ≈ 0.6 — below the
    /// static τ0=0.7 forever (one top-1 unmask per step, ~B+1 steps) but above τ(t) once
    /// r_mask < ~0.71 at α=0.5 (τ(t) = 0.7(1 − 0.5(1 − r)) < 0.6), so the dynamic arm
    /// finishes the block in a burst (~7 steps). Same tokens, fewer steps — exactly the
    /// paper's mechanism.
    func testDynamicTauLivenessSynthetic() throws {
        let vocab = 1000
        // conf = e^c / (e^c + 999) = 0.6  ⇒  c = ln(0.6/0.4 · 999) ≈ 7.313
        let forward: DiffusionEngine.Forward = { windowIds, activeLen in
            var logits = [Float](repeating: 0, count: activeLen * vocab)
            for i in 0 ..< activeLen { logits[i * vocab + 7] = 7.313 }
            return MLXArray(logits).reshaped(1, activeLen, vocab)
        }
        func params(alpha: Float) -> GenerationParams {
            GenerationParams(
                threshold: 0.7, editingThreshold: 0.5,
                blockLength: 16, genLength: 16,
                maskId: 999, eosId: 998,
                dynamicTauAlpha: alpha)
        }
        let prompt = Array(1...16)
        let engine = DiffusionEngine(model: model, speculationK: 4)
        let base = engine.run(prompt: prompt, params: params(alpha: 0),
                              forward: forward, streamBlock: nil)
        let dyn = engine.run(prompt: prompt, params: params(alpha: 0.5),
                             forward: forward, streamBlock: nil)
        XCTAssertEqual(base.tokens, dyn.tokens, "dynamic τ changed the tokens (same argmax)")
        XCTAssertGreaterThanOrEqual(base.stepsPerBlock[0], 16,
            "static baseline should unmask one position per step")
        XCTAssertLessThanOrEqual(dyn.stepsPerBlock[0], 10,
            "τ(t) should accept the remaining positions in a burst once r_mask drops")
        print("[WP-2b-2] liveness: \(base.stepsPerBlock[0]) → \(dyn.stepsPerBlock[0]) steps "
            + "at α=0.5, identical tokens")
    }

    /// τ(t) is a per-step in-graph decision — output must not depend on the speculative
    /// batch size K.
    func testDynamicTauSpeculationInvariance() throws {
        let sampleNames = ["p8_q_noeos", "p20_s_noeos", "p16_q_eos"]
        for name in sampleNames {
            guard let c = traces.cases.first(where: { $0.name == name }) else { continue }
            var p = c.params.toGenerationParams()
            p.dynamicTauAlpha = 0.6
            let k1 = DiffusionEngine(model: model, speculationK: 1)
                .generate(prompt: c.prompt, params: p)
            let k4 = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p)
            XCTAssertEqual(k1.finalSequence, k4.finalSequence,
                "\(name): dynamic-τ output depends on speculationK")
            XCTAssertEqual(k1.stepsPerBlock, k4.stepsPerBlock,
                "\(name): dynamic-τ step counts depend on speculationK")
        }
        print("[WP-2b-2] speculation invariance holds at α=0.6 (K=1 vs K=4)")
    }

    // MARK: - 2b-3 EOS early exit (synthetic forward)

    /// Synthetic forward: every active position predicts EOS, all below τ_mask=0.7 so only the
    /// top-1 fallback fires (one unmask per step); the first generated position carries the
    /// highest confidence so EOS lands there first. Without early exit the block needs ~B
    /// steps; with it, the post-EOS fill settles the block in ~2. Output must be
    /// trim-identical ([eos]).
    private func syntheticEosForward(vocab: Int, eosId: Int, firstGenPos: Int)
        -> DiffusionEngine.Forward {
        { windowIds, activeLen in
            let W = windowIds.dim(windowIds.ndim - 1)
            var logits = [Float](repeating: 0, count: activeLen * vocab)
            for i in 0 ..< activeLen {
                let absPos = W - activeLen + i
                // conf ≈ 0.64 at the first generated position, ≈ 0.40 elsewhere (both < 0.7).
                logits[i * vocab + eosId] = absPos == firstGenPos ? 7.5 : 6.5
            }
            return MLXArray(logits).reshaped(1, activeLen, vocab)
        }
    }

    private func syntheticParams(eosEarlyExit: Bool) -> GenerationParams {
        GenerationParams(
            threshold: 0.7, editingThreshold: 0.5,
            eosEarlyStop: true,
            blockLength: 16, genLength: 16,
            maskId: 999, eosId: 998,
            eosEarlyExit: eosEarlyExit)
    }

    func testEosEarlyExitSettlesEarlyTrimIdentical() throws {
        let prompt = Array(1...16)   // exactly one prefill block; block 1 is all-generated
        let forward = syntheticEosForward(vocab: 1000, eosId: 998, firstGenPos: 16)

        let off = DiffusionEngine(model: model, speculationK: 4)
            .run(prompt: prompt, params: syntheticParams(eosEarlyExit: false),
                 forward: forward, streamBlock: nil)
        let on = DiffusionEngine(model: model, speculationK: 4)
            .run(prompt: prompt, params: syntheticParams(eosEarlyExit: true),
                 forward: forward, streamBlock: nil)

        XCTAssertEqual(off.tokens, [998], "baseline must trim to [eos]")
        XCTAssertEqual(on.tokens, off.tokens, "early exit changed the trimmed output")
        XCTAssertGreaterThanOrEqual(off.stepsPerBlock[0], 16,
            "baseline should need ~one step per position (top-1 fallback only)")
        XCTAssertLessThanOrEqual(on.stepsPerBlock[0], 3,
            "early exit should settle the block within ~2 steps after the EOS lands")
        XCTAssertTrue(on.metrics.effectiveEosEarlyExit)
        XCTAssertFalse(off.metrics.effectiveEosEarlyExit)
        print("[WP-2b-3] early exit: \(off.stepsPerBlock[0]) → \(on.stepsPerBlock[0]) steps, "
            + "trim-identical output")
    }

    /// The fill is a per-step in-graph decision applied at batch boundaries — K-invariant.
    func testEosEarlyExitSpeculationInvariance() throws {
        let prompt = Array(1...16)
        let forward = syntheticEosForward(vocab: 1000, eosId: 998, firstGenPos: 16)
        let k1 = DiffusionEngine(model: model, speculationK: 1)
            .run(prompt: prompt, params: syntheticParams(eosEarlyExit: true),
                 forward: forward, streamBlock: nil)
        let k4 = DiffusionEngine(model: model, speculationK: 4)
            .run(prompt: prompt, params: syntheticParams(eosEarlyExit: true),
                 forward: forward, streamBlock: nil)
        XCTAssertEqual(k1.finalSequence, k4.finalSequence)
        XCTAssertEqual(k1.stepsPerBlock, k4.stepsPerBlock)
        print("[WP-2b-3] early-exit speculation invariance holds (K=1 vs K=4)")
    }

    // MARK: - Combined + MultiBD graph path

    /// Both knobs together at nBuf=2 (τ_add=0.1): exercises the dual-width window under
    /// dynamic τ and the early-exit fill ops (toy weights never emit EOS, so this is a
    /// graph-shape + identity gate, not an EOS-behavior gate — that's the synthetic test).
    func testCombinedKnobsMultiBDIdentity() throws {
        var failures: [String] = []
        for c in traces.cases {
            var p = c.params.toGenerationParams()
            p.dynamicTauAlpha = 0.6
            p.eosEarlyStop = true
            p.eosEarlyExit = true
            p.nBuf = 2
            p.tauAdd = 0.1
            let un = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p)
            let ca = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p)
            if ca.finalSequence != un.finalSequence {
                failures.append("\(c.name): cached != uncached (combined knobs, nBuf=2)")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "WP-2b combined-knob identity failures (\(failures.count)):\n"
            + failures.joined(separator: "\n"))
        print("[WP-2b] combined dynamic-τ + early-exit at nBuf=2: cached==uncached on "
            + "\(traces.cases.count) cases")
    }
}
