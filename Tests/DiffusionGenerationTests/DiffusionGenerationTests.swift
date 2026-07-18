import XCTest
import MLX
import MLXNN
import DiffusionModel
@testable import DiffusionGeneration

/// Smoke coverage for the real engine entry points on a random-weight toy model — the
/// structural contract of `DiffusionEngine.Output` (incl. the M6 `Metrics`), not parity
/// (that is `DenoisingLoopParityTests`' job, against reference fixtures).
///
/// Replaced the legacy random-latent `DiffusionGeneration` stub test when the stub was
/// deleted (handoff §3 item 1 — bench and server no longer referenced it).
final class DiffusionGenerationTests: XCTestCase {
    static let blockLength = 8
    static let genLength = 24

    private func makeEngine(instrument: Bool = false) -> DiffusionEngine {
        let config = LLaDA2MoeConfig(
            vocabSize: 200, hiddenSize: 64, intermediateSize: 128, numHiddenLayers: 2,
            numAttentionHeads: 4, numKeyValueHeads: 2, rmsNormEps: 1e-6,
            ropeTheta: 600_000, padTokenId: 198, numExperts: 4, numSharedExperts: 1,
            numExpertsPerTok: 2, nGroup: 2, topkGroup: 1, moeIntermediateSize: 32,
            firstKDenseReplace: 1, headDim: 16, partialRotaryFactor: 0.5)
        let container = DiffusionModel(config: config)
        MLXRandom.seed(7)
        var randomized: [String: MLXArray] = [:]
        for (key, value) in container.model.parameters().flattened() {
            randomized[key] = MLXRandom.normal(value.shape) * 0.05
        }
        try! container.model.update(
            parameters: ModuleParameters.unflattened(randomized), verify: .all)
        return DiffusionEngine(model: container.model, instrument: instrument)
    }

    private func makeParams() -> GenerationParams {
        GenerationParams.mode(
            .q, blockLength: Self.blockLength, genLength: Self.genLength,
            maskId: 199, eosId: 198)
    }

    func testEngineOutputContract() {
        let engine = makeEngine()
        let prompt = [3, 5, 7, 11, 13]  // tail shares a block with generation start
        var streamed: [[Int]] = []
        let output = engine.generate(prompt: prompt, params: makeParams()) { streamed.append($0) }

        let blocks = output.blockCommits.count
        XCTAssertGreaterThan(blocks, 0)
        // Random toy weights never emit eos, so the full padded window commits.
        XCTAssertEqual(output.finalSequence.count,
                       ((prompt.count + Self.genLength + Self.blockLength - 1)
                        / Self.blockLength) * Self.blockLength)
        XCTAssertEqual(Array(output.finalSequence.prefix(prompt.count)), prompt)
        XCTAssertEqual(streamed.count, blocks)

        // Metrics contract: one entry per block; every block runs ≥1 step; the honest
        // forward count includes speculative overshoot, so it is ≥ the logical step count.
        XCTAssertEqual(output.stepsPerBlock.count, blocks)
        XCTAssertEqual(output.metrics.postStepsPerBlock.count, blocks)
        XCTAssertTrue(output.stepsPerBlock.allSatisfy { $0 >= 1 })
        XCTAssertGreaterThanOrEqual(
            output.metrics.forwardsEvaluated, output.stepsPerBlock.reduce(0, +))
        XCTAssertEqual(output.metrics.prefillSeconds, 0)  // cache-disabled path: no prefill
    }

    func testCachedMatchesUncachedWithInstrumentation() {
        let engine = makeEngine(instrument: true)
        let prompt = Array(1 ... 10).map { $0 * 3 % 150 }
        let params = makeParams()

        let uncached = engine.generate(prompt: prompt, params: params)
        let cached = engine.generateCached(prompt: prompt, params: params)

        // Instrumentation must not perturb tokens (M5(b) invariant re-checked under
        // instrument: true, which adds evals at phase boundaries).
        XCTAssertEqual(cached.tokens, uncached.tokens)
        XCTAssertEqual(cached.finalSequence, uncached.finalSequence)

        // Cached path counts the prompt-prefill + per-commit capture forwards.
        let prefillBlocks = prompt.count / Self.blockLength
        XCTAssertEqual(
            cached.metrics.forwardsEvaluated - prefillBlocks - cached.blockCommits.count,
            uncached.metrics.forwardsEvaluated)
        XCTAssertGreaterThan(cached.metrics.denoiseSeconds, 0)
        XCTAssertGreaterThan(cached.metrics.commitSeconds, 0)

        // Step 3 (final-plan §2): the sub-phase timers must actually BITE under instrument — a
        // mis-wired timer reads 0 and looks like "phase free" (the F-m failure mode; only a
        // behavioural guard, not a timing one, catches it). Every phase positive, and their sum
        // contained within denoiseSeconds (non-overlapping sub-intervals of the timed phase).
        for (name, secs) in [
            ("forward", cached.metrics.forwardSeconds),
            ("sampler", cached.metrics.samplerSeconds),
            ("selection", cached.metrics.selectionSeconds),
            ("loopControl", cached.metrics.loopControlSeconds),
        ] {
            XCTAssertGreaterThan(secs, 0, "\(name)Seconds must be > 0 under instrument")
        }
        let subSum = cached.metrics.forwardSeconds + cached.metrics.samplerSeconds
            + cached.metrics.selectionSeconds + cached.metrics.loopControlSeconds
        XCTAssertLessThanOrEqual(subSum, cached.metrics.denoiseSeconds + 1e-6)
    }

    /// Step 3 (final-plan §2, guardrail §8): the diagnostic must be INSTRUMENT-ONLY. The served
    /// path (`instrument: false`) adds no sub-phase evals, so its output must be byte-identical to
    /// the instrumented run's, and its sub-phase timers must be exactly 0 (never populated).
    /// Mirrors `ModuleAblationTests.testNoneIsBitIdenticalToBaseline`.
    func testInstrumentTimersAreDiagnosticOnly() {
        let prompt = Array(1 ... 10).map { $0 * 3 % 150 }
        let params = makeParams()

        let plain = makeEngine(instrument: false).generate(prompt: prompt, params: params)
        let instrumented = makeEngine(instrument: true).generate(prompt: prompt, params: params)

        // Instrument-gated evals must not perturb the tokens or the committed sequence.
        XCTAssertEqual(plain.tokens, instrumented.tokens)
        XCTAssertEqual(plain.finalSequence, instrumented.finalSequence)

        // Off the served path the timers are never touched.
        XCTAssertEqual(plain.metrics.forwardSeconds, 0)
        XCTAssertEqual(plain.metrics.samplerSeconds, 0)
        XCTAssertEqual(plain.metrics.selectionSeconds, 0)
        XCTAssertEqual(plain.metrics.loopControlSeconds, 0)
    }
}
