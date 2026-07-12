import XCTest
import MLX
import DiffusionCore
@testable import DiffusionModel
@testable import DiffusionGeneration

final class CreditDecodingTests: XCTestCase {

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

    /// Verification that Credit Decoding uncached output matches cached output.
    func testCreditDecodingCachedUncachedIdentity() throws {
        var failures: [String] = []
        for c in traces.cases {
            var p = c.params.toGenerationParams()
            p.creditDecodingEnabled = true
            p.creditAlpha = 0.5
            p.creditBeta = 0.9
            p.creditGamma = 0.5
            
            let un = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p)
            let ca = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p)
                
            if ca.finalSequence != un.finalSequence {
                failures.append("\(c.name): cached != uncached when credit decoding is enabled")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "Credit Decoding cached/uncached identity failures (\(failures.count)):\n"
            + failures.joined(separator: "\n"))
        print("[CreditDecoding] cached==uncached identity verified on \(traces.cases.count) cases")
    }

    /// Verification that Credit Decoding is speculation-invariant (K=1 vs K=4 outputs and step counts must match).
    func testCreditDecodingSpeculationInvariance() throws {
        let sampleNames = ["p8_q_noeos", "p20_s_noeos", "p16_q_eos"]
        for name in sampleNames {
            guard let c = traces.cases.first(where: { $0.name == name }) else { continue }
            var p = c.params.toGenerationParams()
            p.creditDecodingEnabled = true
            p.creditAlpha = 0.5
            p.creditBeta = 0.9
            p.creditGamma = 0.5

            let k1 = DiffusionEngine(model: model, speculationK: 1)
                .generate(prompt: c.prompt, params: p)
            let k4 = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p)
                
            XCTAssertEqual(k1.finalSequence, k4.finalSequence,
                "\(name): Credit Decoding output depends on speculationK")
            XCTAssertEqual(k1.stepsPerBlock, k4.stepsPerBlock,
                "\(name): Credit Decoding step counts depend on speculationK")
        }
        print("[CreditDecoding] speculation invariance holds (K=1 vs K=4)")
    }

    /// Liveness test: synthetic forward pass where positions predict underconfident (0.4) but stable.
    /// Without credit decoding, raw confidence is < 0.7, taking at least B steps (only top-1 fallback unmasks).
    /// With credit decoding, credit accumulation boosts the logits of token 7, letting it clear the threshold and settle the block early.
    func testCreditDecodingLivenessSynthetic() throws {
        let vocab = 1000
        let B = 16
        
        let forward: DiffusionEngine.Forward = { windowIds, activeLen, frozen in
            var logits = [Float](repeating: 0, count: activeLen * vocab)
            for i in 0 ..< activeLen {
                logits[i * vocab + 7] = 6.5
                // Other logits are 0.0, so P(7) = e^6.5 / (e^6.5 + 999) approx 0.40
            }
            return MLXArray(logits).reshaped(1, activeLen, vocab)
        }
        
        func getParams(enabled: Bool) -> GenerationParams {
            GenerationParams(
                threshold: 0.7, editingThreshold: 0.5,
                blockLength: B, genLength: B,
                maskId: 999, eosId: 998,
                creditDecodingEnabled: enabled,
                creditAlpha: 5.0,
                creditBeta: 0.9,
                creditGamma: 0.5
            )
        }
        
        let prompt = Array(1...B)
        let engine = DiffusionEngine(model: model, speculationK: 1)
        
        let base = engine.run(prompt: prompt, params: getParams(enabled: false), forward: forward, streamBlock: nil)
        let cred = engine.run(prompt: prompt, params: getParams(enabled: true), forward: forward, streamBlock: nil)
        
        XCTAssertEqual(base.tokens, cred.tokens, "Credit Decoding changed the tokens (same argmax)")
        XCTAssertGreaterThanOrEqual(base.stepsPerBlock[0], B, "Without credit decoding, should take at least B steps")
        XCTAssertLessThanOrEqual(cred.stepsPerBlock[0], 3, "With credit decoding, should settle the block in <= 3 steps")
        
        print("[CreditDecoding] liveness check: baseline \(base.stepsPerBlock[0]) steps -> credit decoding \(cred.stepsPerBlock[0]) steps")
    }
}
