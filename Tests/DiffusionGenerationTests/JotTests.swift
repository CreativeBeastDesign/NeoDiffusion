import XCTest
import MLX
import DiffusionCore
@testable import DiffusionModel
@testable import DiffusionGeneration

final class JotTests: XCTestCase {

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

    /// Verification that JOT disabled matches baseline exactly.
    func testJotParityWhenDisabled() throws {
        var failures: [String] = []
        for c in traces.cases {
            var p = c.params.toGenerationParams()
            p.jotEnabled = false
            
            let un = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p)
            let ca = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p)
                
            XCTAssertEqual(ca.finalSequence, un.finalSequence, "\(c.name): cached != uncached when jot is disabled")
            XCTAssertFalse(ca.metrics.effectiveJotEnabled)
        }
    }

    /// Liveness test: synthetic forward pass where positions predict stably.
    /// Verify that stable positions are frozen (added to frozen mask) and their predictions written.
    func testJotLivenessSynthetic() throws {
        let vocab = 1000
        let B = 16
        
        // Track the frozen masks received by the forward pass
        var observedFrozenMasks: [MLXArray] = []
        
        let forward: DiffusionEngine.Forward = { windowIds, activeLen, frozen in
            if let frozen = frozen {
                observedFrozenMasks.append(frozen)
            }
            
            let W = windowIds.dim(windowIds.ndim - 1)
            var logits = [Float](repeating: 0, count: activeLen * vocab)
            for i in 0 ..< activeLen {
                let absPos = W - activeLen + i
                if absPos == 15 {
                    // Low confidence to keep it masked, preventing block from settling early
                    logits[i * vocab + 7] = 0.0
                } else {
                    // Logits: predict token 7 at conf 0.95 (c ≈ 9.85)
                    logits[i * vocab + 7] = 9.85
                }
            }
            return MLXArray(logits).reshaped(1, activeLen, vocab)
        }
        
        let p = GenerationParams(
            threshold: 0.7, editingThreshold: 0.5,
            blockLength: B, genLength: B,
            maskId: 999, eosId: 998,
            jotEnabled: true,
            jotK: 2,
            jotThreshold: 0.9
        )
        
        let prompt = Array(repeating: 1, count: B)
        let engine = DiffusionEngine(model: model, speculationK: 1)
        let output = engine.run(prompt: prompt, params: p, forward: forward, streamBlock: nil)
        
        // We ran with jotK = 2.
        // Step 0: stableCount becomes 1, frozenMask is false.
        // Step 1: stableCount becomes 2 (since argmax token is 7 again), frozenMask becomes true.
        // Forward at Step 2 should receive a non-zero frozen mask!
        XCTAssertGreaterThan(observedFrozenMasks.count, 1, "Should run at least 2 steps")
        
        var foundFrozen = false
        for fm in observedFrozenMasks {
            let sum = fm.asType(.int32).sum().item(Int32.self)
            if sum > 0 {
                foundFrozen = true
            }
        }
        XCTAssertTrue(foundFrozen, "Expected JOT to freeze stable tokens and pass the frozen mask to forward")
        XCTAssertEqual(output.tokens[0..<15], Array(repeating: 7, count: 15))
        XCTAssertEqual(output.tokens[15], 999) // remains masked
    }

    /// Verify collision resolution: if a delta edit occurs at a frozen position,
    /// JOT unfreezes it and sets it back to mask.
    func testJotCollisionWithDeltaEdit() throws {
        let vocab = 1000
        let B = 16
        
        var stepCount = 0
        var observedFrozenMasks: [MLXArray] = []
        
        let forward: DiffusionEngine.Forward = { windowIds, activeLen, frozen in
            if let frozen = frozen {
                observedFrozenMasks.append(frozen)
            }
            
            let W = windowIds.dim(windowIds.ndim - 1)
            var logits = [Float](repeating: 0, count: activeLen * vocab)
            
            for i in 0 ..< activeLen {
                let absPos = W - activeLen + i
                if absPos == 15 {
                    // Low confidence to keep it masked, preventing block from settling early
                    logits[i * vocab + 7] = 0.0
                } else if absPos == 5 && stepCount >= 2 {
                    // Edit position 5 to token 8 on step 2+
                    logits[i * vocab + 8] = 9.85
                } else {
                    logits[i * vocab + 7] = 9.85
                }
            }
            
            stepCount += 1
            return MLXArray(logits).reshaped(1, activeLen, vocab)
        }
        
        let p = GenerationParams(
            threshold: 0.7, editingThreshold: 0.5,
            blockLength: B, genLength: B,
            maskId: 999, eosId: 998,
            jotEnabled: true,
            jotK: 2,
            jotThreshold: 0.9
        )
        
        let prompt = Array(repeating: 1, count: B)
        let engine = DiffusionEngine(model: model, speculationK: 1)
        let output = engine.run(prompt: prompt, params: p, forward: forward, streamBlock: nil)
        
        // Verify output tokens at position 5 is 8 (the edit resolved correctly after unfreezing)
        XCTAssertEqual(output.tokens[5], 8, "Edit collision failed to resolve")
        XCTAssertEqual(output.tokens[0..<5], Array(repeating: 7, count: 5))
        XCTAssertEqual(output.tokens[6..<15], Array(repeating: 7, count: 9))
    }
}
