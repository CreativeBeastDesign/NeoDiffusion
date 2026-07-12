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

    /// Faithful JOT (WP-3a v2) plumbing parity: with a freeze threshold no softmax probability
    /// can ever reach (2.0 > 1.0), *nothing* freezes, so the per-layer K/V hold is inert and the
    /// held-K/V forward must reproduce the cached baseline token-for-token. This proves the
    /// faithful path is wired without changing the trajectory when it has nothing to hold — the
    /// analogue of `testJotParityWhenDisabled` for the v2 mechanism. (A model-level check that
    /// held K/V actually suppress the v1 cascade is a bench measurement, not a unit test.)
    func testFaithfulJotParityWhenNothingFreezes() throws {
        for c in traces.cases {
            var pJot = c.params.toGenerationParams()
            pJot.jotEnabled = true
            pJot.jotFaithful = true
            pJot.jotThreshold = 2.0   // unreachable → no position is ever frozen

            var pBase = c.params.toGenerationParams()
            pBase.jotEnabled = false

            // Faithful JOT is gated to speculationK == 1.
            let jot = DiffusionEngine(model: model, speculationK: 1)
                .generateCached(prompt: c.prompt, params: pJot)
            let base = DiffusionEngine(model: model, speculationK: 1)
                .generateCached(prompt: c.prompt, params: pBase)

            XCTAssertTrue(jot.metrics.effectiveJotFaithful, "\(c.name): faithful flag not echoed")
            XCTAssertEqual(jot.finalSequence, base.finalSequence,
                "\(c.name): faithful JOT with no freezing must match the cached baseline")
        }
    }

    /// Option C (WP-3a §11) inert-parity: with an unreachable freeze threshold nothing freezes, so
    /// no frozen prefix ever forms and `subBlockCommit` never fires — the run must be byte-identical
    /// to the cached baseline. Proves the sub-block-commit plumbing is inert when it has no prefix.
    func testSubBlockCommitInertWhenNothingFreezes() throws {
        for c in traces.cases {
            var p = c.params.toGenerationParams()
            p.jotEnabled = true
            p.jotFaithful = true
            p.subBlockCommit = true
            p.subBlockMinPrefix = 2
            p.jotThreshold = 2.0   // unreachable → nothing freezes → no prefix commit

            let pBase = c.params.toGenerationParams()

            let out = DiffusionEngine(model: model, speculationK: 1)
                .generateCached(prompt: c.prompt, params: p)
            let base = DiffusionEngine(model: model, speculationK: 1)
                .generateCached(prompt: c.prompt, params: pBase)

            XCTAssertEqual(out.finalSequence, base.finalSequence,
                "\(c.name): Option C must be inert when nothing freezes")
        }
    }

    /// Option C end-to-end under freezing: with `jotK=1` + a low threshold, contiguous frozen
    /// prefixes form and `subBlockCommit` fires mid-block. The run must still complete and commit
    /// the full sequence, block-aligned and the same total length as the baseline (both fill every
    /// block with `eosEarlyStop` off). Exercises the prefix capture + window-shrink + slot-reshape
    /// path; a position/shape bug there would crash, hang, or change the length. Single case with a
    /// short gen length — the point is machinery correctness, not a sweep (that's the bench's job).
    func testSubBlockCommitCompletesUnderFreezing() throws {
        let c = try XCTUnwrap(traces.cases.first)
        var p = c.params.toGenerationParams()
        p.jotEnabled = true
        p.jotFaithful = true
        p.subBlockCommit = true
        p.subBlockMinPrefix = 4
        p.jotK = 1
        p.jotThreshold = 0.5
        p.eosEarlyStop = false
        p.genLength = 3 * p.blockLength   // a few blocks is enough to trigger + reassemble

        var base = c.params.toGenerationParams()
        base.eosEarlyStop = false
        base.genLength = 3 * p.blockLength

        let out = DiffusionEngine(model: model, speculationK: 1)
            .generateCached(prompt: c.prompt, params: p)
        let baseOut = DiffusionEngine(model: model, speculationK: 1)
            .generateCached(prompt: c.prompt, params: base)

        XCTAssertEqual(out.finalSequence.count, baseOut.finalSequence.count,
            "\(c.name): Option C must still commit every block (same total length)")
        // Sub-commits must reassemble into whole blocks, not ragged offsets.
        XCTAssertEqual(out.finalSequence.count % p.blockLength, 0,
            "\(c.name): committed length must stay block-aligned after sub-block commits")
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
                if absPos >= 14 {
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
            blockLength: B, genLength: B - 1, // generated tokens count is B - promptLength = 15
            maskId: 999, eosId: 998,
            jotEnabled: true,
            jotK: 2,
            jotThreshold: 0.9
        )
        
        let prompt = [1]
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
        XCTAssertEqual(Array(output.tokens[0..<13]), Array(repeating: 7, count: 13))
        XCTAssertEqual(output.tokens[13], 0) // fallback-unmasked position 14 (logits 0.0)
        XCTAssertEqual(output.tokens[14], 0) // fallback-unmasked position 15 (logits 0.0)
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
                if absPos >= 14 {
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
            blockLength: B, genLength: B - 1, // generated tokens count is B - promptLength = 15
            maskId: 999, eosId: 998,
            jotEnabled: true,
            jotK: 2,
            jotThreshold: 0.9
        )
        
        let prompt = [1]
        let engine = DiffusionEngine(model: model, speculationK: 1)
        let output = engine.run(prompt: prompt, params: p, forward: forward, streamBlock: nil)
        
        // Verify output tokens at position 5 is 8 (the edit resolved correctly after unfreezing)
        // absolute position 5 corresponds to output.tokens index 4.
        XCTAssertEqual(output.tokens[4], 8, "Edit collision failed to resolve")
        XCTAssertEqual(Array(output.tokens[0..<4]), Array(repeating: 7, count: 4))
        XCTAssertEqual(Array(output.tokens[5..<13]), Array(repeating: 7, count: 8))
        XCTAssertEqual(output.tokens[13], 0)
        XCTAssertEqual(output.tokens[14], 0)
    }

    func testBoolIndexing() throws {
        let x = MLXArray(0..<10).reshaped([5, 2])
        let mask = MLXArray([true, false, true, false, true])
        let indexed = x[mask]
        print("Indexed array shape:", indexed.shape)
        XCTAssertEqual(indexed.shape, [3, 2])
    }
}
