import XCTest
import MLX
import DiffusionCore
@testable import DiffusionModel
@testable import DiffusionGeneration

/// WP-1b acceptance (training-free MultiBD, arXiv:2606.29215 Alg. 5).
///
/// There is no nBuf=2 reference trace — the reference implements SingleBD only — so these are
/// internal-consistency gates on the same loop fixtures as `DenoisingLoopParityTests`:
///   • disabled parity: nBuf=2 with activation off (τ_add=2.0) is byte-identical to nBuf=1;
///   • cached==uncached identity with activation *on*: the block-causal active-window mask is
///     proven against the full-window `.strict` mask (the semantics anchor);
///   • K-invariance with events: activation/commit timing depends only on logical steps;
///   • sync budget, in-order streaming, EOS-cancel, metrics accounting.
final class MultiBDParityTests: XCTestCase {

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

    private func multiParams(
        _ c: DenoisingLoopParityTests.Case, tauAdd: Float, tauSemi: Float = 0.9
    ) -> GenerationParams {
        var p = c.params.toGenerationParams()
        p.nBuf = 2
        p.tauAdd = tauAdd
        p.tauSemi = tauSemi
        return p
    }

    /// The hard gate: nBuf=2 with activation disabled (τ_add=2.0 — progress ≤ 1 never exceeds
    /// it) must be token-identical to nBuf=1 on BOTH the cache-disabled and cached paths,
    /// including steps/block and the logical-step accounting.
    func testMultiBDDisabledParity() throws {
        var failures: [String] = []
        for c in traces.cases {
            let single = c.params.toGenerationParams()
            let multi = multiParams(c, tauAdd: 2.0)
            let s1 = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: single)
            let m1 = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: multi)
            if m1.finalSequence != s1.finalSequence {
                failures.append("\(c.name): uncached nBuf2/τ_add=2 final_x != nBuf1")
            }
            if m1.stepsPerBlock != s1.stepsPerBlock {
                failures.append("\(c.name): uncached stepsPerBlock \(m1.stepsPerBlock) "
                    + "!= \(s1.stepsPerBlock)")
            }
            if m1.metrics.logicalStepsTotal != m1.stepsPerBlock.reduce(0, +) {
                failures.append("\(c.name): logicalStepsTotal != sum(stepsPerBlock) at nBuf1-equivalent")
            }
            let s2 = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: single)
            let m2 = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: multi)
            if m2.finalSequence != s2.finalSequence {
                failures.append("\(c.name): cached nBuf2/τ_add=2 final_x != nBuf1")
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "WP-1b disabled-parity failures (\(failures.count)):\n"
            + failures.joined(separator: "\n"))
        print("[WP-1b] disabled parity: nBuf=2 + τ_add=2.0 identical to nBuf=1 on "
            + "\(traces.cases.count) cases, both paths")
    }

    /// Mask-correctness proof: with activation exercised, the cached path (committed KV +
    /// block-causal active-window mask) must equal the uncached path (full window under the
    /// `.strict` mask — the trusted MultiBD semantics by construction).
    func testMultiBDCachedUncachedIdentity() throws {
        var failures: [String] = []
        var activated = 0
        for c in traces.cases {
            for tauAdd: Float in [0.1, 0.5] {
                let p = multiParams(c, tauAdd: tauAdd)
                let un = DiffusionEngine(model: model, speculationK: 4)
                    .generate(prompt: c.prompt, params: p)
                let ca = DiffusionEngine(model: model, speculationK: 4)
                    .generateCached(prompt: c.prompt, params: p)
                if ca.finalSequence != un.finalSequence {
                    failures.append("\(c.name) τ_add=\(tauAdd): cached != uncached "
                        + "(activations un \(un.metrics.activationSteps.count) "
                        + "ca \(ca.metrics.activationSteps.count))")
                }
                activated += un.metrics.activationSteps.count
            }
        }
        XCTAssertTrue(failures.isEmpty,
            "WP-1b cached/uncached identity failures (\(failures.count)):\n"
            + failures.joined(separator: "\n"))
        XCTAssertGreaterThan(activated, 0,
            "no case ever activated a second block — the identity test exercised nothing")
        print("[WP-1b] cached==uncached with activation on: \(traces.cases.count) cases × 2 τ_add, "
            + "\(activated) activation events exercised")
    }

    /// Scheduling events are per-step in-graph decisions applied at batch boundaries, so the
    /// output must be independent of the speculative batch size K.
    func testMultiBDSpeculationInvariance() throws {
        let sampleNames = ["p8_q_noeos", "p20_s_noeos", "p16_q_eos"]
        for name in sampleNames {
            guard let c = traces.cases.first(where: { $0.name == name }) else { continue }
            for tauAdd: Float in [0.1, 0.5] {
                let p = multiParams(c, tauAdd: tauAdd)
                let k1 = DiffusionEngine(model: model, speculationK: 1)
                    .generate(prompt: c.prompt, params: p)
                let k4 = DiffusionEngine(model: model, speculationK: 4)
                    .generate(prompt: c.prompt, params: p)
                XCTAssertEqual(k1.finalSequence, k4.finalSequence,
                    "\(name) τ_add=\(tauAdd): output depends on speculationK")
                XCTAssertEqual(k1.metrics.activationSteps, k4.metrics.activationSteps,
                    "\(name) τ_add=\(tauAdd): activation timing depends on speculationK")
            }
        }
        print("[WP-1b] speculation invariance holds with activation events (K=1 vs K=4)")
    }

    /// Sync budget with events: every phase still costs one readback per ≤K steps; each event
    /// (activation or commit) may split a batch, and each commit adds its id readback.
    func testMultiBDSyncBudget() throws {
        let K = 4
        for c in traces.cases {
            let p = multiParams(c, tauAdd: 0.1)
            let out = DiffusionEngine(model: model, speculationK: K)
                .generate(prompt: c.prompt, params: p)
            let L = out.metrics.logicalStepsTotal
            let blocks = out.stepsPerBlock.count
            let activations = out.metrics.activationSteps.count
            let budget = (L + K - 1) / K + activations + 2 * blocks
            XCTAssertLessThanOrEqual(out.syncPoints, budget,
                "\(c.name): \(out.syncPoints) syncs exceeds budget \(budget) "
                + "(L \(L), activations \(activations), blocks \(blocks))")
        }
        print("[WP-1b] sync budget holds at τ_add=0.1 (events counted)")
    }

    /// Streaming contract (M7 server): `streamBlock` fires exactly once per committed block,
    /// strictly in block order, each carrying that block's ids.
    func testMultiBDInOrderStreaming() throws {
        for c in traces.cases {
            let p = multiParams(c, tauAdd: 0.1)
            var streamed: [[Int]] = []
            let out = DiffusionEngine(model: model, speculationK: 4)
                .generateCached(prompt: c.prompt, params: p) { streamed.append($0) }
            XCTAssertEqual(streamed.count, out.blockCommits.count,
                "\(c.name): streamed \(streamed.count) blocks, committed \(out.blockCommits.count)")
            let B = c.params.blockLength
            for (b, ids) in streamed.enumerated() {
                XCTAssertEqual(ids, Array(out.blockCommits[b].suffix(B)),
                    "\(c.name): streamed block \(b) != committed block \(b)")
            }
        }
        print("[WP-1b] in-order block streaming preserved with MultiBD activation")
    }

    /// EOS early-stop with a live trailing slot: the trailing block is cancelled — nothing is
    /// streamed or committed past the EOS block, and the run terminates cleanly.
    func testMultiBDEosCancelsTrailing() throws {
        guard let c = traces.cases.first(where: { $0.name == "p16_q_eos" }) else {
            throw XCTSkip("eos fixture case missing")
        }
        var p = multiParams(c, tauAdd: 0.1)
        p.eosEarlyStop = true
        var streamed: [[Int]] = []
        let out = DiffusionEngine(model: model, speculationK: 4)
            .generateCached(prompt: c.prompt, params: p) { streamed.append($0) }
        XCTAssertEqual(streamed.count, out.blockCommits.count)
        if let eosBlock = out.metrics.eosBlockIndex {
            XCTAssertEqual(out.blockCommits.count, eosBlock + 1,
                "blocks committed past the EOS block despite eos_early_stop")
        }
        XCTAssertEqual(out.finalSequence.count, out.blockCommits.last?.count ?? 0,
            "finalSequence extends past the last committed block (trailing leaked)")
        if out.tokens.contains(c.params.eosId) {
            XCTAssertEqual(out.tokens.last, c.params.eosId,
                "tokens must be trimmed at the first eos (inclusive)")
        }
        print("[WP-1b] EOS early-stop cancels the trailing slot cleanly "
            + "(\(out.blockCommits.count) blocks, eos block \(String(describing: out.metrics.eosBlockIndex)))")
    }

    /// Metrics accounting: the dual-active bookkeeping invariant and the effective-parameter
    /// echoes (provenance rule from the elastic-cache campaign, F7).
    func testMultiBDMetricsAccounting() throws {
        // Exclude true-EOS cases only ("p16_q_eos"; note "noeos" contains "eos" as a substring):
        // a cancelled trailing slot legitimately breaks the dual-step identity.
        for c in traces.cases where !c.name.hasSuffix("_eos") || c.name.hasSuffix("_noeos") {
            let p = multiParams(c, tauAdd: 0.3)
            let out = DiffusionEngine(model: model, speculationK: 4)
                .generate(prompt: c.prompt, params: p)
            let m = out.metrics
            // Without EOS cancellation every dual step belongs to exactly two committed blocks.
            XCTAssertEqual(out.stepsPerBlock.reduce(0, +) - m.logicalStepsTotal, m.dualActiveSteps,
                "\(c.name): sum(stepsPerBlock) − logicalStepsTotal != dualActiveSteps")
            XCTAssertEqual(m.effectiveNBuf, 2)
            XCTAssertEqual(m.effectiveTauAdd, 0.3)
            XCTAssertEqual(m.effectiveTauSemi, 0.9)
            XCTAssertEqual(m.effectiveSpeculationK, 4)
            XCTAssertEqual(m.trailingStarvedStepsPerBlock.count, out.stepsPerBlock.count)
            if !m.activationSteps.isEmpty {
                XCTAssertGreaterThan(m.dualActiveSteps, 0,
                    "\(c.name): activation fired but no dual-active steps recorded")
            }
        }
        print("[WP-1b] metrics accounting invariants hold (dual-step identity, effective echoes)")
    }
}

/// WP-1b unit tests that need no model fixtures: the active-window mask and the BlockBuffer
/// MultiBD state machine.
final class MultiBDUnitTests: XCTestCase {

    /// The active-window mask must equal the corresponding rows/columns of the full-window
    /// `.strict` mask: prefix columns all-allowed, front row blind to trailing columns,
    /// trailing row all-allowed.
    func testActiveWindowMask() throws {
        let B = 4, P = 8, A = 2 * B
        let mask = BlockDiffusionMask.activeWindowMask(
            prefixLen: P, activeLen: A, blockLength: B)
        XCTAssertEqual(mask.shape, [1, 1, A, P + A])

        let full = BlockDiffusionMask.build(totalLength: P + A, blockLength: B)
        let expected = full[0..., 0..., P..., 0...]  // rows of the two active blocks
        XCTAssertTrue((mask .== expected).all().item(Bool.self),
            "active-window mask != full-window strict mask slice")

        let vals = mask.reshaped([A, P + A]).asArray(Float.self)
        for q in 0 ..< A {
            for k in 0 ..< (P + A) {
                let v = vals[q * (P + A) + k]
                let qBlock = q / B          // 0 = front, 1 = trailing
                let allowed = k < P || (P + (qBlock + 1) * B) > k
                XCTAssertEqual(v, allowed ? 0 : -Float.infinity,
                    "mask[\(q),\(k)]: expected \(allowed ? "0" : "-inf"), got \(v)")
            }
        }
    }

    /// Single-active-block window: the mask degenerates to all-zero (the `mask: nil` fast path).
    func testActiveWindowMaskSingleBlock() throws {
        let mask = BlockDiffusionMask.activeWindowMask(prefixLen: 8, activeLen: 4, blockLength: 4)
        XCTAssertTrue((mask .== MLXArray(Float(0))).all().item(Bool.self))
    }

    func testBlockBufferTwoActive() throws {
        var buffer = BlockBuffer(nBuf: 2)
        let s0 = buffer.activate(blockIndex: 3)
        let s1 = buffer.activate(blockIndex: 4)
        XCTAssertEqual(buffer.activeCount, 2)
        XCTAssertEqual(buffer.activeSlotIndices, [s0, s1])

        buffer.markSettled(slotIndex: s0)          // front settles first — allowed
        buffer.markCommitted(slotIndex: s0)
        XCTAssertEqual(buffer.lastCommittedBlock, 3)
        XCTAssertEqual(buffer.activeSlotIndices, [s1])

        let s2 = buffer.activate(blockIndex: 5)    // reuses the inCache slot
        XCTAssertEqual(buffer.activeCount, 2)
        XCTAssertEqual(buffer.activeSlotIndices.map { buffer.slots[$0].blockIndex }, [4, 5])

        buffer.cancel(slotIndex: s2)               // EOS cancel: active → dummy
        XCTAssertEqual(buffer.activeCount, 1)
        XCTAssertNil(buffer.slots[s2].blockIndex)
        XCTAssertEqual(buffer.slots[s2].state, .dummy)
    }
}
