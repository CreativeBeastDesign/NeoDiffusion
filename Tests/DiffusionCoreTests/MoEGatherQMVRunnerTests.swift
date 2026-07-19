import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import DiffusionCore

/// Step 2 of the kernel plan (`Plans/step5-kernel-handoff.md`): correctness tests for the
/// `MoEGatherQMVRunner` walking-skeleton port of `affine_gather_qmv_fast`, behind the
/// default-off `MoEFusedQMVConfig.enabled` flag. Both paths run the *same* quantized weights,
/// so 4-bit quantization error cancels — the tolerance below measures kernel arithmetic only
/// (the port mirrors stock's FP32-accumulate order, so differences should be near-exact).
final class MoEGatherQMVRunnerTests: XCTestCase {

    // Toy dims: the kernel requires H and I to be multiples of 512 (qmv_fast block size);
    // group size 64 divides both.
    static let E = 8
    static let H = 512
    static let I = 512
    static let K = 2
    static let T = 4

    /// Builds a toy `SwitchGLU` with random (non-zero) weights — `SwitchGLU`'s own init zero-
    /// fills — then quantizes it the same way `LLaDAMoEDispatchBench` does
    /// (`testCaptureMoEBlockTrace`): 4-bit affine, group 64, `SwitchLinear` submodules only.
    /// Weight replacement follows `CoreFixtureTests`' `update(parameters:verify:)` pattern (the
    /// property-wrapper setter fatalErrors on a direct second assignment).
    /// - Parameter weightDType: dtype the raw (pre-quantization) weights are cast to before
    ///   `MLXNN.quantize` runs. Quantizing float32 weights yields float32 scales/biases;
    ///   quantizing float16 weights yields float16 scales/biases (runner doc comment, `dtype
    ///   contract`). Defaults to `.float32` — `MLXRandom.normal`'s native dtype — preserving the
    ///   original tests' behavior exactly.
    private func makeToyGLU(seed: UInt64, weightDType: DType = .float32) -> SwitchGLU {
        let glu = SwitchGLU(hiddenSize: Self.H, intermediateSize: Self.I, numExperts: Self.E)
        MLXRandom.seed(seed)
        let params: [String: MLXArray] = [
            "gate_proj.weight": MLXRandom.normal([Self.E, Self.I, Self.H]).asType(weightDType),
            "up_proj.weight": MLXRandom.normal([Self.E, Self.I, Self.H]).asType(weightDType),
            "down_proj.weight": MLXRandom.normal([Self.E, Self.H, Self.I]).asType(weightDType),
        ]
        try! glu.update(parameters: ModuleParameters.unflattened(params), verify: .all)
        MLXNN.quantize(
            model: glu, groupSize: 64, bits: 4, mode: .affine,
            filter: { _, module in module is SwitchLinear })
        eval(glu)
        return glu
    }

    private func toyInputs(
        indices: [Int32]? = nil, xDType: DType = .float16
    ) -> (x: MLXArray, indices: MLXArray) {
        MLXRandom.seed(7)
        let x = MLXRandom.normal([Self.T, Self.H]).asType(xDType)
        let idx = indices ?? (0 ..< (Self.T * Self.K)).map { Int32($0 % Self.E) }
        let indicesArr = MLXArray(idx).reshaped([Self.T, Self.K])
        eval(x, indicesArr)
        return (x, indicesArr)
    }

    /// Relative equivalence gate vs stock. Since iteration 6 (VPT=8) the kernel spans 8 input
    /// values per thread, not stock's 16 — the accumulation ORDER differs, so bit-exactness is
    /// gone by design and only FP-reorder noise remains (observed ~2e-5 relative). The gate is
    /// relative because these toy weights are unscaled N(0,1) — outputs are O(500), where an
    /// absolute 1e-3 would silently demand bit-exactness. The production-shape gate
    /// (`testFusedQMVRunnerMini`) stays absolute 1e-3 and passes with ~100x headroom.
    private func assertMatchesStock(
        _ fused: MLXArray, _ stock: MLXArray, rtol: Float = 1e-4,
        _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        let maxDelta = MLX.abs(fused.asType(.float32) - stock.asType(.float32))
            .max().item(Float.self)
        let scale = MLX.abs(stock.asType(.float32)).max().item(Float.self)
        XCTAssertLessThan(maxDelta, rtol * max(scale, 1e-6), message, file: file, line: line)
    }

    override func tearDown() {
        // Never let one test's flag flip leak into another.
        MoEFusedQMVConfig.enabled = false
        super.tearDown()
    }

    func testFlagDefaultOff() throws {
        try XCTSkipIf(
            ProcessInfo.processInfo.environment["NEODIFFUSION_FUSED_QMV"] == "1",
            "env var forces the flag on for this process; default-off can't be observed here")
        XCTAssertFalse(MoEFusedQMVConfig.enabled, "fused gather-QMV path must default off")
    }

    /// Core Step 2 gate: flag-on output matches flag-off (stock `gatherQuantizedMM`) output on
    /// identical quantized weights, and the fused path provably ran (dispatch counter).
    func testFusedMatchesStock() throws {
        let glu = makeToyGLU(seed: 3)
        let (x, indices) = toyInputs()

        let stock = glu(x, indices: indices)
        eval(stock)

        MoEFusedQMVConfig.enabled = true
        defer { MoEFusedQMVConfig.enabled = false }
        let before = MoEFusedQMVConfig.dispatchCount
        let fused = glu(x, indices: indices)
        eval(fused)

        XCTAssertEqual(
            MoEFusedQMVConfig.dispatchCount, before + 1,
            "fused path did not dispatch — eligibility fell back to stock, test proves nothing")
        XCTAssertEqual(fused.shape, [Self.T, Self.K, Self.H])
        XCTAssertEqual(fused.dtype, stock.dtype)

        assertMatchesStock(fused, stock, "fused kernel diverges from stock gatherQuantizedMM")
        XCTAssertFalse(
            allClose(stock, MLXArray.zeros(like: stock)).item(Bool.self),
            "stock output is all zeros — comparison is vacuous")
    }

    /// Same gate under adversarial routing: every pair hits the same expert (max reuse), then
    /// each pair a distinct expert (max spread).
    func testFusedMatchesStockAdversarialIndices() throws {
        let glu = makeToyGLU(seed: 11)
        for pattern in [
            [Int32](repeating: 3, count: Self.T * Self.K),
            (0 ..< (Self.T * Self.K)).map { Int32($0 % Self.E) }.reversed().map { $0 },
        ] {
            let (x, indices) = toyInputs(indices: pattern)
            let stock = glu(x, indices: indices)
            MoEFusedQMVConfig.enabled = true
            let fused = glu(x, indices: indices)
            MoEFusedQMVConfig.enabled = false
            eval(stock, fused)
            assertMatchesStock(fused, stock, "divergence for index pattern \(pattern)")
        }
    }

    /// Ineligible dims (H=64 not a multiple of 512) must silently fall back to stock — same
    /// output, no fused dispatch.
    func testIneligibleShapeFallsBack() throws {
        let glu = SwitchGLU(hiddenSize: 64, intermediateSize: 64, numExperts: 4)
        MLXRandom.seed(9)
        let params: [String: MLXArray] = [
            "gate_proj.weight": MLXRandom.normal([4, 64, 64]),
            "up_proj.weight": MLXRandom.normal([4, 64, 64]),
            "down_proj.weight": MLXRandom.normal([4, 64, 64]),
        ]
        try! glu.update(parameters: ModuleParameters.unflattened(params), verify: .all)
        MLXNN.quantize(
            model: glu, groupSize: 64, bits: 4, mode: .affine,
            filter: { _, module in module is SwitchLinear })

        let x = MLXRandom.normal([2, 64]).asType(.float16)
        let indices = MLXArray([0, 1, 2, 3] as [Int32]).reshaped([2, 2])

        let stock = glu(x, indices: indices)
        MoEFusedQMVConfig.enabled = true
        defer { MoEFusedQMVConfig.enabled = false }
        let before = MoEFusedQMVConfig.dispatchCount
        let flagged = glu(x, indices: indices)
        eval(stock, flagged)

        XCTAssertEqual(MoEFusedQMVConfig.dispatchCount, before, "ineligible shape must not dispatch")
        XCTAssertTrue(allClose(stock, flagged).item(Bool.self))
    }

    /// Flag off must run the stock `gatherQuantizedMM` path — non-zero output, deterministic
    /// across calls.
    /// THE definitive mechanism test for the 2026-07-19 dtype finding (logbook post-close
    /// addendum II): with f16 weights (→ f16 scales) and f16 x — TRUE production dtypes — the
    /// STOCK path promotes (f16, f16) → f16 and pays no scale casts. Pre-casting the resident
    /// scales/biases to f32 once (exactly what `NEODIFFUSION_PRECAST_SCALES` does at load)
    /// flips the promotion: output dtype becomes f32, and the values genuinely change (higher-
    /// precision intermediates) while staying close — precision shift, not corruption. Every
    /// link of the causal chain (resident dtype → promotion → output dtype → numerics) is
    /// asserted here; the speed link is the env-gated `testPrecastScalesMiniTiming`.
    /// The fused-kernel flag stays off throughout — this is about STOCK behavior.
    func testPrecastScalesPromotionMechanism() throws {
        let glu = makeToyGLU(seed: 31, weightDType: .float16)
        let (x, indices) = toyInputs(xDType: .float16)
        XCTAssertFalse(MoEFusedQMVConfig.enabled, "precondition: stock path only")
        let dispatchesBefore = MoEFusedQMVConfig.dispatchCount

        guard let g = glu.gateProj as? QuantizedSwitchLinear,
              let u = glu.upProj as? QuantizedSwitchLinear,
              let d = glu.downProj as? QuantizedSwitchLinear,
              let bG = g.biases, let bU = u.biases, let bD = d.biases
        else { return XCTFail("toy GLU is not affine-quantized") }
        XCTAssertEqual(g.scales.dtype, .float16, "f16 weights must yield f16 scales")

        // Arm A — stock, f16 resident scales: promote(f16, f16) = f16, the _half_ gather.
        let outA = glu(x, indices: indices)
        eval(outA)
        XCTAssertEqual(outA.dtype, .float16,
            "stock at production dtypes must output f16 — if this fails, the _half_-variant "
            + "finding is wrong and the campaign records need re-opening")

        // Pre-cast resident scales/biases to f32 — the loader flag's exact mechanism.
        let precast: [String: MLXArray] = [
            "gate_proj.scales": g.scales.asType(.float32),
            "gate_proj.biases": bG.asType(.float32),
            "up_proj.scales": u.scales.asType(.float32),
            "up_proj.biases": bU.asType(.float32),
            "down_proj.scales": d.scales.asType(.float32),
            "down_proj.biases": bD.asType(.float32),
        ]
        try glu.update(parameters: ModuleParameters.unflattened(precast), verify: .noUnusedKeys)
        eval(glu)

        // Arm B — identical quantized values, f32 resident scales: promotion flips to f32.
        let outB = glu(x, indices: indices)
        eval(outB)
        XCTAssertEqual(outB.dtype, .float32,
            "f32 resident scales must flip the promotion — astype short-circuit means stock "
            + "was never casting at f16/f16")
        XCTAssertEqual(
            MoEFusedQMVConfig.dispatchCount, dispatchesBefore,
            "fused path must not have run — this test is about stock promotion")

        // Numerics: a real precision change (not bit-preserving), but close (not corruption).
        let a32 = outA.asType(.float32)
        let maxDelta = MLX.abs(outB - a32).max().item(Float.self)
        let scale = MLX.abs(a32).max().item(Float.self)
        XCTAssertFalse(
            allClose(outB, a32, rtol: 0.0, atol: 0.0).item(Bool.self),
            "outputs are bit-identical — then promotion does NOT change numerics and the "
            + "pre-cast A/B's trajectory divergence needs another explanation")
        XCTAssertLessThan(
            maxDelta, 2e-2 * max(scale, 1e-6),
            "f16→f32 promotion should shift values by ~f16-rounding magnitude only")
        print("[precast-mechanism] maxΔ = \(maxDelta) (relative \(maxDelta / max(scale, 1e-6))) "
            + "— precision shift confirmed, dtype f16→f32 confirmed")
    }

    func testFlagOffPathUnchanged() throws {
        let glu = makeToyGLU(seed: 5)
        let (x, indices) = toyInputs()

        XCTAssertFalse(MoEFusedQMVConfig.enabled, "precondition: flag must be off for this test")

        let out1 = glu(x, indices: indices)
        let out2 = glu(x, indices: indices)
        eval(out1, out2)

        XCTAssertEqual(out1.shape, [Self.T, Self.K, Self.H])
        XCTAssertFalse(
            allClose(out1, MLXArray.zeros(like: out1)).item(Bool.self),
            "stock path should not produce all-zero output on random weights/inputs")
        XCTAssertTrue(
            allClose(out1, out2).item(Bool.self),
            "stock path must be deterministic across repeated calls on identical inputs")
    }

    // MARK: - Step 3: broadened correctness evidence

    /// Seed × activation-dtype sweep on the eligible toy dims. Weights are quantized straight
    /// from `MLXRandom.normal`'s native float32 (the `makeToyGLU` default), so the `.float32`
    /// x arm is the AGENTS.md "toy-config FP32 parity" gate for this kernel end to end: float32
    /// weights → float32 scales → float32 activations, no fp16 anywhere in the path. The
    /// `.float16` arm repeats the original single-seed gate at two more seeds.
    func testSeedDtypeSweep() throws {
        for seed: UInt64 in [3, 17, 42] {
            let glu = makeToyGLU(seed: seed)
            for xDType: DType in [.float32, .float16] {
                let (x, indices) = toyInputs(xDType: xDType)

                let stock = glu(x, indices: indices)
                eval(stock)

                MoEFusedQMVConfig.enabled = true
                let before = MoEFusedQMVConfig.dispatchCount
                let fused = glu(x, indices: indices)
                eval(fused)
                MoEFusedQMVConfig.enabled = false

                XCTAssertEqual(
                    MoEFusedQMVConfig.dispatchCount, before + 1,
                    "seed \(seed) xDType \(xDType): fused path did not dispatch — test proves nothing")
                assertMatchesStock(
                    fused, stock, "seed \(seed) xDType \(xDType): fused kernel diverges from stock")
            }
        }
    }

    /// Mirrors the served artefact's dtype pairing: 16-bit stored tensors (F16), quantized to
    /// fp16 scales/biases, evaluated with float32 activations (the generation loop's dtype).
    /// Exercises the runner's `outDType` mirror logic (stock promotes x vs. scales: fp16 scales
    /// + fp32 x → fp32 output) at toy scale, and asserts the fused path's output dtype matches
    /// stock's exactly, not just its values.
    func testProductionDtypePairing() throws {
        let glu = makeToyGLU(seed: 21, weightDType: .float16)
        let (x, indices) = toyInputs(xDType: .float32)

        let stock = glu(x, indices: indices)
        eval(stock)

        MoEFusedQMVConfig.enabled = true
        defer { MoEFusedQMVConfig.enabled = false }
        let before = MoEFusedQMVConfig.dispatchCount
        let fused = glu(x, indices: indices)
        eval(fused)

        XCTAssertEqual(
            MoEFusedQMVConfig.dispatchCount, before + 1,
            "fused path did not dispatch — eligibility fell back to stock, test proves nothing")
        XCTAssertEqual(
            fused.dtype, stock.dtype,
            "fused output dtype must match stock's promoted dtype (fp16 scales + fp32 x)")
        assertMatchesStock(
            fused, stock, "fused kernel diverges from stock at the production dtype pairing")
    }

    /// Decode-shaped edge case: a single token (T=1) still routed to `K` experts. The stock
    /// generation loop's post-prefill steps are exactly this shape.
    func testSingleTokenEdge() throws {
        let glu = makeToyGLU(seed: 13)
        MLXRandom.seed(23)
        let x = MLXRandom.normal([1, Self.H]).asType(.float16)
        let indices = MLXArray((0 ..< Int32(Self.K)).map { $0 }).reshaped([1, Self.K])
        eval(x, indices)

        let stock = glu(x, indices: indices)
        eval(stock)

        MoEFusedQMVConfig.enabled = true
        defer { MoEFusedQMVConfig.enabled = false }
        let before = MoEFusedQMVConfig.dispatchCount
        let fused = glu(x, indices: indices)
        eval(fused)

        XCTAssertEqual(
            MoEFusedQMVConfig.dispatchCount, before + 1,
            "fused path did not dispatch — eligibility fell back to stock, test proves nothing")
        XCTAssertEqual(fused.shape, [1, Self.K, Self.H])
        assertMatchesStock(fused, stock, "fused kernel diverges from stock at T=1")
    }

    /// `CoreFixtureTests`' toy config (`hidden=128`, `moeInter=64`, `numExperts=4`) fails
    /// `MoEGatherQMVRunner.isEligible` on both dims (well under the 512-value block size the
    /// kernel is specialized for) — quantizing at fixture scale and flipping the flag on must be
    /// a provable no-op: identical output to flag-off, and the dispatch counter proves the fused
    /// path never ran (a silent stock fallback, not an accidental match).
    func testFixtureScaleDimsFallBackInert() throws {
        let E = CoreFixtureTests.numExperts       // 4
        let H = CoreFixtureTests.hidden            // 128
        let I = CoreFixtureTests.moeInter          // 64
        let T = 6
        let K = CoreFixtureTests.expertsPerTok     // 2
        XCTAssertFalse(
            MoEGatherQMVRunner.isEligible(hiddenSize: H, intermediateSize: I, groupSize: 64, bits: 4),
            "precondition: fixture-scale dims are expected to be ineligible for the fused path")

        let glu = SwitchGLU(hiddenSize: H, intermediateSize: I, numExperts: E)
        MLXRandom.seed(29)
        let params: [String: MLXArray] = [
            "gate_proj.weight": MLXRandom.normal([E, I, H]),
            "up_proj.weight": MLXRandom.normal([E, I, H]),
            "down_proj.weight": MLXRandom.normal([E, H, I]),
        ]
        try! glu.update(parameters: ModuleParameters.unflattened(params), verify: .all)
        MLXNN.quantize(
            model: glu, groupSize: 64, bits: 4, mode: .affine,
            filter: { _, module in module is SwitchLinear })
        eval(glu)

        MLXRandom.seed(31)
        let x = MLXRandom.normal([T, H]).asType(.float16)
        let indices = MLXArray((0 ..< (T * K)).map { Int32($0 % E) }).reshaped([T, K])
        eval(x, indices)

        let flagOff = glu(x, indices: indices)
        eval(flagOff)

        MoEFusedQMVConfig.enabled = true
        defer { MoEFusedQMVConfig.enabled = false }
        let before = MoEFusedQMVConfig.dispatchCount
        let flagOn = glu(x, indices: indices)
        eval(flagOn)

        XCTAssertEqual(
            MoEFusedQMVConfig.dispatchCount, before,
            "fixture-scale dims (H=\(H), I=\(I)) must not dispatch the fused path")
        XCTAssertTrue(
            allClose(flagOff, flagOn).item(Bool.self),
            "flag-on output must be identical to flag-off at fixture scale (provable inertness)")
    }
}
