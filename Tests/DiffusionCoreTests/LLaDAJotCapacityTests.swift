import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import DiffusionCore

/// WP-3a §10 — the static capacity-gather MoE path in ``LLaDA2SparseMoEBlock``.
///
/// The gather runs the experts over only the `capacity` most-active (non-frozen) tokens and
/// scatters back. Correctness property verified here: **when `capacity ≥ numActive` the gather is
/// numerically identical to the Option-A full-compute-then-mask path** (every active token is in
/// the buffer, frozen tokens are zeroed either way). When `capacity < numActive` the surplus
/// active tokens are dropped (overflow → FFN 0), which the last case pins.
final class LLaDAJotCapacityTests: XCTestCase {

    // Small toy shapes (values irrelevant; equivalence between code paths is the point).
    static let H = 8, I = 4, E = 8, K = 2, G = 2, KG = 1
    static let scaling: Float = 1.0

    private func makeBlock() throws -> LLaDA2SparseMoEBlock {
        let (H, I, E, K, G, KG) = (Self.H, Self.I, Self.E, Self.K, Self.G, Self.KG)
        let block = LLaDA2SparseMoEBlock(
            hiddenSize: H, moeIntermediateSize: I, numExperts: E, numSharedExperts: 1,
            numExpertsPerTok: K, nGroup: G, topkGroup: KG, routedScalingFactor: Self.scaling)
        MLXRandom.seed(7)
        let w: [String: MLXArray] = [
            "gate.weight": MLXRandom.normal([E, H]) * 0.5,
            "gate.expert_bias": MLXRandom.normal([E]) * 0.1,
            "experts.gate_proj.weight": MLXRandom.normal([E, I, H]) * 0.5,
            "experts.up_proj.weight": MLXRandom.normal([E, I, H]) * 0.5,
            "experts.down_proj.weight": MLXRandom.normal([E, H, I]) * 0.5,
            "shared_experts.gate_proj.weight": MLXRandom.normal([I, H]) * 0.5,
            "shared_experts.up_proj.weight": MLXRandom.normal([I, H]) * 0.5,
            "shared_experts.down_proj.weight": MLXRandom.normal([H, I]) * 0.5,
        ]
        try block.update(parameters: ModuleParameters.unflattened(w), verify: .all)
        eval(block)
        return block
    }

    /// Count rows (over the token axis) that are all-zero.
    private func zeroRows(_ y: MLXArray) -> Int {
        let flat = y.reshaped(-1, y.dim(y.ndim - 1))         // [T, H]
        let rowAbs = abs(flat).sum(axis: -1)                 // [T]
        let zero = (rowAbs .== MLXArray(Float(0))).asType(.int32).sum()
        return Int(zero.item(Int32.self))
    }

    func testCapacityGatherMatchesMaskWhenNoOverflow() throws {
        let block = try makeBlock()
        let T = 8
        let x = (MLXRandom.normal([1, T, Self.H]) * 0.5)
        // 5 active (0..4), 3 frozen (5..7).
        let frozenBools = [false, false, false, false, false, true, true, true]
        let frozen = MLXArray(frozenBools).reshaped(1, T)
        let numActive = 5

        let full = block(x)                                   // frozen = nil
        let maskA = block(x, frozen: frozen)                  // Option A (compute all, zero frozen)
        let capEq = block(x, frozen: frozen, capacity: numActive)      // C == active
        let capHi = block(x, frozen: frozen, capacity: numActive + 1)  // C  > active
        let capLo = block(x, frozen: frozen, capacity: numActive - 1)  // C  < active (overflow)
        eval(full, maskA, capEq, capHi, capLo)

        // Option A: frozen rows zeroed, active rows equal the full compute.
        XCTAssertEqual(zeroRows(maskA), 3, "Option A must zero exactly the 3 frozen rows")
        let activeSlice = 0 ..< 5
        XCTAssertTrue(
            allClose(maskA[0, activeSlice, 0...], full[0, activeSlice, 0...], atol: 1e-5).item(Bool.self),
            "Option A active rows must equal the full compute")

        // capacity ≥ numActive ⇒ bit-for-bit the Option-A result.
        XCTAssertTrue(allClose(capEq, maskA, atol: 1e-5).item(Bool.self),
            "capacity == numActive must match the mask path")
        XCTAssertTrue(allClose(capHi, maskA, atol: 1e-5).item(Bool.self),
            "capacity > numActive must match the mask path")

        // capacity < numActive ⇒ one active token dropped (its FFN row becomes zero).
        XCTAssertEqual(zeroRows(capLo), 4,
            "capacity < numActive must drop exactly one active row (3 frozen + 1 overflow)")
    }
}
