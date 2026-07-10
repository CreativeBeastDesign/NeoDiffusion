import Foundation
import MLX

/// Block-diffusion attention mask builder (analytic — phase-2 §5 deviation 3: built directly
/// from block indices instead of the reference's O(total²) materialized tril expansion).
///
/// **`.strict` is NeoDiffusion's real semantics** (André's call, 2026-07-07). This is a
/// deviation from the *published inference script*, not from the paper's specified algorithm:
/// the paper describes cross-block causal / within-block bidirectional attention, and `.strict`
/// implements exactly that. The stock reference `generate()` is believed to have a bug (see
/// `.referenceBias`) — we do not copy the bug.
public enum BlockDiffusionMask {

    /// Which numeric semantics the additive mask carries.
    ///
    /// - `strict`: 0 on allowed pairs, -inf on disallowed — true block causality
    ///   (within-block bidirectional, cross-block causal). **The shipped default**, and the
    ///   semantics the paper's algorithm and every NeoDiffusion design doc assume; module
    ///   fixtures use it, and ExactPrefixCache correctness depends on it.
    /// - `referenceBias`: +1 on allowed pairs, 0 on disallowed — a **diagnostic/compat mode
    ///   only**, reproducing a *suspected bug* in the published reference `generate()`
    ///   (verified 2026-07-07): it passes a 0/1-valued 4D mask, and transformers applies 4D
    ///   masks additively as-is, so nothing is masked out — future blocks leak into earlier
    ///   ones with a mere +1 bias on allowed pairs. Not used in real inference; retained so
    ///   we can reproduce stock-`generate()` numerics when investigating discrepancies. Do not
    ///   ship this.
    public enum Semantics: Sendable {
        case strict
        case referenceBias
    }

    /// Additive mask `[1, 1, totalLength, totalLength]`: position `i` may attend to `j`
    /// iff `block(j) <= block(i)` with `block(p) = p / blockLength`.
    public static func build(
        totalLength: Int,
        blockLength: Int,
        semantics: Semantics = .strict,
        dtype: DType = .float32
    ) -> MLXArray {
        precondition(totalLength % blockLength == 0, "totalLength must be a multiple of blockLength")
        // Integer floor division: MLX `/` promotes int operands to float, which would make
        // this a token-level causal mask instead of a block-causal one.
        let blockIndex = MLXArray(0 ..< Int32(totalLength)).floorDivide(Int32(blockLength))
        let allowed = blockIndex.expandedDimensions(axis: -1) .>= blockIndex.expandedDimensions(axis: 0)
        let mask: MLXArray
        switch semantics {
        case .strict:
            mask = which(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))
        case .referenceBias:
            mask = which(allowed, MLXArray(Float(1)), MLXArray(Float(0)))
        }
        return mask.asType(dtype).expandedDimensions(axes: [0, 1])
    }
}
