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

    /// Additive mask for a cached **multi-block active window** (WP-1b MultiBD): rows are the
    /// `activeLen` active-window queries, columns are `prefixLen` committed keys followed by
    /// the `activeLen` active keys. Committed columns are all-allowed (every committed key is
    /// in a ≤-front block — the ExactPrefixCache argument); the active×active part is strict
    /// block-causal, so an earlier active block never sees a later one while the later block
    /// attends the earlier (arXiv:2606.29215 §C.3 running-set semantics).
    ///
    /// Shape `[1, 1, activeLen, prefixLen + activeLen]`. For `activeLen == blockLength`
    /// (single active block) the result is all-zero — equivalent to the `mask: nil` fast path;
    /// callers should skip the mask entirely in that case.
    public static func activeWindowMask(
        prefixLen: Int,
        activeLen: Int,
        blockLength: Int,
        dtype: DType = .float32
    ) -> MLXArray {
        precondition(activeLen % blockLength == 0, "activeLen must be a multiple of blockLength")
        let activePart = build(
            totalLength: activeLen, blockLength: blockLength, semantics: .strict, dtype: dtype)
        guard prefixLen > 0 else { return activePart }
        let prefixPart = MLXArray.zeros([1, 1, activeLen, prefixLen], dtype: dtype)
        return concatenated([prefixPart, activePart], axis: -1)
    }

    /// S2D2 self-verification mask (WP-2a; arXiv:2603.25702 Eq. 3, the "2L trick" over a full
    /// block): the verifier window is `[draft copy (B) | mask copy (B)]` at duplicated absolute
    /// positions, attending the committed prefix freely. Structure `M_ver = [[A_B, 0], [A_<B, I_B]]`:
    ///
    /// - draft rows (first copy): **causal** over the draft copy — position i sees drafts ≤ i —
    ///   so draft K/V encode tokens under autoregressive context;
    /// - verifier rows (second copy, position i): drafts **strictly** < i plus its own masked
    ///   position (identity) — the block-size-1 AR view: "condition on drafts left of i, keep i
    ///   masked".
    ///
    /// Shape `[1, 1, 2B, prefixLen + 2B]`. The full-B form (the paper verifies only the span,
    /// we pay the fixed 2B width) keeps tensor shapes static across steps — required by the
    /// K-step lazy batching. Non-span verifier rows are don't-care outputs.
    public static func s2d2VerifierMask(
        prefixLen: Int,
        blockLength: Int,
        dtype: DType = .float32
    ) -> MLXArray {
        let B = Int32(blockLength)
        let rows = MLXArray(0 ..< 2 * B).reshaped(2 * blockLength, 1)
        let cols = MLXArray(0 ..< 2 * B).reshaped(1, 2 * blockLength)
        let rowIsDraft = rows .< B
        let colIsDraft = cols .< B
        let rowPos = rows % B
        let colPos = cols % B
        let draftCausal = rowIsDraft .&& colIsDraft .&& (colPos .<= rowPos)
        let verifierStrict = (.!rowIsDraft) .&& colIsDraft .&& (colPos .< rowPos)
        let verifierSelf = (.!rowIsDraft) .&& (.!colIsDraft) .&& (colPos .== rowPos)
        let allowed = draftCausal .|| verifierStrict .|| verifierSelf
        let pairPart = which(allowed, MLXArray(Float(0)), MLXArray(-Float.infinity))
            .asType(dtype).expandedDimensions(axes: [0, 1])
        guard prefixLen > 0 else { return pairPart }
        let prefixPart = MLXArray.zeros([1, 1, 2 * blockLength, prefixLen], dtype: dtype)
        return concatenated([prefixPart, pairPart], axis: -1)
    }
}
