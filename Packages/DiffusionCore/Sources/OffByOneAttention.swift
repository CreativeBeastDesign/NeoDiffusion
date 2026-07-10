import Foundation
import MLX
import MLXNN

/// Self-attention matching the reference `SumiAttention` + `eager_attention_forward`
/// (`modeling_sumi.py`): split Q/K/V projections (no bias), **no** qk-norm, full RoPE
/// applied before attention, GQA via explicit `repeat_kv`, and the **off-by-one softmax**
/// in FP32 — the reason MLX's fused `scaledDotProductAttention` cannot be used here
/// (`_supports_sdpa = False` in the reference; the sink changes the normaliser).
///
/// Attention is fully bidirectional at every step (`is_causal = False` unconditionally);
/// `mask` exists only for padded-batch use. Sumi generation feeds an all-ones 2D mask,
/// which the reference expands to an all-zero additive mask — numerically identical to
/// `mask: nil` here.
public final class OffByOneAttention: Module {
    public let numHeads: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let scale: Float

    @ModuleInfo(key: "q_proj") public var qProj: Linear
    @ModuleInfo(key: "k_proj") public var kProj: Linear
    @ModuleInfo(key: "v_proj") public var vProj: Linear
    @ModuleInfo(key: "o_proj") public var oProj: Linear

    public init(
        hiddenSize: Int,
        numHeads: Int,
        numKVHeads: Int,
        headDim: Int,
        qkvBias: Bool = false,
        oBias: Bool = false
    ) {
        self.numHeads = numHeads
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.scale = pow(Float(headDim), -0.5)

        self._qProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, numHeads * headDim, bias: qkvBias), key: "q_proj")
        self._kProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, numKVHeads * headDim, bias: qkvBias), key: "k_proj")
        self._vProj = ModuleInfo(
            wrappedValue: Linear(hiddenSize, numKVHeads * headDim, bias: qkvBias), key: "v_proj")
        self._oProj = ModuleInfo(
            wrappedValue: Linear(numHeads * headDim, hiddenSize, bias: oBias), key: "o_proj")

        super.init()
    }

    /// - Parameters:
    ///   - x: hidden states `[B, L, hiddenSize]`
    ///   - mask: optional additive attention mask `[B, 1, L, L]` (padding only; nil in generation)
    ///   - cos/sin: FP32 rotary tables `[B, L, headDim]` (full rotary) from ``PartialRotaryEmbedding``
    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)

        var queries = qProj(x).reshaped(B, L, numHeads, headDim).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)
        let values = vProj(x).reshaped(B, L, numKVHeads, headDim).transposed(0, 2, 1, 3)

        queries = PartialRotaryEmbedding.apply(queries, cos: cos, sin: sin)
        keys = PartialRotaryEmbedding.apply(keys, cos: cos, sin: sin)

        let attended = Self.attend(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask)
        let output = attended
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, numHeads * headDim)
        return oProj(output)
    }

    /// Off-by-one attention. Hot path (`mask == nil`, the generation case): MLX's fused
    /// SDPA plus a sigmoid(row-LSE) rescale — algebraically exact, see ``attendFast``.
    /// Masked path falls back to the eager form mirroring `eager_attention_forward`.
    ///
    /// **Routing contract**: the fast path fires only on literal `nil`. Do NOT pass a
    /// zero-valued additive mask to mean "no mask" (the HF all-ones-2D expansion) — it is
    /// numerically identical but routes to the O(L²)-materialising eager path on every
    /// step. `SumiModel`/`SumiEngine` pass `nil` throughout generation (verified — this is
    /// what the fast path's measurements were taken on).
    static func attend(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?
    ) -> MLXArray {
        guard let mask else {
            return attendFast(queries: queries, keys: keys, values: values, scale: scale)
        }
        return attendEager(queries: queries, keys: keys, values: values, scale: scale, mask: mask)
    }

    /// Fused off-by-one attention in ONE kernel call: MLX's SDPA supports native attention
    /// sinks (added for gpt-oss-style models), and Sumi's `softmax_one` is exactly one
    /// **zero-logit sink per head** — `exp(0)` joins the softmax denominator inside the
    /// fused kernel. Semantics gated against the verbatim sink oracle in the parity tests.
    static func attendFast(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float
    ) -> MLXArray {
        let sinks = MLXArray.zeros([queries.dim(1)], dtype: queries.dtype)
        return MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil,
            sinks: sinks)
    }

    /// Two-pass fallback/oracle (sumi-plan.md §4 finding 2 follow-up), kept for A/B and in
    /// case the native-sinks path regresses upstream.
    ///
    /// Derivation (**sourced** by algebra, gated against the sink oracle in tests): with
    /// row max `m` and `Z = Σ exp(x_j − m)`,
    ///   `softmax_one(x)_i = exp(x_i − m) / (Z + exp(−m)) = softmax(x)_i · Z / (Z + exp(−m))`
    /// and `Z / (Z + exp(−m)) = sigmoid(m + log Z) = sigmoid(logsumexp(x))`. Attention output
    /// is linear in the row weights, so
    ///   `offByOne(Q,K,V) = SDPA(Q,K,V) ⊙ sigmoid(rowLSE(Q·Kᵀ·scale))`.
    static func attendFastLSE(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float
    ) -> MLXArray {
        let ctx = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: nil)
        let lse = rowLogSumExp(queries: queries, keys: keys, scale: scale)
        let gate = sigmoid(lse).expandedDimensions(axis: -1).asType(ctx.dtype)
        return ctx * gate
    }

    /// Row-wise `logsumexp(Q·Kᵀ·scale)` in FP32, chunked over the key axis (running
    /// max/sum, flash-style) so peak memory is one `[B, H, S, chunk]` tile. GQA is handled
    /// by broadcasting `[B, Hkv, nRep, S, D] × [B, Hkv, 1, T, D]ᵀ` — no repeated-K copy.
    static func rowLogSumExp(
        queries: MLXArray, keys: MLXArray, scale: Float, chunkSize: Int = 256
    ) -> MLXArray {
        let (B, Hq, S, D) = (queries.dim(0), queries.dim(1), queries.dim(2), queries.dim(3))
        let Hkv = keys.dim(1)
        let nRep = Hq / Hkv
        // Matmul in the input dtype (FP16 on real weights — MLX GEMM accumulates FP32
        // internally); all reductions in FP32. FP32 inputs (parity fixtures) are unchanged.
        let q = queries.reshaped(B, Hkv, nRep, S, D)
        let L = keys.dim(2)

        var runningMax: MLXArray? = nil
        var runningSum: MLXArray? = nil
        var start = 0
        while start < L {
            let end = min(start + chunkSize, L)
            let tile = keys[.ellipsis, start ..< end, 0...]
                .expandedDimensions(axis: 2)                           // [B, Hkv, 1, T, D]
            let scores = (matmul(q, tile.swappedAxes(-1, -2)) * scale)
                .asType(.float32)                                      // [B, Hkv, nRep, S, T]
            let tileMax = scores.max(axis: -1)
            if let m = runningMax, let z = runningSum {
                let newMax = maximum(m, tileMax)
                runningSum = z * exp(m - newMax)
                    + exp(scores - newMax.expandedDimensions(axis: -1)).sum(axis: -1)
                runningMax = newMax
            } else {
                runningMax = tileMax
                runningSum = exp(scores - tileMax.expandedDimensions(axis: -1)).sum(axis: -1)
            }
            start = end
        }
        return (runningMax! + log(runningSum!)).reshaped(B, Hq, S)
    }

    /// Eager off-by-one attention mirroring `eager_attention_forward`: `repeat_kv` GQA
    /// expansion, `matmul(q, kᵀ) * scale` (reference op order), additive mask, `softmax_one`
    /// in FP32, cast back, `weights @ v`. Masked path + validation oracle for ``attendFast``.
    static func attendEager(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?
    ) -> MLXArray {
        let nRep = queries.dim(1) / keys.dim(1)
        let k = nRep == 1 ? keys : LLaDA2Attention.repeatKV(keys, nRep: nRep)
        let v = nRep == 1 ? values : LLaDA2Attention.repeatKV(values, nRep: nRep)

        var scores = matmul(queries, k.swappedAxes(-1, -2)) * scale
        if let mask { scores = scores + mask.asType(scores.dtype) }
        let weights = softmaxOne(scores.asType(.float32)).asType(queries.dtype)
        return matmul(weights, v)
    }

    /// Oracle variant using the verbatim sink formulation; pins `attend`'s agreement with
    /// the reference `softmax_one` construction (see `testSoftmaxOneFormulations`).
    static func attendSinkOracle(
        queries: MLXArray, keys: MLXArray, values: MLXArray, scale: Float, mask: MLXArray?
    ) -> MLXArray {
        let nRep = queries.dim(1) / keys.dim(1)
        let k = nRep == 1 ? keys : LLaDA2Attention.repeatKV(keys, nRep: nRep)
        let v = nRep == 1 ? values : LLaDA2Attention.repeatKV(values, nRep: nRep)

        var scores = matmul(queries, k.swappedAxes(-1, -2)) * scale
        if let mask { scores = scores + mask.asType(scores.dtype) }
        let weights = softmaxOneSink(scores.asType(.float32)).asType(queries.dtype)
        return matmul(weights, v)
    }
}
