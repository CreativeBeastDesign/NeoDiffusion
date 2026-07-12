import Foundation
import MLX
import MLXNN

/// One transformer layer: pre-norm attention and feed-forward with residual adds in the
/// activation dtype. The feed-forward is either a dense ``LLaDA2MLP`` (layer 0) or a
/// ``LLaDA2SparseMoEBlock`` (layers ≥ `first_k_dense_replace`); assembly is the model
/// adapter's job — this layer just composes whatever it is given.
public final class LLaDA2DecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") public var inputLayernorm: LLaDA2RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayernorm: LLaDA2RMSNorm
    @ModuleInfo(key: "attention") public var attention: LLaDA2Attention
    @ModuleInfo(key: "mlp") public var mlp: Module

    public init(
        hiddenSize: Int,
        rmsNormEps: Float,
        attention: LLaDA2Attention,
        mlp: Module
    ) {
        precondition(mlp is LLaDA2MLP || mlp is LLaDA2SparseMoEBlock,
                     "mlp must be LLaDA2MLP or LLaDA2SparseMoEBlock")
        self._inputLayernorm = ModuleInfo(
            wrappedValue: LLaDA2RMSNorm(dimensions: hiddenSize, eps: rmsNormEps),
            key: "input_layernorm")
        self._postAttentionLayernorm = ModuleInfo(
            wrappedValue: LLaDA2RMSNorm(dimensions: hiddenSize, eps: rmsNormEps),
            key: "post_attention_layernorm")
        self._attention = ModuleInfo(wrappedValue: attention, key: "attention")
        self._mlp = ModuleInfo(wrappedValue: mlp, key: "mlp")
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray?, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        var hidden = x + attention(inputLayernorm(x), mask: mask, cos: cos, sin: sin)
        let ffnInput = postAttentionLayernorm(hidden)
        let ffnOutput: MLXArray
        switch mlp {
        case let dense as LLaDA2MLP: ffnOutput = dense(ffnInput)
        case let moe as LLaDA2SparseMoEBlock: ffnOutput = moe(ffnInput)
        default: fatalError("unsupported mlp module type \(type(of: mlp))")
        }
        hidden = hidden + ffnOutput
        return hidden
    }

    /// Cache-aware layer forward (phase-2 §5 M5): attention runs against `cache`'s committed K/V
    /// over the active window; the feed-forward is unchanged (per-position). Mirrors
    /// ``callAsFunction(_:mask:cos:sin:)`` exactly but for the cached attention path. `mask` is
    /// nil for a single active block; WP-1b passes the block-causal active-window mask.
    public func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, cache: LayerKVCache,
        mask: MLXArray? = nil, frozen: MLXArray? = nil
    ) -> MLXArray {
        var hidden = x + attention(inputLayernorm(x), cos: cos, sin: sin, cache: cache, mask: mask)
        let ffnInput = postAttentionLayernorm(hidden)
        let ffnOutput: MLXArray
        switch mlp {
        case let dense as LLaDA2MLP: ffnOutput = dense(ffnInput)
        case let moe as LLaDA2SparseMoEBlock: ffnOutput = moe(ffnInput, frozen: frozen)
        default: fatalError("unsupported mlp module type \(type(of: mlp))")
        }
        hidden = hidden + ffnOutput
        return hidden
    }

    /// Faithful-JOT cache-aware layer forward (WP-3a v2): attention holds frozen columns' K/V
    /// (`jot`) so finalized tokens contribute a constant representation to their neighbours; the
    /// MoE still skips frozen tokens (their FFN output is unused — Δ never edits a frozen position
    /// and the held K/V make the skipped hidden invisible downstream). Mirrors the M5 cached
    /// overload exactly but for the held-K/V attention path.
    public func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray, cache: LayerKVCache,
        jot: LayerJotCache, frozen: MLXArray, mask: MLXArray? = nil
    ) -> MLXArray {
        var hidden = x + attention(
            inputLayernorm(x), cos: cos, sin: sin, cache: cache, jot: jot, frozen: frozen, mask: mask)
        let ffnInput = postAttentionLayernorm(hidden)
        let ffnOutput: MLXArray
        switch mlp {
        case let dense as LLaDA2MLP: ffnOutput = dense(ffnInput)
        case let moe as LLaDA2SparseMoEBlock: ffnOutput = moe(ffnInput, frozen: frozen)
        default: fatalError("unsupported mlp module type \(type(of: mlp))")
        }
        hidden = hidden + ffnOutput
        return hidden
    }

    /// Cache-aware layer forward (WP-1a): attention runs against committed and active caches,
    /// selectively recomputing or reusing the active KV depending on drift.
    public func callAsFunction(
        _ x: MLXArray, cos: MLXArray, sin: MLXArray,
        cache: LayerKVCache, activeCache: LayerActiveCache,
        prefixLen: Int, recomputeActive: Bool
    ) -> MLXArray {
        var hidden = x + attention(
            inputLayernorm(x), cos: cos, sin: sin,
            cache: cache, activeCache: activeCache,
            prefixLen: prefixLen, recomputeActive: recomputeActive)
        let ffnInput = postAttentionLayernorm(hidden)
        let ffnOutput: MLXArray
        switch mlp {
        case let dense as LLaDA2MLP: ffnOutput = dense(ffnInput)
        case let moe as LLaDA2SparseMoEBlock: ffnOutput = moe(ffnInput)
        default: fatalError("unsupported mlp module type \(type(of: mlp))")
        }
        hidden = hidden + ffnOutput
        return hidden
    }
}
