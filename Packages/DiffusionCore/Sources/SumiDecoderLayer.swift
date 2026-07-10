import Foundation
import MLX
import MLXNN

/// One Sumi transformer layer (`SumiEncoderLayer` in the reference): pre-norm off-by-one
/// attention and a dense SwiGLU feed-forward, residual adds in the activation dtype.
/// Sumi is dense throughout — no MoE variant, no qk-norm (weight keys: `self_attn`, `mlp`,
/// `input_layernorm`, `post_attention_layernorm`).
public final class SumiDecoderLayer: Module {
    @ModuleInfo(key: "input_layernorm") public var inputLayernorm: LLaDA2RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") public var postAttentionLayernorm: LLaDA2RMSNorm
    @ModuleInfo(key: "self_attn") public var selfAttn: OffByOneAttention
    @ModuleInfo(key: "mlp") public var mlp: LLaDA2MLP

    public init(
        hiddenSize: Int,
        intermediateSize: Int,
        rmsNormEps: Float,
        attention: OffByOneAttention
    ) {
        self._inputLayernorm = ModuleInfo(
            wrappedValue: LLaDA2RMSNorm(dimensions: hiddenSize, eps: rmsNormEps),
            key: "input_layernorm")
        self._postAttentionLayernorm = ModuleInfo(
            wrappedValue: LLaDA2RMSNorm(dimensions: hiddenSize, eps: rmsNormEps),
            key: "post_attention_layernorm")
        self._selfAttn = ModuleInfo(wrappedValue: attention, key: "self_attn")
        self._mlp = ModuleInfo(
            wrappedValue: LLaDA2MLP(hiddenSize: hiddenSize, intermediateSize: intermediateSize),
            key: "mlp")
        super.init()
    }

    public func callAsFunction(
        _ x: MLXArray, mask: MLXArray? = nil, cos: MLXArray, sin: MLXArray
    ) -> MLXArray {
        let hidden = x + selfAttn(inputLayernorm(x), mask: mask, cos: cos, sin: sin)
        return hidden + mlp(postAttentionLayernorm(hidden))
    }
}
