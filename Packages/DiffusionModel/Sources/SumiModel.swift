import Foundation
import MLX
import MLXNN
import DiffusionCore

/// The outer Sumi model (`SumiForMaskGeneration` in the reference): inner transformer +
/// untied `lm_head`. Assembly only — compute blocks live in DiffusionCore.
public class SumiModel: Module {
    public let config: SumiConfig

    @ModuleInfo(key: "model") public var model: SumiInnerModel
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public init(config: SumiConfig) {
        self.config = config
        self._model = ModuleInfo(wrappedValue: SumiInnerModel(config: config), key: "model")
        self._lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false), key: "lm_head")
        super.init()
    }

    /// Full forward with `_compute_logits` semantics (generation_sumi.py): hidden states
    /// through the stack, output head, logits **truncated to `vocab_size`** (training may pad
    /// the head to a TP multiple) and **upcast to FP32** — the samplers' softmaxes and
    /// confidences all run on this fp32 form.
    ///
    /// - Parameters:
    ///   - inputIds: token ids `[B, L]`
    ///   - mask: optional additive attention mask `[B, 1, L, L]`; `nil` is the generation
    ///     path (fully bidirectional — the reference's all-ones 2D mask expands to an
    ///     all-zero additive mask, asserted identical in the fixture dumper)
    ///   - positionIds: absolute positions `[B, L]` (`arange` over the canvas)
    /// - Returns: FP32 logits `[B, L, vocabSize]`
    public func callAsFunction(
        _ inputIds: MLXArray, mask: MLXArray? = nil, positionIds: MLXArray
    ) -> MLXArray {
        let hidden = model(inputIds, mask: mask, positionIds: positionIds)
        return lmHead(hidden)[.ellipsis, ..<config.vocabSize].asType(.float32)
    }

    /// Convenience forward over a full canvas: absolute `arange` positions, no mask.
    public func logits(forTokens inputIds: MLXArray) -> MLXArray {
        let totalLength = inputIds.dim(inputIds.ndim - 1)
        let positionIds = MLXArray(0 ..< Int32(totalLength)).expandedDimensions(axis: 0)
        return self(inputIds, mask: nil, positionIds: positionIds)
    }
}

/// The inner Sumi transformer (`SumiModel` in the reference): embeddings, 36 uniform dense
/// layers, final norm, and the (weight-free) rotary table generator.
public class SumiInnerModel: Module {
    @ModuleInfo(key: "embed_tokens") public var embedTokens: Embedding
    @ModuleInfo(key: "norm") public var norm: LLaDA2RMSNorm
    @ModuleInfo(key: "layers") public var layers: [SumiDecoderLayer]

    /// Not a Module child on purpose: no weights, stays out of the parameter tree.
    public let rotaryEmbedding: PartialRotaryEmbedding

    public init(config: SumiConfig) {
        self._embedTokens = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "embed_tokens")
        self._norm = ModuleInfo(
            wrappedValue: LLaDA2RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")

        // Full rotary (no partial factor), θ from rope_parameters (500k for Sumi-7B).
        self.rotaryEmbedding = PartialRotaryEmbedding(
            headDim: config.headDim,
            partialRotaryFactor: 1.0,
            ropeTheta: config.ropeTheta)

        var layerList = [SumiDecoderLayer]()
        for _ in 0 ..< config.numHiddenLayers {
            let attention = OffByOneAttention(
                hiddenSize: config.hiddenSize,
                numHeads: config.numAttentionHeads,
                numKVHeads: config.numKeyValueHeads,
                headDim: config.headDim,
                qkvBias: config.qkvBias,
                oBias: config.attentionBias)
            layerList.append(
                SumiDecoderLayer(
                    hiddenSize: config.hiddenSize,
                    intermediateSize: config.intermediateSize,
                    rmsNormEps: config.rmsNormEps,
                    attention: attention))
        }
        self._layers = ModuleInfo(wrappedValue: layerList, key: "layers")

        super.init()
    }

    public func callAsFunction(
        _ inputIds: MLXArray, mask: MLXArray? = nil, positionIds: MLXArray
    ) -> MLXArray {
        var hidden = embedTokens(inputIds)
        let (cos, sin) = rotaryEmbedding.cosSin(positionIds: positionIds)
        for layer in layers {
            hidden = layer(hidden, mask: mask, cos: cos, sin: sin)
        }
        return norm(hidden)
    }
}
