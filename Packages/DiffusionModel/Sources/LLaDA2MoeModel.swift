import Foundation
import MLX
import MLXNN
import DiffusionCore

/// The outer LLaDA2 MoE model wrapper containing the inner model and the output head.
/// Assembly only — the compute blocks live in DiffusionCore (phase-1 §4: kernels/model-IO
/// are adapters, blocks are core).
public class LLaDA2MoeModel: Module {
    public let config: LLaDA2MoeConfig

    @ModuleInfo(key: "model") public var model: LLaDA2MoeInnerModel
    @ModuleInfo(key: "lm_head") public var lmHead: Linear

    public init(config: LLaDA2MoeConfig) {
        self.config = config
        self._model = ModuleInfo(wrappedValue: LLaDA2MoeInnerModel(config: config), key: "model")
        // Untied output head; logits must be cast to FP32 by the forward pass (§2.6).
        self._lmHead = ModuleInfo(
            wrappedValue: Linear(config.hiddenSize, config.vocabSize, bias: false), key: "lm_head")
        super.init()
    }

    /// Full forward: hidden states through the stack, then the output head with logits cast to
    /// **FP32** (§2.6 — confidence thresholds were tuned against fp32 softmax; don't cheap out).
    ///
    /// - Parameters:
    ///   - inputIds: token ids `[B, L]`
    ///   - mask: additive block-diffusion mask `[1, 1, L, L]` (see ``BlockDiffusionMask``)
    ///   - positionIds: absolute positions `[B, L]` over the padded total length
    /// - Returns: FP32 logits `[B, L, vocabSize]`
    public func callAsFunction(
        _ inputIds: MLXArray, mask: MLXArray?, positionIds: MLXArray
    ) -> MLXArray {
        let hidden = model(inputIds, mask: mask, positionIds: positionIds)
        return lmHead(hidden).asType(.float32)
    }

    /// Cache-aware forward (phase-2 §5 M5): active-window hidden states through the cached inner
    /// stack, then the output head with logits cast to **FP32** (§2.6). Returns `[1, B, vocab]`.
    ///
    /// - Parameters:
    ///   - activeIds: active-window token ids `[1, B]`
    ///   - positionIds: absolute positions `[1, B]` of the active window
    ///   - caches: one committed-K/V cache per layer (updated in place)
    public func callAsFunction(
        _ activeIds: MLXArray, positionIds: MLXArray, caches: [LayerKVCache]
    ) -> MLXArray {
        let hidden = model(activeIds, positionIds: positionIds, caches: caches)
        return lmHead(hidden).asType(.float32)
    }

    /// Number of decoder layers (for sizing a per-layer cache array).
    public var layerCount: Int { model.layers.count }

    /// Convenience forward that assembles the block-diffusion mask and absolute position ids
    /// for a full padded sequence, then returns FP32 logits. `inputIds` length must be a
    /// multiple of `blockLength`. `.strict` is NeoDiffusion's real semantics (§6); the total
    /// sequence is treated as a single window (the M4/M5 parity setup).
    ///
    /// - Parameter inputIds: token ids `[B, L]` with `L % blockLength == 0`
    public func logits(
        forTokens inputIds: MLXArray,
        blockLength: Int,
        maskSemantics: BlockDiffusionMask.Semantics = .strict
    ) -> MLXArray {
        let totalLength = inputIds.dim(inputIds.ndim - 1)
        let mask = BlockDiffusionMask.build(
            totalLength: totalLength, blockLength: blockLength,
            semantics: maskSemantics, dtype: .float32)
        let positionIds = MLXArray(0 ..< Int32(totalLength)).expandedDimensions(axis: 0)
        return self(inputIds, mask: mask, positionIds: positionIds)
    }
}

/// The inner LLaDA2 transformer: embeddings, decoder layers, final norm, and the
/// (weight-free) rotary table generator.
public class LLaDA2MoeInnerModel: Module {
    @ModuleInfo(key: "word_embeddings") public var wordEmbeddings: Embedding
    @ModuleInfo(key: "norm") public var norm: LLaDA2RMSNorm
    @ModuleInfo(key: "layers") public var layers: [LLaDA2DecoderLayer]

    /// Not a Module child on purpose: it has no weights and must stay out of the
    /// parameter tree (the checkpoint has no rotary tensors).
    public let rotaryEmbedding: PartialRotaryEmbedding

    public init(config: LLaDA2MoeConfig) {
        self._wordEmbeddings = ModuleInfo(
            wrappedValue: Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize),
            key: "word_embeddings")
        self._norm = ModuleInfo(
            wrappedValue: LLaDA2RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps),
            key: "norm")

        self.rotaryEmbedding = PartialRotaryEmbedding(
            headDim: config.headDim,
            partialRotaryFactor: config.partialRotaryFactor,
            ropeTheta: config.ropeTheta)

        var layerList = [LLaDA2DecoderLayer]()
        for index in 0 ..< config.numHiddenLayers {
            layerList.append(Self.makeLayer(index: index, config: config))
        }
        self._layers = ModuleInfo(wrappedValue: layerList, key: "layers")

        super.init()
    }

    /// Inner forward (phase-2 §2.1/§2.2/§2.3): embed tokens, share one RoPE table across all
    /// layers, run the decoder stack under the additive block-diffusion mask, then the final norm.
    ///
    /// - Parameters:
    ///   - inputIds: token ids `[B, L]`
    ///   - mask: additive attention mask `[1, 1, L, L]` (see ``BlockDiffusionMask``); `nil`
    ///     means full bidirectional attention
    ///   - positionIds: absolute positions `[B, L]` over the padded total length (§1)
    public func callAsFunction(
        _ inputIds: MLXArray, mask: MLXArray?, positionIds: MLXArray
    ) -> MLXArray {
        var hidden = wordEmbeddings(inputIds)
        let (cos, sin) = rotaryEmbedding.cosSin(positionIds: positionIds)
        for layer in layers {
            hidden = layer(hidden, mask: mask, cos: cos, sin: sin)
        }
        return norm(hidden)
    }

    /// Cache-aware inner forward (phase-2 §5 M5): embed the **active window only**, share one
    /// RoPE table over its absolute positions, run each decoder layer against its committed K/V
    /// cache (no mask), then the final norm. Returns active-window hidden states `[1, B, H]`.
    ///
    /// `caches` must have one entry per layer, aligned with `layers`; each is updated in place
    /// (its `pending*` set to this window's K/V) for a later commit.
    public func callAsFunction(
        _ activeIds: MLXArray, positionIds: MLXArray, caches: [LayerKVCache]
    ) -> MLXArray {
        precondition(caches.count == layers.count, "one cache per layer required")
        var hidden = wordEmbeddings(activeIds)
        let (cos, sin) = rotaryEmbedding.cosSin(positionIds: positionIds)
        for (layer, cache) in zip(layers, caches) {
            hidden = layer(hidden, cos: cos, sin: sin, cache: cache)
        }
        return norm(hidden)
    }

    static func makeLayer(index: Int, config: LLaDA2MoeConfig) -> LLaDA2DecoderLayer {
        let attention = LLaDA2Attention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numKVHeads: config.numKeyValueHeads,
            headDim: config.headDim,
            rmsNormEps: config.rmsNormEps,
            useQkNorm: config.useQkNorm,
            useQkvBias: config.useQkvBias,
            useDenseBias: config.useBias)

        let mlp: Module
        if index < config.firstKDenseReplace {
            mlp = LLaDA2MLP(
                hiddenSize: config.hiddenSize,
                intermediateSize: config.intermediateSize ?? 4 * config.hiddenSize)
        } else {
            mlp = LLaDA2SparseMoEBlock(
                hiddenSize: config.hiddenSize,
                moeIntermediateSize: config.moeIntermediateSize ?? config.hiddenSize / 4,
                numExperts: config.numExperts,
                numSharedExperts: config.numSharedExperts,
                numExpertsPerTok: config.numExpertsPerTok,
                nGroup: config.nGroup,
                topkGroup: config.topkGroup,
                routedScalingFactor: config.routedScalingFactor)
        }

        return LLaDA2DecoderLayer(
            hiddenSize: config.hiddenSize,
            rmsNormEps: config.rmsNormEps,
            attention: attention,
            mlp: mlp)
    }
}
