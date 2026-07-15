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
    /// stack, then the output head with logits cast to **FP32** (§2.6). Returns `[1, A, vocab]`.
    ///
    /// - Parameters:
    ///   - activeIds: active-window token ids `[1, A]` (A = one block, or two for WP-1b MultiBD)
    ///   - positionIds: absolute positions `[1, A]` of the active window
    ///   - caches: one committed-K/V cache per layer (updated in place)
    ///   - mask: optional additive mask `[1, 1, A, committed + A]`; nil for a single active
    ///     block, the block-causal active-window mask for a two-block window (WP-1b)
    public func callAsFunction(
        _ activeIds: MLXArray, positionIds: MLXArray, caches: [LayerKVCache],
        mask: MLXArray? = nil, frozen: MLXArray? = nil, ablation: ModuleAblation = .none,
        reuseRouter: Bool = false
    ) -> MLXArray {
        let hidden = model(activeIds, positionIds: positionIds, caches: caches, mask: mask,
                           frozen: frozen, ablation: ablation, reuseRouter: reuseRouter)
        return applyLMHead(hidden, ablation: ablation)
    }

    /// LM head, with the `.lmHead` diagnostic ablation (see ``ModuleAblation``). Substitutes a
    /// broadcast constant of the same `[.., vocab]` shape — not materialised — so the delta from
    /// `.none` is the [hidden → vocab] projection's own cost.
    func applyLMHead(_ hidden: MLXArray, ablation: ModuleAblation) -> MLXArray {
        guard ablation == .lmHead else { return lmHead(hidden).asType(.float32) }
        // The substitute MUST depend on `hidden`.
        //
        // The first version returned a plain constant. That made the logits independent of the
        // transformer stack, so **MLX dead-code-eliminated the entire model** — every layer,
        // embedding included — and the arm reported lm_head at 86% of the forward (impossible;
        // gather_qmm alone is 42.9%). Caught by the §5.7 sanity gate, not by a test: a timing
        // check cannot tell "this module is free" from "this module was deleted", which is the
        // same trap `.moeExpertGEMMs` was explicitly built to avoid.
        //
        // Summing hidden's channels is O(H) per position — negligible against a
        // [hiddenSize × 157184] projection — and keeps the whole stack alive.
        var shape = hidden.shape
        shape[shape.count - 1] = lmHead.weight.dim(0)  // [vocabSize, hiddenSize]
        let probe = hidden.sum(axis: -1, keepDims: true).asType(.float32)  // [..., 1]
        return broadcast(probe, to: shape)
    }

    /// Faithful-JOT cache-aware forward (WP-3a v2) + FlashBlock (WP-3b): active-window hidden states through the cached
    /// stack with per-layer frozen-K/V holds, then the FP32 output head. Returns `[1, A, vocab]`.
    public func callAsFunction(
        _ activeIds: MLXArray, positionIds: MLXArray, caches: [LayerKVCache],
        jotCaches: [LayerJotCache], frozen: MLXArray, mask: MLXArray? = nil, capacity: Int? = nil,
        flashBlockEnabled: Bool = false, flashBlockTau: Int = 4, isFirstStepOfBlock: Bool = false,
        dirtyPerSeq: [Int] = [0], ablation: ModuleAblation = .none
    ) -> MLXArray {
        let hidden = model(
            activeIds, positionIds: positionIds, caches: caches,
            jotCaches: jotCaches, frozen: frozen, mask: mask, capacity: capacity,
            flashBlockEnabled: flashBlockEnabled, flashBlockTau: flashBlockTau,
            isFirstStepOfBlock: isFirstStepOfBlock, dirtyPerSeq: dirtyPerSeq,
            ablation: ablation)
        return applyLMHead(hidden, ablation: ablation)
    }

    /// Elastic-Cache aware model forward pass (WP-1a).
    public func callAsFunction(
        _ activeIds: MLXArray, positionIds: MLXArray,
        caches: [LayerKVCache], activeCache: ActiveBlockCache,
        prefixLen: Int, recomputeActiveFlags: [Bool]
    ) -> MLXArray {
        let hidden = model(activeIds, positionIds: positionIds, caches: caches,
                           activeCache: activeCache, prefixLen: prefixLen,
                           recomputeActiveFlags: recomputeActiveFlags)
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
        _ activeIds: MLXArray, positionIds: MLXArray, caches: [LayerKVCache],
        mask: MLXArray? = nil, frozen: MLXArray? = nil, ablation: ModuleAblation = .none,
        reuseRouter: Bool = false
    ) -> MLXArray {
        precondition(caches.count == layers.count, "one cache per layer required")
        var hidden = wordEmbeddings(activeIds)
        let (cos, sin) = rotaryEmbedding.cosSin(positionIds: positionIds)
        for (layer, cache) in zip(layers, caches) {
            hidden = layer(hidden, cos: cos, sin: sin, cache: cache, mask: mask, frozen: frozen,
                           ablation: ablation, reuseRouter: reuseRouter)
        }
        return norm(hidden)
    }

    /// Faithful-JOT cache-aware inner forward (WP-3a v2) + FlashBlock (WP-3b): as the M5 cached forward, but each layer
    /// holds frozen columns' K/V via its ``LayerJotCache`` so finalized tokens keep a constant
    /// representation. `jotCaches` and `caches` are both aligned with `layers` and updated in place.
    public func callAsFunction(
        _ activeIds: MLXArray, positionIds: MLXArray, caches: [LayerKVCache],
        jotCaches: [LayerJotCache], frozen: MLXArray, mask: MLXArray? = nil, capacity: Int? = nil,
        flashBlockEnabled: Bool = false, flashBlockTau: Int = 4, isFirstStepOfBlock: Bool = false,
        dirtyPerSeq: [Int] = [0], ablation: ModuleAblation = .none
    ) -> MLXArray {
        precondition(caches.count == layers.count, "one cache per layer required")
        precondition(jotCaches.count == layers.count, "one jot cache per layer required")
        var hidden = wordEmbeddings(activeIds)
        let (cos, sin) = rotaryEmbedding.cosSin(positionIds: positionIds)
        for i in 0 ..< layers.count {
            hidden = layers[i](
                hidden, cos: cos, sin: sin, cache: caches[i],
                jot: jotCaches[i], frozen: frozen, mask: mask, capacity: capacity,
                flashBlockEnabled: flashBlockEnabled, flashBlockTau: flashBlockTau,
                isFirstStepOfBlock: isFirstStepOfBlock, dirtyPerSeq: dirtyPerSeq,
                ablation: ablation)
        }
        return norm(hidden)
    }

    /// Elastic-Cache aware inner forward pass (WP-1a).
    public func callAsFunction(
        _ activeIds: MLXArray, positionIds: MLXArray,
        caches: [LayerKVCache], activeCache: ActiveBlockCache,
        prefixLen: Int, recomputeActiveFlags: [Bool]
    ) -> MLXArray {
        precondition(caches.count == layers.count, "one cache per layer required")
        precondition(activeCache.layers.count == layers.count, "one active cache per layer required")
        precondition(recomputeActiveFlags.count == layers.count, "recompute flags must match layer count")
        
        var hidden = wordEmbeddings(activeIds)
        let (cos, sin) = rotaryEmbedding.cosSin(positionIds: positionIds)
        for i in 0 ..< layers.count {
            hidden = layers[i](hidden, cos: cos, sin: sin,
                               cache: caches[i], activeCache: activeCache.layers[i],
                               prefixLen: prefixLen, recomputeActive: recomputeActiveFlags[i])
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
