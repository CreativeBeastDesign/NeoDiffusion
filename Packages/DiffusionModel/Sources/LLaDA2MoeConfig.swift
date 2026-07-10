import Foundation

/// Configuration for the LLaDA2 MoE model, matching Hugging Face's configuration keys.
/// Missing keys fallback to the default table defined in configuration_llada2_moe.py.
public struct LLaDA2MoeConfig: Codable, Equatable {
    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int?
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let hiddenAct: String
    public let useQkvBias: Bool
    public let useQkNorm: Bool
    public let useBias: Bool
    public let rmsNormEps: Float
    public let normHead: Bool
    public let tieWordEmbeddings: Bool
    public let embeddingDropout: Float
    public let attentionDropout: Float
    public let outputDropout: Float
    public let initializerRange: Float
    public let maxPositionEmbeddings: Int
    public let ropeTheta: Float
    public let useCache: Bool
    public let useSlidingWindow: Bool
    public let slidingWindow: Int
    public let maxWindowLayers: Int
    public let padTokenId: Int
    public let numExperts: Int
    public let numSharedExperts: Int
    public let numExpertsPerTok: Int
    public let nGroup: Int
    public let topkGroup: Int
    public let routedScalingFactor: Float
    public let moeIntermediateSize: Int?
    public let firstKDenseReplace: Int
    public let headDim: Int
    public let outputRouterLogits: Bool
    public let partialRotaryFactor: Float

    public var hiddenDim: Int { hiddenSize }
    public var numLayers: Int { numHiddenLayers }

    public init(
        vocabSize: Int = 30592,
        hiddenSize: Int = 1024,
        intermediateSize: Int? = nil,
        numHiddenLayers: Int = 24,
        numAttentionHeads: Int = 16,
        numKeyValueHeads: Int = 0,
        hiddenAct: String = "silu",
        useQkvBias: Bool = false,
        useQkNorm: Bool = true,
        useBias: Bool = true,
        rmsNormEps: Float = 1e-05,
        normHead: Bool = false,
        tieWordEmbeddings: Bool = false,
        embeddingDropout: Float = 0.1,
        attentionDropout: Float = 0.1,
        outputDropout: Float = 0.1,
        initializerRange: Float = 0.02,
        maxPositionEmbeddings: Int = 16384,
        ropeTheta: Float = 10000.0,
        useCache: Bool = true,
        useSlidingWindow: Bool = false,
        slidingWindow: Int = 4096,
        maxWindowLayers: Int = 28,
        padTokenId: Int = 126081,
        numExperts: Int = 16,
        numSharedExperts: Int = 0,
        numExpertsPerTok: Int = 2,
        nGroup: Int = 8,
        topkGroup: Int = 4,
        routedScalingFactor: Float = 2.5,
        moeIntermediateSize: Int? = nil,
        firstKDenseReplace: Int = 0,
        headDim: Int? = nil,
        outputRouterLogits: Bool = false,
        partialRotaryFactor: Float = 0.5
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.hiddenAct = hiddenAct
        self.useQkvBias = useQkvBias
        self.useQkNorm = useQkNorm
        self.useBias = useBias
        self.rmsNormEps = rmsNormEps
        self.normHead = normHead
        self.tieWordEmbeddings = tieWordEmbeddings
        self.embeddingDropout = embeddingDropout
        self.attentionDropout = attentionDropout
        self.outputDropout = outputDropout
        self.initializerRange = initializerRange
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.ropeTheta = ropeTheta
        self.useCache = useCache
        self.useSlidingWindow = useSlidingWindow
        self.slidingWindow = slidingWindow
        self.maxWindowLayers = maxWindowLayers
        self.padTokenId = padTokenId
        self.numExperts = numExperts
        self.numSharedExperts = numSharedExperts
        self.numExpertsPerTok = numExpertsPerTok
        self.nGroup = nGroup
        self.topkGroup = topkGroup
        self.routedScalingFactor = routedScalingFactor
        self.moeIntermediateSize = moeIntermediateSize
        self.firstKDenseReplace = firstKDenseReplace
        self.headDim = headDim ?? (hiddenSize / numAttentionHeads)
        self.outputRouterLogits = outputRouterLogits
        self.partialRotaryFactor = partialRotaryFactor
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case hiddenAct = "hidden_act"
        case useQkvBias = "use_qkv_bias"
        case useQkNorm = "use_qk_norm"
        case useBias = "use_bias"
        case rmsNormEps = "rms_norm_eps"
        case normHead = "norm_head"
        case tieWordEmbeddings = "tie_word_embeddings"
        case embeddingDropout = "embedding_dropout"
        case attentionDropout = "attention_dropout"
        case outputDropout = "output_dropout"
        case initializerRange = "initializer_range"
        case maxPositionEmbeddings = "max_position_embeddings"
        case ropeTheta = "rope_theta"
        case useCache = "use_cache"
        case useSlidingWindow = "use_sliding_window"
        case slidingWindow = "sliding_window"
        case maxWindowLayers = "max_window_layers"
        case padTokenId = "pad_token_id"
        case numExperts = "num_experts"
        case numSharedExperts = "num_shared_experts"
        case numExpertsPerTok = "num_experts_per_tok"
        case nGroup = "n_group"
        case topkGroup = "topk_group"
        case routedScalingFactor = "routed_scaling_factor"
        case moeIntermediateSize = "moe_intermediate_size"
        case firstKDenseReplace = "first_k_dense_replace"
        case headDim = "head_dim"
        case outputRouterLogits = "output_router_logits"
        case partialRotaryFactor = "partial_rotary_factor"
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 30592
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 1024
        self.hiddenSize = hiddenSize
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize)
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 24
        let numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 16
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? 0
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.useQkvBias = try container.decodeIfPresent(Bool.self, forKey: .useQkvBias) ?? false
        self.useQkNorm = try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
        self.useBias = try container.decodeIfPresent(Bool.self, forKey: .useBias) ?? true
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-05
        self.normHead = try container.decodeIfPresent(Bool.self, forKey: .normHead) ?? false
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.embeddingDropout = try container.decodeIfPresent(Float.self, forKey: .embeddingDropout) ?? 0.1
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.1
        self.outputDropout = try container.decodeIfPresent(Float.self, forKey: .outputDropout) ?? 0.1
        self.initializerRange = try container.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 16384
        self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
        self.useCache = try container.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
        self.useSlidingWindow = try container.decodeIfPresent(Bool.self, forKey: .useSlidingWindow) ?? false
        self.slidingWindow = try container.decodeIfPresent(Int.self, forKey: .slidingWindow) ?? 4096
        self.maxWindowLayers = try container.decodeIfPresent(Int.self, forKey: .maxWindowLayers) ?? 28
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId) ?? 126081
        self.numExperts = try container.decodeIfPresent(Int.self, forKey: .numExperts) ?? 16
        self.numSharedExperts = try container.decodeIfPresent(Int.self, forKey: .numSharedExperts) ?? 0
        self.numExpertsPerTok = try container.decodeIfPresent(Int.self, forKey: .numExpertsPerTok) ?? 2
        self.nGroup = try container.decodeIfPresent(Int.self, forKey: .nGroup) ?? 8
        self.topkGroup = try container.decodeIfPresent(Int.self, forKey: .topkGroup) ?? 4
        self.routedScalingFactor = try container.decodeIfPresent(Float.self, forKey: .routedScalingFactor) ?? 2.5
        self.moeIntermediateSize = try container.decodeIfPresent(Int.self, forKey: .moeIntermediateSize)
        self.firstKDenseReplace = try container.decodeIfPresent(Int.self, forKey: .firstKDenseReplace) ?? 0
        self.headDim = try container.decodeIfPresent(Int.self, forKey: .headDim) ?? (hiddenSize / numAttentionHeads)
        self.outputRouterLogits = try container.decodeIfPresent(Bool.self, forKey: .outputRouterLogits) ?? false
        self.partialRotaryFactor = try container.decodeIfPresent(Float.self, forKey: .partialRotaryFactor) ?? 0.5
    }
}
