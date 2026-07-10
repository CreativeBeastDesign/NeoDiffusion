import Foundation

/// Configuration for the Sumi uniform-diffusion model (`tohoku-nlp/sumi-7b`), matching
/// Hugging Face's configuration keys. Missing keys fall back to the default table defined
/// in `configuration_sumi.py` (cached under `Tools/reference/sumi/`), mirroring HF's
/// absent-key semantics — e.g. `rms_norm_eps` defaults to 1e-6 but the shipped config.json
/// overrides it to 1e-5.
///
/// `head_dim` and `num_key_value_heads` reproduce the reference `__post_init__`:
/// absent `head_dim` resolves to `hidden_size / num_attention_heads`, absent
/// `num_key_value_heads` resolves to `num_attention_heads` (MHA).
public struct SumiConfig: Codable, Equatable {
    /// Quantization metadata written by the conversion script into the converted artefact's
    /// config.json (absent from the upstream HF config).
    public struct Quantization: Codable, Equatable {
        public let bits: Int
        public let groupSize: Int
        public let mode: String

        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case mode
        }
    }

    /// Nested `rope_parameters` object (transformers 5.x layout). The reference reads
    /// `rope_theta` and `rope_type` from this dict; Sumi ships `{500000.0, "default"}`.
    public struct RopeParameters: Codable, Equatable {
        public let ropeTheta: Float
        public let ropeType: String

        public init(ropeTheta: Float = 10000.0, ropeType: String = "default") {
            self.ropeTheta = ropeTheta
            self.ropeType = ropeType
        }

        enum CodingKeys: String, CodingKey {
            case ropeTheta = "rope_theta"
            case ropeType = "rope_type"
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 10000.0
            self.ropeType = try container.decodeIfPresent(String.self, forKey: .ropeType) ?? "default"
        }
    }

    public let vocabSize: Int
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let numHiddenLayers: Int
    public let numAttentionHeads: Int
    public let numKeyValueHeads: Int
    public let hiddenAct: String
    public let maxPositionEmbeddings: Int
    public let initializerRange: Float
    public let rmsNormEps: Float
    public let useCache: Bool
    public let padTokenId: Int?
    public let bosTokenId: Int
    public let eosTokenId: Int
    public let tieWordEmbeddings: Bool
    public let ropeParameters: RopeParameters
    public let attentionBias: Bool
    public let attentionDropout: Float
    public let mlpBias: Bool
    public let headDim: Int
    public let addQkvBias: Bool
    public let quantization: Quantization?

    public var hiddenDim: Int { hiddenSize }
    public var numLayers: Int { numHiddenLayers }
    public var ropeTheta: Float { ropeParameters.ropeTheta }
    /// Reference: `qkv_bias = config.attention_bias or config.add_qkv_bias`.
    public var qkvBias: Bool { attentionBias || addQkvBias }

    public init(
        vocabSize: Int = 32000,
        hiddenSize: Int = 4096,
        intermediateSize: Int = 11008,
        numHiddenLayers: Int = 32,
        numAttentionHeads: Int = 32,
        numKeyValueHeads: Int? = nil,
        hiddenAct: String = "silu",
        maxPositionEmbeddings: Int = 2048,
        initializerRange: Float = 0.02,
        rmsNormEps: Float = 1e-6,
        useCache: Bool = false,
        padTokenId: Int? = nil,
        bosTokenId: Int = 1,
        eosTokenId: Int = 2,
        tieWordEmbeddings: Bool = false,
        ropeParameters: RopeParameters = RopeParameters(),
        attentionBias: Bool = false,
        attentionDropout: Float = 0.0,
        mlpBias: Bool = false,
        headDim: Int? = nil,
        addQkvBias: Bool = false,
        quantization: Quantization? = nil
    ) {
        self.vocabSize = vocabSize
        self.hiddenSize = hiddenSize
        self.intermediateSize = intermediateSize
        self.numHiddenLayers = numHiddenLayers
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads = numKeyValueHeads ?? numAttentionHeads
        self.hiddenAct = hiddenAct
        self.maxPositionEmbeddings = maxPositionEmbeddings
        self.initializerRange = initializerRange
        self.rmsNormEps = rmsNormEps
        self.useCache = useCache
        self.padTokenId = padTokenId
        self.bosTokenId = bosTokenId
        self.eosTokenId = eosTokenId
        self.tieWordEmbeddings = tieWordEmbeddings
        self.ropeParameters = ropeParameters
        self.attentionBias = attentionBias
        self.attentionDropout = attentionDropout
        self.mlpBias = mlpBias
        self.headDim = headDim ?? (hiddenSize / numAttentionHeads)
        self.addQkvBias = addQkvBias
        self.quantization = quantization
    }

    enum CodingKeys: String, CodingKey {
        case vocabSize = "vocab_size"
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case numHiddenLayers = "num_hidden_layers"
        case numAttentionHeads = "num_attention_heads"
        case numKeyValueHeads = "num_key_value_heads"
        case hiddenAct = "hidden_act"
        case maxPositionEmbeddings = "max_position_embeddings"
        case initializerRange = "initializer_range"
        case rmsNormEps = "rms_norm_eps"
        case useCache = "use_cache"
        case padTokenId = "pad_token_id"
        case bosTokenId = "bos_token_id"
        case eosTokenId = "eos_token_id"
        case tieWordEmbeddings = "tie_word_embeddings"
        case ropeParameters = "rope_parameters"
        case attentionBias = "attention_bias"
        case attentionDropout = "attention_dropout"
        case mlpBias = "mlp_bias"
        case headDim = "head_dim"
        case addQkvBias = "add_qkv_bias"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        self.vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 32000
        let hiddenSize = try container.decodeIfPresent(Int.self, forKey: .hiddenSize) ?? 4096
        self.hiddenSize = hiddenSize
        self.intermediateSize = try container.decodeIfPresent(Int.self, forKey: .intermediateSize) ?? 11008
        self.numHiddenLayers = try container.decodeIfPresent(Int.self, forKey: .numHiddenLayers) ?? 32
        let numAttentionHeads = try container.decodeIfPresent(Int.self, forKey: .numAttentionHeads) ?? 32
        self.numAttentionHeads = numAttentionHeads
        self.numKeyValueHeads =
            try container.decodeIfPresent(Int.self, forKey: .numKeyValueHeads) ?? numAttentionHeads
        self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
        self.maxPositionEmbeddings =
            try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 2048
        self.initializerRange = try container.decodeIfPresent(Float.self, forKey: .initializerRange) ?? 0.02
        self.rmsNormEps = try container.decodeIfPresent(Float.self, forKey: .rmsNormEps) ?? 1e-6
        self.useCache = try container.decodeIfPresent(Bool.self, forKey: .useCache) ?? false
        self.padTokenId = try container.decodeIfPresent(Int.self, forKey: .padTokenId)
        self.bosTokenId = try container.decodeIfPresent(Int.self, forKey: .bosTokenId) ?? 1
        self.eosTokenId = try container.decodeIfPresent(Int.self, forKey: .eosTokenId) ?? 2
        self.tieWordEmbeddings =
            try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.ropeParameters =
            try container.decodeIfPresent(RopeParameters.self, forKey: .ropeParameters)
            ?? RopeParameters()
        self.attentionBias = try container.decodeIfPresent(Bool.self, forKey: .attentionBias) ?? false
        self.attentionDropout = try container.decodeIfPresent(Float.self, forKey: .attentionDropout) ?? 0.0
        self.mlpBias = try container.decodeIfPresent(Bool.self, forKey: .mlpBias) ?? false
        self.headDim =
            try container.decodeIfPresent(Int.self, forKey: .headDim)
            ?? (hiddenSize / numAttentionHeads)
        self.addQkvBias = try container.decodeIfPresent(Bool.self, forKey: .addQkvBias) ?? false
        self.quantization = try container.decodeIfPresent(Quantization.self, forKey: .quantization)
    }
}
