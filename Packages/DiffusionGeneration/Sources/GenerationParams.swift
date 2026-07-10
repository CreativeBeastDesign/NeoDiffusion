import Foundation

/// Decoding parameters for the block-diffusion denoising loop (phase-2 §1 / gotcha 10).
///
/// These mirror the *live* arguments of the reference `generate()` — the dead `steps` /
/// `minimal_topk` knobs are not represented (deviation 6). The served defaults are the model
/// card's **Q Mode** (`.q`), not `generate()`'s signature defaults (0.95/0.9); see `.q`/`.s`.
///
/// Temperature is kept for completeness but Phase 2 parity runs are temperature 0 only
/// (argmax + softmax confidence); non-zero temperature is not exercised by any M5 gate.
public struct GenerationParams: Sendable, Equatable {
    /// Confidence threshold τ_mask for accepting a *masked* position's sampled token (Γ / M2T).
    /// If fewer than `numToTransfer` masked positions clear it, the top-`numToTransfer` by
    /// confidence are taken instead (guarantees forward progress).
    public var threshold: Float
    /// Confidence threshold τ_edit for *editing* an already-drafted (unmasked, non-prompt)
    /// token (Δ / T2T). `0.0` in S Mode edits on any confidence + change; the reference keeps
    /// the strict `>` comparison, so τ_edit=0 still requires confidence strictly greater than 0.
    public var editingThreshold: Float
    /// Number of "global" refinement iterations permitted *after* the active block has no masks
    /// left. `post_steps` counts only mask-free iterations; the loop breaks once it exceeds this.
    public var maxPostSteps: Int
    /// Minimum masked positions to unmask per step (reference `num_to_transfer`, live = 1).
    public var numToTransfer: Int
    /// Stop generating further blocks once a fully-unmasked block contains `eosId`.
    public var eosEarlyStop: Bool
    /// Sampling temperature. `0` = greedy (argmax); the only value Phase 2 parity covers.
    public var temperature: Float
    /// Block granularity (model card: 32; toy fixtures: 16). Total length is padded to a multiple.
    public var blockLength: Int
    /// Maximum number of tokens to generate beyond the prompt.
    public var genLength: Int
    /// Placeholder id for not-yet-generated positions (`mask_id`; card 156895, toy remapped).
    public var maskId: Int
    /// End-of-sequence id (`eos_id` = `pad_token_id`; card 156892, toy remapped). Output is
    /// trimmed at the first `eosId` in the generated region (inclusive).
    public var eosId: Int

    /// Whether Elastic-Cache is enabled (WP-1a).
    public var elasticCacheEnabled: Bool
    /// Drift threshold γ for Elastic-Cache.
    public var elasticGamma: Float
    /// Sliding window size β for Elastic-Cache active prediction.
    public var elasticBeta: Int

    public init(
        threshold: Float,
        editingThreshold: Float,
        maxPostSteps: Int = 16,
        numToTransfer: Int = 1,
        eosEarlyStop: Bool = false,
        temperature: Float = 0.0,
        blockLength: Int = 32,
        genLength: Int = 2048,
        maskId: Int = 156895,
        eosId: Int = 156892,
        elasticCacheEnabled: Bool = false,
        elasticGamma: Float = 0.9,
        elasticBeta: Int = 16
    ) {
        self.threshold = threshold
        self.editingThreshold = editingThreshold
        self.maxPostSteps = maxPostSteps
        self.numToTransfer = numToTransfer
        self.eosEarlyStop = eosEarlyStop
        self.temperature = temperature
        self.blockLength = blockLength
        self.genLength = genLength
        self.maskId = maskId
        self.eosId = eosId
        self.elasticCacheEnabled = elasticCacheEnabled
        self.elasticGamma = elasticGamma
        self.elasticBeta = elasticBeta
    }

    /// The two served modes from the LLaDA2.1-mini model card (phase-2 §1, gotcha 10).
    public enum Mode: String, Sendable {
        /// **Q Mode** — quality: τ_mask 0.7, τ_edit 0.5. The default served mode.
        case q
        /// **S Mode** — speed: τ_mask 0.5, τ_edit 0.0 (editing effectively off past the mask pass).
        case s

        public var thresholds: (mask: Float, edit: Float) {
            switch self {
            case .q: return (0.7, 0.5)
            case .s: return (0.5, 0.0)
            }
        }
    }

    /// Build params for a served mode, filling the card thresholds and leaving the rest to
    /// caller-supplied values (block length, gen length, special ids come from the model/config).
    public static func mode(
        _ mode: Mode,
        blockLength: Int,
        genLength: Int,
        maskId: Int,
        eosId: Int,
        maxPostSteps: Int = 16,
        numToTransfer: Int = 1,
        eosEarlyStop: Bool = false,
        temperature: Float = 0.0,
        elasticCacheEnabled: Bool = false,
        elasticGamma: Float = 0.9,
        elasticBeta: Int = 16
    ) -> GenerationParams {
        let (mask, edit) = mode.thresholds
        return GenerationParams(
            threshold: mask,
            editingThreshold: edit,
            maxPostSteps: maxPostSteps,
            numToTransfer: numToTransfer,
            eosEarlyStop: eosEarlyStop,
            temperature: temperature,
            blockLength: blockLength,
            genLength: genLength,
            maskId: maskId,
            eosId: eosId,
            elasticCacheEnabled: elasticCacheEnabled,
            elasticGamma: elasticGamma,
            elasticBeta: elasticBeta)
    }
}
