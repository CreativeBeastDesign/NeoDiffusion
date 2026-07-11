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

    /// Number of Block-Buffer slots (WP-1b MultiBD, arXiv:2606.29215 Alg. 5). `1` = SingleBD
    /// (the Phase-2 parity semantics, byte-identical); `2` refines two blocks concurrently.
    public var nBuf: Int
    /// Activation threshold τ_add (**sourced**: Alg. 4/5 — "the latest active block has
    /// progress > τ_add"): the next block activates when the newest active block's decoded
    /// fraction over its *generated* positions strictly exceeds τ_add. Paper (LLaDA2.1-Mini,
    /// Table 4): 0.10 math, 0.90 code. Default 2.0 = never activate (parity configuration).
    public var tauAdd: Float
    /// Semi-completion threshold τ_semi (**sourced**: Alg. 5 lines 12–14 + §C.4): a block with
    /// zero above-threshold acceptances this step receives the forced top-1 acceptance only if
    /// its *preceding* block is semi-complete — progress > τ_semi — or already committed (the
    /// front block's predecessor is committed, so the front keeps today's unconditional
    /// fallback). Paper (LLaDA2.1-Mini): 0.90.
    public var tauSemi: Float

    /// Single-model speculative decoding policy (WP-2a). `.none` = the parity-gated default;
    /// `.s2d2` = block-size-1 AR self-verification (arXiv:2603.25702) — cached path only,
    /// nBuf == 1 only, output legitimately differs from vanilla decoding (hybrid trajectory).
    public enum SpeculationKind: String, Sendable, Equatable {
        case none
        case s2d2
    }
    /// Which speculation policy runs (WP-2a).
    public var speculation: SpeculationKind
    /// S2D2 min-span routing threshold τ_span: verify only when the first contiguous masked
    /// span has at least this many positions. v1 ships always-verify (1); reserved for the
    /// routing sweep arm.
    public var tauSpan: Int

    /// WP-2b-2 dynamic confidence threshold (arXiv:2601.17917): adaptation strength α in
    /// τ(t) = τ0·(1 − α(1 − r_mask)), where r_mask is the start-of-step masked fraction over
    /// the slot's *generated* positions. `0` disables (τ(t) ≡ τ0 — exact parity); the paper's
    /// optimum is α≈0.6 at τ0=0.9. Applies to τ_mask (Γ) only — τ_edit stays static
    /// (decision 2026-07-11, one-variable discipline; see wp2b logbook).
    public var dynamicTauAlpha: Float
    /// WP-2b-3 EOS early exit (arXiv:2601.17917): once a Γ/Δ-settled, non-prompt EOS exists
    /// in the front block, every still-masked window position after the first EOS is filled
    /// with EOS in-graph, so the block (and any trailing slot) settles immediately. Output is
    /// trim-invariant by construction (trim is inclusive of the first EOS). Requires
    /// `eosEarlyStop`; default `false` = exact parity.
    public var eosEarlyExit: Bool

    /// Whether Elastic-Cache is enabled (WP-1a).
    public var elasticCacheEnabled: Bool
    /// Drift threshold γ for Elastic-Cache.
    public var elasticGamma: Float
    /// Sliding window size β for Elastic-Cache active prediction.
    public var elasticBeta: Int
    /// Optional static layer boundary for Proposal A (Static Depth Pruning).
    public var elasticStaticBoundary: Int?

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
        nBuf: Int = 1,
        tauAdd: Float = 2.0,
        tauSemi: Float = 0.9,
        speculation: SpeculationKind = .none,
        tauSpan: Int = 1,
        dynamicTauAlpha: Float = 0.0,
        eosEarlyExit: Bool = false,
        elasticCacheEnabled: Bool = false,
        elasticGamma: Float = 0.9,
        elasticBeta: Int = 16,
        elasticStaticBoundary: Int? = nil
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
        self.nBuf = nBuf
        self.tauAdd = tauAdd
        self.tauSemi = tauSemi
        self.speculation = speculation
        self.tauSpan = tauSpan
        self.dynamicTauAlpha = dynamicTauAlpha
        self.eosEarlyExit = eosEarlyExit
        self.elasticCacheEnabled = elasticCacheEnabled
        self.elasticGamma = elasticGamma
        self.elasticBeta = elasticBeta
        self.elasticStaticBoundary = elasticStaticBoundary
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
        nBuf: Int = 1,
        tauAdd: Float = 2.0,
        tauSemi: Float = 0.9,
        speculation: SpeculationKind = .none,
        tauSpan: Int = 1,
        dynamicTauAlpha: Float = 0.0,
        eosEarlyExit: Bool = false,
        elasticCacheEnabled: Bool = false,
        elasticGamma: Float = 0.9,
        elasticBeta: Int = 16,
        elasticStaticBoundary: Int? = nil
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
            nBuf: nBuf,
            tauAdd: tauAdd,
            tauSemi: tauSemi,
            speculation: speculation,
            tauSpan: tauSpan,
            dynamicTauAlpha: dynamicTauAlpha,
            eosEarlyExit: eosEarlyExit,
            elasticCacheEnabled: elasticCacheEnabled,
            elasticGamma: elasticGamma,
            elasticBeta: elasticBeta,
            elasticStaticBoundary: elasticStaticBoundary)
    }
}
