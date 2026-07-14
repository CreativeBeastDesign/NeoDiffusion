import DiffusionCore
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

    /// Whether JOT (Just on Time) token-level early stopping is enabled.
    public var jotEnabled: Bool
    /// Number of steps a token's prediction must remain stable before it is frozen.
    public var jotK: Int
    /// Confidence threshold above which a token can be frozen.
    public var jotThreshold: Float
    /// WP-3a v2: use the **faithful** JOT mechanism — hold each frozen token's per-layer K/V at
    /// its pre-freeze (converged) value so neighbours attend to a constant representation, instead
    /// of the v1 behaviour that only zeroed the frozen MoE output and let the K/V drift (the
    /// perturbation-cascade misimplementation, `Plans/jot-logbook.md`). Cached path only, single
    /// active block, Elastic-Cache off; requires `speculationK == 1` (the K/V hold is not
    /// snapshot/rolled-back across a K>1 batch). No effect unless `jotEnabled`.
    public var jotFaithful: Bool
    /// WP-3a §11 "Option C": sub-block prefix commit. When a contiguous **frozen prefix** of the
    /// front active block reaches `subBlockMinPrefix` tokens, commit it early into the
    /// ExactPrefixCache and shrink the active window to the suffix — cutting attention-query *and*
    /// MoE work for the block's remaining steps, with no new sync point (commit boundaries are
    /// already the coarse readback). It is adaptive block sizing gated by the JOT convergence
    /// signal; the within-block-bidirectional staleness gamble (a prefix committed before the
    /// suffix settles) is the quality question. Requires `jotEnabled`; cached, nBuf=1,
    /// `speculationK==1`, Elastic/speculation off. `false` = disabled.
    public var subBlockCommit: Bool
    /// Minimum contiguous frozen-prefix length that triggers a sub-block commit (Option C). Small
    /// values commit aggressively (more, smaller segments); larger values only slide when a big
    /// prefix has settled. Ignored unless `subBlockCommit`.
    public var subBlockMinPrefix: Int
    /// WP-3a §10: static fixed-capacity MoE gather (the sync-free FLOP-skip, "Option-B enabler").
    /// When `> 0` **and** faithful JOT is on, each MoE layer gathers the `⌈ratio·T⌉` most-active
    /// (non-frozen) tokens into a compile-time-sized buffer, runs the experts over just those, and
    /// scatters back — actually skipping expert GEMMs for frozen tokens with **no GPU→CPU sync**
    /// (the capacity is a Swift Int, not a device scalar). `0` disables (falls back to the Option-A
    /// full-compute-then-mask path). Overflow semantics: if the active count exceeds the capacity,
    /// surplus active tokens are dropped (FFN=0) — size the ratio ≥ the expected active fraction
    /// (~0.56 at the measured 44% freeze rate). Marginal at `T=blockLength=32` (launch-bound);
    /// meant to be swept at larger `blockLength` where expert arithmetic dominates.
    public var moeCapacityRatio: Float

    /// Whether FlashBlock attention caching is enabled.
    public var flashBlockEnabled: Bool
    /// Dirty token threshold for FlashBlock cache refresh.
    public var flashBlockTau: Int

    /// Whether Temporal Self-Consistency Voting is enabled.
    public var temporalVotingEnabled: Bool
    /// Decay parameter α for exponential step-weighting in voting.
    public var temporalVotingAlpha: Float
    /// Cutoff ratio (t_start) below which intermediate outputs are discarded.
    public var temporalVotingCutoff: Float

    /// Whether In-Place Chain-of-Thought (ICE) is enabled.
    public var iceEnabled: Bool
    /// Confidence threshold above which ICE early exit is triggered.
    public var iceTau: Float
    /// Number of reasoning steps in the ICE template.
    public var iceNt: Int
    /// Length of the thinking section in ICE.
    public var iceThinkingLength: Int

    /// Whether Credit Decoding is enabled.
    public var creditDecodingEnabled: Bool
    /// Boost scale α for Credit Decoding.
    public var creditAlpha: Float
    /// Decay discount factor β for Credit Decoding.
    public var creditBeta: Float
    /// Exponent γ for concave transform in Credit Decoding.
    public var creditGamma: Float

    /// Module ablation for in-situ attribution. **Diagnostic only — always `.none` in serving.**
    /// Defined in DiffusionCore (the modules it switches live there); see ``ModuleAblation``.
    public var moduleAblation: ModuleAblation = .none

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
        elasticStaticBoundary: Int? = nil,
        jotEnabled: Bool = false,
        jotK: Int = 2,
        jotThreshold: Float = 0.9,
        jotFaithful: Bool = false,
        moeCapacityRatio: Float = 0,
        subBlockCommit: Bool = false,
        subBlockMinPrefix: Int = 8,
        flashBlockEnabled: Bool = false,
        flashBlockTau: Int = 4,
        temporalVotingEnabled: Bool = false,
        temporalVotingAlpha: Float = 0.0,
        temporalVotingCutoff: Float = 0.9,
        iceEnabled: Bool = false,
        iceTau: Float = 0.9,
        iceNt: Int = 3,
        iceThinkingLength: Int = 96,
        creditDecodingEnabled: Bool = false,
        creditAlpha: Float = 0.5,
        creditBeta: Float = 0.9,
        creditGamma: Float = 0.5
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
        self.jotEnabled = jotEnabled
        self.jotK = jotK
        self.jotThreshold = jotThreshold
        self.jotFaithful = jotFaithful
        self.moeCapacityRatio = moeCapacityRatio
        self.subBlockCommit = subBlockCommit
        self.subBlockMinPrefix = subBlockMinPrefix
        self.flashBlockEnabled = flashBlockEnabled
        self.flashBlockTau = flashBlockTau
        self.temporalVotingEnabled = temporalVotingEnabled
        self.temporalVotingAlpha = temporalVotingAlpha
        self.temporalVotingCutoff = temporalVotingCutoff
        self.iceEnabled = iceEnabled
        self.iceTau = iceTau
        self.iceNt = iceNt
        self.iceThinkingLength = iceThinkingLength
        self.creditDecodingEnabled = creditDecodingEnabled
        self.creditAlpha = creditAlpha
        self.creditBeta = creditBeta
        self.creditGamma = creditGamma
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

    /// Build params for a served mode: fills the card thresholds and the core decoding knobs, and
    /// leaves every optional feature (MultiBD / speculation / dynamic-τ / elastic / JOT / ICE /
    /// credit / temporal voting) at its `init` default. Callers that want a feature set the
    /// corresponding field on the returned value — e.g. `var p = .mode(.q, …); p.creditAlpha = 1`.
    /// This keeps the factory small and means adding a WP touches only `init`, not this signature.
    public static func mode(
        _ mode: Mode,
        blockLength: Int,
        genLength: Int,
        maskId: Int,
        eosId: Int,
        maxPostSteps: Int = 16,
        numToTransfer: Int = 1,
        eosEarlyStop: Bool = false,
        temperature: Float = 0.0
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
            eosId: eosId)
    }
}

public enum ICEPhase: Sendable {
    case reasoning
    case answer
}
