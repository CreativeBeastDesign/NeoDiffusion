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

    /// WP-2b-2 non-linear dynamic confidence threshold schedule (arXiv:2601.17917, enhanced).
    public enum DynamicTauSchedule: String, Sendable, Equatable, CaseIterable {
        /// Half-cosine decay: τ(u) = τ0·(1 − α·½(1 − cos(π·u))), with zero derivative at u=0
        /// and u=1, protecting early context and smoothing late arrival at the floor.
        case cosine
        /// Convex power decay: τ(u) = τ0·(1 − α·u^p), concentrating threshold drop into the final positions.
        case power
        /// Linear decay: τ(u) = τ0·(1 − α·u) (legacy Streaming-dLLM formulation).
        case linear
    }

    /// WP-2b-2 dynamic confidence threshold (arXiv:2601.17917): adaptation strength α in
    /// τ(t) = τ0·(1 − α·g(1 − r_mask)), where r_mask is the start-of-step masked fraction over
    /// the slot's *generated* positions. `0` disables (τ(t) ≡ τ0 — exact parity); the paper's
    /// optimum is α≈0.6 at τ0=0.9.
    public var dynamicTauAlpha: Float
    /// Schedule shape for dynamic confidence thresholding (WP-2b). Default: `.cosine`.
    public var dynamicTauSchedule: DynamicTauSchedule
    /// Exponent p for `.power` schedule (default: 2.0).
    public var dynamicTauPower: Float
    /// Floor clamp for dynamic confidence thresholding (WP-2b).
    /// Guarantees τ(t) >= dynamicTauFloor, eliminating the un-editable deadband where tokens
    /// unmasked with confidence < editingThreshold could never be corrected by Δ.
    /// When nil (default), the engine clamps at `editingThreshold` (0.50 in Q mode).
    /// To reproduce legacy un-clamped behavior, set explicitly to 0.0.
    public var dynamicTauFloor: Float?
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
    /// Sync-free temporal cadence for FlashBlock prefix attention refresh (`0` = use legacy `flashBlockTau`;
    /// `> 0` = refresh on Step 0, Step 1, and every `N` steps thereafter, reusing prefix attention in between).
    public var flashBlockRefreshInterval: Int = 0

    /// Whether Unified Dual Caching is enabled (WP-4e).
    public var dualCachingEnabled: Bool
    /// KV Cache Quantization mode for Tier-1 Prefix storage (.none, .fp8, .k8v4, .fp4).
    public var kvQuantization: KVCacheQuantization
    /// Whether to use the pre-allocated memory pool for prefix KV.
    public var kvPoolEnabled: Bool
    /// Initial capacity for the KV pool in tokens.
    public var kvPoolCapacity: Int
    /// CSA2 static layer classification enabled (Tier-2 within-block).
    public var csa2Enabled: Bool
    /// CSA2 static layer reuse boundary (layers < boundary reuse on refinement steps).
    public var csa2Boundary: Int?
    /// CSA2 layer stride (e.g. 2 for alternating).
    public var csa2Stride: Int
    /// CSA2 temporal step stride (`1` = reuse every step >= minStep; `2` = alternate refresh/reuse so active KV is at most 1 step old).
    public var csa2StepStride: Int = 1
    /// CSA2 minimum step in block before active KV reuse begins (`1` = legacy; `2` = capture Step 1 post-mask KV first).
    public var csa2MinStep: Int = 1
    /// Token stability locking (SureLock / FreeCache unified mode).
    public var sureLockEnabled: Bool
    /// Skip the redundant `captureAndCommit` full model forward pass at block settle when the break step `j`'s
    /// `pendingKeys`/`pendingValues` already hold the clean K/V for `committedActive` (saves 1 full forward per block with 100% bitwise parity).
    public var skipRedundantCommitForward: Bool = true

    /// Whether the clean-commit skip may actually fire for this configuration.
    ///
    /// The skip commits the settle step's pending K/V instead of recomputing it. That is exact only
    /// if the settle step ran a *plain, fresh* forward over exactly the committed tokens, i.e. the
    /// same computation `captureAndCommit` would run. This is an **allowlist**: every feature that
    /// makes the step's K/V approximate, stale, or computed over different tokens disables it.
    ///
    /// Excluded, with the failure each would cause `[Sourced: code review 2026-10-18]`:
    /// - CSA2 / dual caching: reuse steps write the *previous* step's active K/V into
    ///   `pendingKeys` (`LLaDA2Attention` CSA2 branch), so a settle on a reuse step commits K/V
    ///   computed over pre-update tokens (possibly `[MASK]`) into the exact prefix cache.
    /// - JOT / SureLock: frozen-column holds put old K/V into pending; JOT can also change
    ///   tokens on the settle step.
    /// - FlashBlock: reuse steps compute attention against cached outputs, not fresh K/V paths.
    /// - Elastic-Cache: approximate active-window recompute.
    /// - Router reuse (N>1) / MoE capacity: deeper-layer hidden states (hence K/V) differ from a
    ///   fresh forward.
    /// - ICE: settle checks only the thinking region, so the step's input may differ from the
    ///   committed tokens.
    /// - nBuf≠1, sub-block commit, speculation, module ablation: window/commit shapes differ.
    public var cleanCommitSkipEligible: Bool {
        skipRedundantCommitForward
            && nBuf == 1 && !subBlockCommit && speculation == .none
            && moduleAblation == .none
            && !csa2Enabled && !dualCachingEnabled
            && !jotEnabled && !sureLockEnabled
            && !flashBlockEnabled
            && !elasticCacheEnabled
            && routerReuseSteps <= 1 && moeCapacityRatio == 0
            && !iceEnabled
    }

    /// Caching presets from the 2026-10-17/18 Cache Optimisation Campaign.
    ///
    /// **Status (errata 2026-10-18)**: only `exact-fast` and `legacy` are output-invariant
    /// (token-identical, 716/716 on 4-bit M1; Studio BF16 parity still OPEN). Every other preset
    /// changes the decoding trajectory and has **no blind quality sheet** — standing rule (1): no
    /// preset ships while its quality gate is open. They are experimental and must be opted into
    /// (``qualityGateOpen``). Token match vs FP16 is a *parity* number, not a quality number.
    /// TPS numbers below are aggregate tokens/seconds from `Tools/analysis/cache_campaign_aggregate.py`
    /// (the original comments quoted mean-of-per-prompt ratios, which overstated gains).
    /// `fp8` here is affine **int8**, not floating-point FP8.
    public enum CachePreset: String, Codable, Sendable, CaseIterable {
        /// Uncompressed FP16 prefix KV + clean-commit skip. Output-invariant.
        /// Aggregate +9.0% TPS at K=4 / +9.7% at K=1 (≈ −9–10% forwards; ms/fwd unchanged).
        case exactFast = "exact-fast"
        /// int8 Keys / int4 Values (group 64) + clean-commit. −59.4% KV bytes. Compute speed ≈ exact-fast
        /// (ms/fwd equal); any TPS difference is trajectory-driven. Quality gate OPEN.
        case balancedK8V4 = "k8v4-clean"
        /// int8 Keys & Values (group 64) + clean-commit. −46.9% KV bytes. Quality gate OPEN.
        case balancedFP8 = "fp8-clean"
        /// KIVI-style int4 (Keys per-channel over the 32-token block, Values per-token) + clean-commit.
        /// −70.3% KV bytes. Incompatible with sub-block commit. Quality gate OPEN.
        case ultraCompactKIVI4 = "kivi4-clean"
        /// k8v4 + periodic CSA2 (boundary 4, stride 2, stepStride 2, minStep 2). **UNMEASURED as
        /// defined**: the campaign arm ran with a clean-commit gate bug that committed stale CSA2 K/V
        /// (fixed 2026-10-18; CSA2 now makes the skip ineligible), and before 2026-10-18 this preset
        /// also set `dualCachingEnabled`, which silently enabled FlashBlock. Quality gate OPEN.
        case dualCodeK8V4 = "dual-k8v4"
        /// Uncompressed FP16 prefix KV, separate capture forward on commit (pre-campaign behaviour).
        case legacy = "legacy"

        /// `true` for presets that change the decoding trajectory and have no scored blind sheet.
        public var qualityGateOpen: Bool {
            switch self {
            case .exactFast, .legacy: return false
            case .balancedK8V4, .balancedFP8, .ultraCompactKIVI4, .dualCodeK8V4: return true
            }
        }
    }

    /// Applies a validated caching preset to this parameter set.
    public mutating func applyCachePreset(_ preset: CachePreset) {
        switch preset {
        case .exactFast:
            skipRedundantCommitForward = true
            kvQuantization = .none
            kvPoolEnabled = false
            csa2Enabled = false
            dualCachingEnabled = false
        case .balancedK8V4:
            skipRedundantCommitForward = true
            kvQuantization = .k8v4
            kvPoolEnabled = false
            csa2Enabled = false
            dualCachingEnabled = false
        case .balancedFP8:
            skipRedundantCommitForward = true
            kvQuantization = .fp8
            kvPoolEnabled = false
            csa2Enabled = false
            dualCachingEnabled = false
        case .ultraCompactKIVI4:
            skipRedundantCommitForward = true
            kvQuantization = .kivi4
            kvPoolEnabled = false
            csa2Enabled = false
            dualCachingEnabled = false
        case .dualCodeK8V4:
            skipRedundantCommitForward = true
            kvQuantization = .k8v4
            kvPoolEnabled = false
            csa2Enabled = true
            csa2Boundary = 4
            csa2Stride = 2
            csa2StepStride = 2
            csa2MinStep = 2
            // Was `true` until 2026-10-18: `dualCachingEnabled` also switches FlashBlock on
            // (interval 2), which the measured `cache-dual-k8v4` arm never did.
            dualCachingEnabled = false
        case .legacy:
            skipRedundantCommitForward = false
            kvQuantization = .none
            kvPoolEnabled = false
            csa2Enabled = false
            dualCachingEnabled = false
        }
    }

    /// Resolved CSA2 configuration if CSA2 or Dual Caching is enabled.
    public var csa2Config: CSA2Config? {
        guard csa2Enabled || dualCachingEnabled else { return nil }
        let effStepStride = dualCachingEnabled && !csa2Enabled ? max(2, csa2StepStride) : csa2StepStride
        let effMinStep = dualCachingEnabled && !csa2Enabled ? max(2, csa2MinStep) : csa2MinStep
        if let boundary = csa2Boundary {
            return .boundary(boundary, stepStride: effStepStride, minStep: effMinStep)
        } else {
            return .alternating(stride: csa2Stride, stepStride: effStepStride, minStep: effMinStep)
        }
    }



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

    /// **Router reuse** (experimental, `0` = off). Recompute the MoE router only every N forwards
    /// and reuse the previous decision in between.
    ///
    /// Motivated by measurement, not theory: the router costs **13.6% of the forward** and is
    /// dispatch-bound (~76× its physical floor — `gather_qmm_handoff.md` §6.2), while consecutive
    /// denoising steps were measured to share **80% of their expert set** (§8.4). If routing barely
    /// changes step to step, recomputing it every step buys little. At N=2 the ceiling is ~6.8% of
    /// the forward; at N=4, ~10%.
    ///
    /// The trade is quality: ~20% of expert choices per step are genuinely new, and reuse serves
    /// those tokens the wrong experts. Needs a quality gate before it is anything but a probe.
    ///
    /// **Known approximation in this implementation**: recompute is keyed off the *global* forward
    /// counter, not a per-block one, so a block boundary can occasionally land on a reuse step and
    /// carry the previous block's routing for one forward. That makes the quality reading
    /// pessimistic (conservative), not optimistic. Shape changes still force a recompute.
    public var routerReuseSteps: Int = 0

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
        dynamicTauSchedule: DynamicTauSchedule = .cosine,
        dynamicTauPower: Float = 2.0,
        dynamicTauFloor: Float? = nil,
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
        flashBlockRefreshInterval: Int = 0,
        dualCachingEnabled: Bool = false,
        kvQuantization: KVCacheQuantization = .none,
        kvPoolEnabled: Bool = false,
        kvPoolCapacity: Int = 256,
        csa2Enabled: Bool = false,
        csa2Boundary: Int? = nil,
        csa2Stride: Int = 2,
        csa2StepStride: Int = 1,
        csa2MinStep: Int = 1,
        sureLockEnabled: Bool = false,
        skipRedundantCommitForward: Bool = true,
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
        self.dynamicTauSchedule = dynamicTauSchedule
        self.dynamicTauPower = dynamicTauPower
        self.dynamicTauFloor = dynamicTauFloor
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
        self.flashBlockRefreshInterval = flashBlockRefreshInterval
        self.dualCachingEnabled = dualCachingEnabled
        self.kvQuantization = kvQuantization
        self.kvPoolEnabled = kvPoolEnabled
        self.kvPoolCapacity = kvPoolCapacity
        self.csa2Enabled = csa2Enabled
        self.csa2Boundary = csa2Boundary
        self.csa2Stride = csa2Stride
        self.csa2StepStride = csa2StepStride
        self.csa2MinStep = csa2MinStep
        self.sureLockEnabled = sureLockEnabled
        self.skipRedundantCommitForward = skipRedundantCommitForward
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
