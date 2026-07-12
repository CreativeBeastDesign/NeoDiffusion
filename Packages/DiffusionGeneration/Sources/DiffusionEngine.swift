import Foundation
import MLX
import MLXNN
import DiffusionCore
import DiffusionModel

/// The block-diffusion denoising loop (phase-2 §1 / M5). Reproduces the reference
/// `generate()` algorithm **token-for-token at temperature 0** under NeoDiffusion's `.strict`
/// block mask (the corrected baseline — never stock `generate()`'s 0/1 soft-bias mask; §6,
/// deviation 8).
///
/// Shape and state:
/// - Built **Block-Buffer-shaped from day one** (``BlockBuffer``, `nBuf == 1` — phase-1 §5).
///   At `nBuf == 1` the slot machine reduces exactly to the reference SingleBD schedule.
/// - Both selection sets come from **one forward per step**: Γ (M2T — masked positions whose
///   confidence clears τ_mask, with a top-`numToTransfer` fallback guaranteeing progress) and
///   Δ (T2T — unmasked, non-prompt positions whose confidence clears τ_edit *and* whose
///   prediction changed). Committed blocks are immutable (phase-1 §5).
/// - Selection, thresholding, argmax and the state update are **MLX array ops** — no Swift-side
///   loops over logits (phase-1 §6). Loop control uses K-step speculative execution: up to `K`
///   steps are built into one graph and a single stacked "break" flag is read back per batch
///   (≤1 blocking readback per K steps + 1 per block commit — the M5(c) sync budget). Overshoot
///   past the break point is discarded via per-step snapshots, so the result stays exact.
///
/// The forward is injected (``Forward``) so the *same* loop drives both the cache-disabled path
/// (M5(a) parity anchor) and the ExactPrefixCache path (M5(b)); see ``generate`` vs the cached
/// entry point.
public final class DiffusionEngine {
    public let model: LLaDA2MoeModel
    /// Speculative batch size for loop control (phase-1 §6, K≈2–4). `K == 1` degenerates to a
    /// blocking readback every step; larger K amortizes the sync at the cost of bounded wasted
    /// forwards on overshoot. Output is identical for any K ≥ 1 (proven by test).
    public let speculationK: Int
    /// When `true`, per-phase wall-clock in ``Output/Metrics`` is made *real* by forcing
    /// `eval` at phase boundaries (prefill end, each commit's capture forward) — the M6 bench
    /// setting (handoff §3). Adds ≤1 extra synchronization per block commit and per prefill
    /// block, so leave `false` (the default) when only tokens or the sync audit matter.
    public let instrument: Bool

    public final class BoundaryHolder {
        public var value: Int = 0
        public init() {}
    }

    /// One decoded step's per-position trace (WP-2a calibration / JOT pre-experiment; emitted
    /// only when `onTrace` is set — an offline instrumentation path, never the serving path).
    /// Arrays are over the front block's `B` positions at that logical step.
    public struct StepTrace {
        public let blockIndex: Int
        public let stepInBlock: Int
        /// Top-1 confidence per position (x0_p).
        public let confidence: [Float]
        /// Positions unmasked this step (Γ, including speculation acceptances).
        public let transferred: [Bool]
        /// Positions edited this step (Δ).
        public let edited: [Bool]
        /// Argmax token id per position at this step.
        public let argmaxToken: [Int]
        /// Masked positions at the START of this step.
        public let masked: [Bool]
    }

    /// Offline trace hook (WP-2a): called once per applied logical step with the front slot's
    /// per-position trace. Adds a per-step readback of small arrays — set it only for
    /// calibration/analysis runs (<50 prompts), never in serving.
    public var onTrace: ((StepTrace) -> Void)?

    public init(model: LLaDA2MoeModel, speculationK: Int = 4, instrument: Bool = false) {
        precondition(speculationK >= 1, "speculationK must be >= 1")
        self.model = model
        self.speculationK = speculationK
        self.instrument = instrument
    }

    /// Result of one `generate` call.
    public struct Output {
        /// Trimmed generated ids (reference tail: up to and including the first `eosId`, else the
        /// full `genLength`), matching `generate()`'s return slice.
        public let tokens: [Int]
        /// The full padded sequence `x` `[total_length]` after every block committed — the
        /// token-for-token parity target against the reference trace's `final_x`.
        public let finalSequence: [Int]
        /// Per-committed-block full-prefix snapshots `x[:(b+1)*B]` (parity vs `block_commits`).
        public let blockCommits: [[Int]]
        /// Number of denoising steps spent per generated block (diagnostic; not a gate).
        /// Caveat (handoff §4): on a budget-break exit this counts the break iteration the
        /// reference does not (reads +1 vs the reference trace). Tokens are unaffected.
        public let stepsPerBlock: [Int]
        /// Count of blocking readbacks performed (M5(c) sync audit).
        public let syncPoints: Int
        /// M6 bench diagnostics (phase-2 §4 M6). Step/forward counts are always exact;
        /// the per-phase wall-clocks are only *real* when the engine was built with
        /// `instrument: true` (otherwise lazy evaluation blurs phase attribution).
        public let metrics: Metrics
        /// The sequence of tokens at each applied logical step (optional, collected if enabled).
        public let trajectorySequences: [[Int]]?

        public init(
            tokens: [Int],
            finalSequence: [Int],
            blockCommits: [[Int]],
            stepsPerBlock: [Int],
            syncPoints: Int,
            metrics: Metrics,
            trajectorySequences: [[Int]]? = nil
        ) {
            self.tokens = tokens
            self.finalSequence = finalSequence
            self.blockCommits = blockCommits
            self.stepsPerBlock = stepsPerBlock
            self.syncPoints = syncPoints
            self.metrics = metrics
            self.trajectorySequences = trajectorySequences
        }
    }

    /// Per-run diagnostics for the M6 metric set.
    public struct Metrics {
        /// Wall-clock spent prefilling pure-prompt blocks into the cache (cached path only).
        public let prefillSeconds: Double
        /// Wall-clock spent inside per-block denoising loops.
        public let denoiseSeconds: Double
        /// Wall-clock spent committing blocks (capture forward + block readback + streaming).
        public let commitSeconds: Double
        /// Per generated block: the reference's `post_steps` counter at loop exit — the number
        /// of mask-free (refinement) iterations, including the iteration that broke the loop.
        public let postStepsPerBlock: [Int]
        /// Forwards actually *evaluated*, honestly counting speculative overshoot (each K-batch
        /// evaluates K forwards even when the break lands early) and, on the cached path, the
        /// per-block commit-cleanliness capture forwards + prompt prefill forwards. TPF against
        /// this counter is the honest figure; against `stepsPerBlock.sum()` it is the logical
        /// (reference-comparable) figure (handoff §3 item 4).
        public let forwardsEvaluated: Int
        /// Per block, per logical step: |Γ| — masked positions transferred that step (M8 E1
        /// trajectory diagnostic; distinguishes "quantization changes final text" from
        /// "quantization changes denoising dynamics").
        public let transfersPerStep: [[Int]]
        /// Per block, per logical step: |Δ| — unmasked non-prompt positions edited that step.
        public let editsPerStep: [[Int]]
        /// Per block, per logical step: mean confidence (`x0_p`) over the positions written
        /// that step (Γ ∪ Δ); 0 where nothing was written (budget-break step).
        public let meanTransferConfidencePerStep: [[Float]]
        /// Index of the committed block whose generated region first contained `eosId`
        /// (the block `eos_early_stop` fires on), or nil if eos never appeared.
        public let eosBlockIndex: Int?

        // MARK: WP-1b MultiBD diagnostics + effective-parameter echoes
        /// Total logical steps taken by the loop (the TPF denominator). At `nBuf == 1` this
        /// equals `stepsPerBlock.sum()`; with two concurrently active blocks a single step
        /// advances both, so `stepsPerBlock.sum() - logicalStepsTotal == dualActiveSteps`.
        public let logicalStepsTotal: Int
        /// Logical steps during which two blocks were active simultaneously.
        public let dualActiveSteps: Int
        /// Global step index (0-based, counted in logical steps) at which each activation
        /// event fired (a trailing block entered the buffer). Empty at `nBuf == 1`.
        public let activationSteps: [Int]
        /// Per generated block: steps this block spent as the *trailing* slot with masks
        /// remaining and zero Γ acceptances — i.e. steps where the τ_semi gate (or the
        /// threshold) starved it. The τ_semi diagnostic.
        public let trailingStarvedStepsPerBlock: [Int]
        /// `denoiseSeconds` split by phase width — the WP-1b step-latency-multiplier inputs:
        /// single-active per-step latency = single/(logicalStepsTotal − dualActiveSteps),
        /// dual = dual/dualActiveSteps. Host-scoped (wall-clock); real per batch because every
        /// speculative batch ends in a blocking readback.
        public let singleActiveDenoiseSeconds: Double
        public let dualActiveDenoiseSeconds: Double
        // MARK: WP-2a speculation diagnostics
        /// Per block, per logical step: tokens written by the speculation policy that step
        /// (accepted draft prefix + the correction token; 0 on non-speculated steps or when
        /// `speculation == .none`). The accepted-tokens histogram source.
        public let acceptedPerStep: [[Int]]
        /// Total active-window tokens processed across all evaluated forwards (width-aware
        /// accounting: a 2B-wide verifier forward counts 2B, a B-wide target forward counts B;
        /// includes speculative overshoot and capture/prefill forwards at their widths).
        /// Width-corrected TPF = tokens / (tokensProcessedInForwards / blockLength).
        public let tokensProcessedInForwards: Int

        /// Effective parameters as the engine actually ran them (provenance rule: benches must
        /// record these echoes, never the CLI inputs — elastic-cache logbook F7).
        public let effectiveNBuf: Int
        public let effectiveTauAdd: Float
        public let effectiveTauSemi: Float
        public let effectiveSpeculationK: Int
        public let effectiveSpeculation: String
        public let effectiveTauSpan: Int
        // MARK: WP-2b effective echoes
        public let effectiveDynamicTauAlpha: Float
        public let effectiveEosEarlyExit: Bool
        // MARK: JOT effective echoes
        public let effectiveJotEnabled: Bool
        public let effectiveJotK: Int
        public let effectiveJotThreshold: Float
        /// WP-3a v2: whether the faithful (per-layer frozen-K/V hold) mechanism ran, vs the v1
        /// MoE-zeroing behaviour. Only meaningful when `effectiveJotEnabled`.
        public let effectiveJotFaithful: Bool
        /// WP-3a §10: the static MoE capacity ratio actually applied (0 = Option-A mask path).
        public let effectiveMoeCapacityRatio: Float
        // MARK: ICE effective echoes
        public let effectiveIceEnabled: Bool
        public let effectiveIceTau: Float
        public let effectiveIceNt: Int
        public let effectiveIceThinkingLength: Int
        // MARK: Credit Decoding effective echoes
        public let effectiveCreditDecodingEnabled: Bool
        public let effectiveCreditAlpha: Float
        public let effectiveCreditBeta: Float
        public let effectiveCreditGamma: Float
    }

    /// Mutable stats shared between the cached entry point's closures and `run` (the capture
    /// forwards fire inside `run`'s commit section, after `run` has begun).
    final class RunStats {
        var prefillSeconds: Double = 0
        var extraForwards: Int = 0
    }

    /// Signature of an injected forward: given the current window ids `[1, W]` and the count of
    /// active-block positions `B`, return **FP32 logits over the last `B` positions** `[1, B, V]`.
    /// The cache-disabled path builds a full `.strict` mask over the window; a cached path can
    /// forward only the active block against committed KV (no mask needed).
    public typealias Forward = (_ windowIds: MLXArray, _ activeLen: Int, _ frozen: MLXArray?) -> MLXArray

    /// Signature of the S2D2 verifier forward (WP-2a): given the `[1, 2B]` pair window
    /// `[draft copy | mask copy]`, return FP32 logits `[1, 2B, V]`. The closure owns the
    /// duplicated absolute position ids and the memoized ``BlockDiffusionMask/s2d2VerifierMask``;
    /// cached path only (conditions on committed prefix KV).
    public typealias VerifierForward = (_ pairWindowIds: MLXArray) -> MLXArray

    // MARK: - Public entry point (cache-disabled)

    /// Cache-disabled generation — the M5(a) parity anchor: every step recomputes the full
    /// window under a fresh block mask, exactly like the reference recomputes its prefix
    /// (deviation 1 not yet applied). `streamBlock` receives each committed block's ids.
    ///
    /// `maskSemantics` defaults to `.strict` (NeoDiffusion's real semantics, §6);
    /// `.referenceBias` reproduces the stock-`generate()` 0/1 soft-bias mask and exists only
    /// for the M6 quality diagnostic (§6 item 4) — never serve it, and never combine it with
    /// the cached path (ExactPrefixCache's exactness argument holds only under `.strict`).
    public func generate(
        prompt: [Int],
        params: GenerationParams,
        maskSemantics: BlockDiffusionMask.Semantics = .strict,
        iceTemplate: [Int]? = nil,
        streamBlock: (([Int]) -> Void)? = nil
    ) -> Output {
        let forward: Forward = { [model] windowIds, activeLen, frozen in
            let W = windowIds.dim(windowIds.ndim - 1)
            let logits = model.logits(
                forTokens: windowIds, blockLength: params.blockLength,
                maskSemantics: maskSemantics)
            return logits[0..., (W - activeLen)..., 0...]
        }
        return run(prompt: prompt, params: params, iceTemplate: iceTemplate, forward: forward, streamBlock: streamBlock)
    }

    // MARK: - Public entry point (ExactPrefixCache enabled)

    /// Cache-enabled generation (phase-2 §5 M5(b), deviation 1). Each denoising step forwards
    /// **only the active block** against the committed-block K/V in ``ExactPrefixCache`` (no mask
    /// needed — every committed key is in an allowed ≤-current block), instead of recomputing the
    /// whole prefix. Proven identical to ``generate`` because a committed block's K/V equal what
    /// the full-window forward computes for those immutable, causal positions.
    ///
    /// Commit-cleanliness (gotcha 6): at each block commit a dedicated **capture forward** runs
    /// over the *final* committed tokens and its K/V is what gets appended — so speculative
    /// overshoot during denoising (which leaves stale `pending` K/V) never corrupts the cache.
    public func generateCached(
        prompt: [Int],
        params: GenerationParams,
        iceTemplate: [Int]? = nil,
        streamBlock: (([Int]) -> Void)? = nil
    ) -> Output {
        let B = params.blockLength
        let cache = ExactPrefixCache(layerCount: model.layerCount)
        let activeCache = ActiveBlockCache(layerCount: model.layerCount)
        // Faithful JOT (WP-3a v2): per-layer frozen-K/V hold. Only allocated/used when the
        // faithful mechanism is on; otherwise it stays inert (the v1 path ignores it).
        let jotCache = JotFreezeCache(layerCount: model.layerCount)
        let stats = RunStats()

        // Faithful JOT preconditions (see LayerJotCache / GenerationParams.jotFaithful):
        // the K/V hold mutates in-graph once per step and is not rolled back across a K>1
        // speculative batch, and it manages the active window's K/V (mutually exclusive with
        // Elastic-Cache and with a two-block window).
        if params.jotEnabled && params.jotFaithful {
            precondition(speculationK == 1,
                "faithful JOT requires speculationK == 1 — the frozen-K/V hold is not "
                + "snapshot/rolled-back across a K>1 speculative batch. K=1 measures the "
                + "K-invariant logical trajectory (the WP-3a algorithmic quantity).")
            precondition(!params.elasticCacheEnabled,
                "faithful JOT and Elastic-Cache both manage the active-window K/V — enable at most one.")
            precondition(params.nBuf == 1,
                "faithful JOT v1 supports a single active block (nBuf == 1).")
            precondition(params.speculation == .none,
                "faithful JOT is not composed with S2D2 speculation (WP-3a v1 scope).")
        }
        precondition(!params.subBlockCommit || (params.jotEnabled && params.jotFaithful),
            "sub-block prefix commit (Option C) requires faithful JOT (it keys off the frozen "
            + "mask and commits into the ExactPrefixCache); enable jotEnabled + jotFaithful.")

        // Capture forward over a full block of `ids` at absolute `startPos`, then commit its K/V.
        func captureAndCommit(_ ids: MLXArray, startPos: Int) {
            let positionIds = MLXArray(Int32(startPos) ..< Int32(startPos + ids.dim(1)))
                .expandedDimensions(axis: 0)
            _ = model(ids, positionIds: positionIds, caches: cache.layers)
            cache.commitBlock()
            stats.extraForwards += 1
            // Instrumented runs pin the capture forward's cost to the phase it belongs to
            // (prefill / commit) instead of letting it evaluate lazily inside the next
            // block's first denoising batch.
            if instrument {
                for layer in cache.layers {
                    if let k = layer.keys, let v = layer.values { eval(k, v) }
                }
            }
        }

        // Prefill: commit the pure-prompt blocks [0, prefillBlocks) into the cache so the first
        // generated block denoises against them. (Uniform with generated blocks — a prompt block
        // is just a pre-settled block; within-block bidirectional, cross-block causal via cache.)
        let prefillStart = Date()
        let prefillBlocks = prompt.count / B
        for b in 0 ..< prefillBlocks {
            let ids = MLXArray(prompt[b * B ..< (b + 1) * B].map { Int32($0) }).reshaped(1, B)
            captureAndCommit(ids, startPos: b * B)
        }
        stats.prefillSeconds = Date().timeIntervalSince(prefillStart)

        var stepIndex = 0

        let boundaryHolder = BoundaryHolder()

        // Memoized active-window masks for dual-active phases (WP-1b): constant across every
        // step of a phase — keyed by committed length since activeLen is 2B whenever used.
        var maskMemo: [Int: MLXArray] = [:]

        // Active-only forward: slice the active window out of the full window; positions are
        // absolute from the committed length (== window length − active length).
        let forward: Forward = { [self, model, cache, activeCache, jotCache, boundaryHolder] windowIds, activeLen, frozen in
            let W = windowIds.dim(windowIds.ndim - 1)
            let activeIds = windowIds[0..., (W - activeLen)...]
            let positionIds = MLXArray(Int32(W - activeLen) ..< Int32(W)).expandedDimensions(axis: 0)

            // Faithful JOT (WP-3a v2): hold frozen columns' per-layer K/V so finalized tokens
            // contribute a constant representation. Single active block only (guarded above).
            if params.jotEnabled && params.jotFaithful, let frozen {
                // Capacity-gather (WP-3a §10): ⌈ratio·activeLen⌉ static buffer for the MoE FLOP
                // skip (nil ⇒ Option-A full-compute-then-mask). activeLen is a Swift Int here.
                let capacity: Int? = params.moeCapacityRatio > 0
                    ? Int((params.moeCapacityRatio * Float(activeLen)).rounded(.up))
                    : nil
                return model(activeIds, positionIds: positionIds, caches: cache.layers,
                             jotCaches: jotCache.layers, frozen: frozen, mask: nil, capacity: capacity)
            }

            // Elastic off (the served default): plain cached forward. The elastic overload
            // materializes full attention weights per layer for the drift test — that
            // instrumentation must never run on the serving path.
            guard params.elasticCacheEnabled else {
                // Single active block: mask nil (every committed key is allowed — the
                // ExactPrefixCache argument). Two active blocks: block-causal active mask
                // (the trailing block sees the front, never vice versa — WP-1b).
                guard activeLen > B else {
                    return model(activeIds, positionIds: positionIds, caches: cache.layers, mask: nil, frozen: frozen)
                }
                let prefixLen = W - activeLen
                let mask = maskMemo[prefixLen] ?? {
                    let m = BlockDiffusionMask.activeWindowMask(
                        prefixLen: prefixLen, activeLen: activeLen, blockLength: B)
                    maskMemo[prefixLen] = m
                    return m
                }()
                return model(activeIds, positionIds: positionIds, caches: cache.layers, mask: mask, frozen: frozen)
            }

            let prefixLen = W - activeLen

            var recomputeFlags = Array(repeating: true, count: model.layerCount)
            if stepIndex > 0 {
                if let staticBoundary = params.elasticStaticBoundary {
                    for l in 0 ..< model.layerCount {
                        recomputeFlags[l] = (l >= staticBoundary)
                    }
                } else if self.speculationK == 1 {
                    // Option 1: Live readback at each step
                    let boundary = self.readBoundary(activeCache: activeCache, layerCount: model.layerCount, gamma: params.elasticGamma)
                    boundaryHolder.value = boundary
                    for l in 0 ..< model.layerCount {
                        recomputeFlags[l] = (l >= boundary)
                    }
                } else {
                    // Proposal B: Use the delayed boundary from the previous batch
                    let boundary = boundaryHolder.value
                    for l in 0 ..< model.layerCount {
                        recomputeFlags[l] = (l >= boundary)
                    }
                }
            }

            let logits = model(activeIds, positionIds: positionIds, caches: cache.layers,
                               activeCache: activeCache, prefixLen: prefixLen,
                               recomputeActiveFlags: recomputeFlags)

            if params.elasticStaticBoundary == nil {
                var similarities: [MLXArray] = []
                for layer in activeCache.layers {
                    if let sim = layer.lastDriftSimilarity {
                        similarities.append(sim)
                    }
                }
                eval(similarities)
            }

            stepIndex += 1
            return logits
        }

        // S2D2 verifier forward (WP-2a): one 2B-wide forward under the M_ver mask at duplicated
        // absolute positions, against the committed prefix KV. Mask memoized per committed
        // length (constant across a block's denoising; commits happen only between phases).
        var verifierMaskMemo: [Int: MLXArray] = [:]
        let verifierForward: VerifierForward? = params.speculation == .s2d2
            ? { [model, cache] pairIds in
                let blockLen = pairIds.dim(1) / 2
                let prefixLen = cache.committedLength
                let blockPositions = MLXArray(Int32(prefixLen) ..< Int32(prefixLen + blockLen))
                let positionIds = concatenated([blockPositions, blockPositions], axis: 0)
                    .expandedDimensions(axis: 0)
                let mask = verifierMaskMemo[prefixLen] ?? {
                    let m = BlockDiffusionMask.s2d2VerifierMask(
                        prefixLen: prefixLen, blockLength: blockLen)
                    verifierMaskMemo[prefixLen] = m
                    return m
                }()
                return model(pairIds, positionIds: positionIds, caches: cache.layers, mask: mask)
            }
            : nil

        // On settle, run the capture forward over the committed tokens and commit clean K/V.
        let onSettled: (Int, MLXArray) -> Void = { _, committedActive in
            // Commit at the current committed length, not blockIndex*B — a block that already
            // sub-committed a prefix (Option C) starts its suffix past the block-aligned offset.
            captureAndCommit(committedActive, startPos: cache.committedLength)
            activeCache.clear()
            // Faithful JOT: the held K/V belong to the settled block's active window; the next
            // block starts with no frozen columns, so drop them (also avoids carrying a stale
            // window across a block boundary).
            jotCache.clear()
            stepIndex = 0
            boundaryHolder.value = 0
        }

        // Option C (WP-3a §11): commit a settled frozen prefix mid-block. Captures the prefix's KV
        // into the ExactPrefixCache at its absolute start and drops the held JOT columns for it.
        let onPrefixSettled: (Int, MLXArray) -> Void = { startPos, prefixIds in
            captureAndCommit(prefixIds, startPos: startPos)
            jotCache.dropPrefix(prefixIds.dim(1))
        }

        return run(prompt: prompt, params: params, iceTemplate: iceTemplate, activeCache: activeCache, boundaryHolder: boundaryHolder,
                   forward: forward, verifierForward: verifierForward,
                   streamBlock: streamBlock, onBlockSettled: onSettled,
                   onPrefixSettled: onPrefixSettled, stats: stats)
    }

    private func readBoundary(activeCache: ActiveBlockCache, layerCount: Int, gamma: Float) -> Int {
        var simsList: [MLXArray] = []
        for l in 0 ..< layerCount {
            if let sim = activeCache.layers[l].lastDriftSimilarity {
                simsList.append(sim.reshaped([1]))
            } else {
                simsList.append(MLXArray(Float(-1.0)).reshaped([1]))
            }
        }
        let sims = concatenated(simsList, axis: 0)
        let stale = sims .< MLXArray(gamma)
        let indices = MLXArray(0 ..< Int32(layerCount))
        let staleIndices = which(stale, indices, MLXArray(Int32(layerCount)))
        let boundaryLayerTensor = staleIndices.min()
        return Int(boundaryLayerTensor.item(Int32.self))
    }

    // MARK: - Shared loop (WP-1b slot scheduler)

    /// In-flight state of one active block in the slot scheduler. `slots[0]` is always the
    /// front (lowest block index) — the only block that may settle and commit (streaming
    /// contract, phase-1 §7).
    struct SlotRun {
        let blockIndex: Int
        let bufferSlot: Int
        var active: MLXArray          // [1, B]
        let promptMask: MLXArray      // [1, B] Bool
        /// Swift-side count of prompt positions in this block (progress denominators are
        /// over *generated* positions only — τ_add/τ_semi semantics, WP-1b).
        let promptCount: Int
        var postSteps: MLXArray       // in-graph post-steps accumulator
        var stepsTaken: Int = 0
        var trajectory = BlockTrajectory()
        /// Steps this block spent as the trailing slot with masks left and zero Γ acceptances
        /// (τ_semi starvation diagnostic).
        var trailingStarvedSteps: Int = 0
        // JOT tracking states (in-graph)
        var jotStableCount: MLXArray   // [1, B] Int32
        var prevPredictions: MLXArray  // [1, B] Int32
        var frozenMask: MLXArray       // [1, B] Bool
        var credit: MLXArray? = nil    // [1, B, V] Credit Decoding scores
        // ICE phase tracking
        var isAnswerPhase: Bool = false
    }

    /// Whether this run's configuration speculates (WP-2a S2D2): policy on, single slot,
    /// and a verifier closure available (cached path).
    private func speculating(
        _ params: GenerationParams, slots: Int, verifierForward: VerifierForward?
    ) -> Bool {
        params.speculation == .s2d2 && slots == 1 && verifierForward != nil
    }

    /// The block schedule shared by the cache-disabled and cached paths. `forward` abstracts how
    /// logits over the active window are produced.
    ///
    /// WP-1b (arXiv:2606.29215 Alg. 5): a slot scheduler. Up to `params.nBuf` blocks refine
    /// concurrently; τ_add activates the next block off the newest block's progress; the front
    /// block commits strictly in order, then the trailing slot is promoted. At `nBuf == 1`
    /// every path below reduces exactly to the Phase-2 SingleBD schedule (parity-gated).
    func run(
        prompt: [Int],
        params: GenerationParams,
        iceTemplate: [Int]? = nil,
        activeCache: ActiveBlockCache? = nil,
        boundaryHolder: BoundaryHolder? = nil,
        forward: Forward,
        verifierForward: VerifierForward? = nil,
        streamBlock: (([Int]) -> Void)?,
        onBlockSettled: ((_ blockIndex: Int, _ committedActive: MLXArray) -> Void)? = nil,
        onPrefixSettled: ((_ startPos: Int, _ prefixIds: MLXArray) -> Void)? = nil,
        stats: RunStats? = nil
    ) -> Output {
        let B = params.blockLength
        precondition(!params.subBlockCommit || onPrefixSettled != nil,
            "subBlockCommit (Option C) requires the cached path's onPrefixSettled hook")
        precondition(params.nBuf >= 1 && params.nBuf <= 2, "WP-1b implements nBuf in 1...2")
        precondition(!(params.elasticCacheEnabled && params.nBuf > 1),
            "Elastic-Cache x MultiBD composability is untested (roadmap §5 cell open); "
            + "run one at a time")
        precondition(params.speculation == .none || verifierForward != nil,
            "S2D2 requires the cached path (the verifier conditions on committed prefix KV)")
        precondition(!(params.speculation != .none && params.nBuf > 1),
            "speculation x MultiBD is untested (roadmap §5 cell 1b×2a open); run one at a time")
        precondition(!(params.speculation != .none && params.elasticCacheEnabled),
            "speculation x Elastic-Cache: WP-1a is closed; do not combine")
        precondition(!(params.dynamicTauAlpha > 0 && params.speculation != .none),
            "dynamic τ x speculation is untested (roadmap §5 cell 2a×2b-2: threshold changes "
            + "acceptance dynamics); run one at a time")
        precondition(!params.eosEarlyExit || params.eosEarlyStop,
            "eosEarlyExit extends eosEarlyStop (WP-2b-3); enable both")
        let promptLength = prompt.count
        let numBlocks = (promptLength + params.genLength + B - 1) / B
        let prefillBlocks = promptLength / B
        let maskId = Int32(params.maskId)

        var buffer = BlockBuffer(nBuf: params.nBuf)
        var syncPoints = 0

        // Committed prefix: the pure-prompt blocks [0, prefillBlocks). Their ids are prompt
        // tokens by construction (prefillBlocks * B <= promptLength).
        var committedIds: [Int] = Array(prompt[0 ..< prefillBlocks * B])
        var prefixArray: MLXArray = committedIds.isEmpty
            ? MLXArray.zeros([1, 0], dtype: .int32)
            : MLXArray(committedIds.map { Int32($0) }).reshaped(1, prefillBlocks * B)

        var blockCommits: [[Int]] = []
        var stepsPerBlock: [Int] = []
        var postStepsPerBlock: [Int] = []
        var denoiseForwards = 0
        var denoiseSeconds = 0.0

        var trajectorySequences: [[Int]] = []
        let originalOnTrace = self.onTrace
        if params.temporalVotingEnabled {
            self.onTrace = { trace in
                let active = trace.argmaxToken
                trajectorySequences.append(committedIds + active)
                originalOnTrace?(trace)
            }
        }
        var commitSeconds = 0.0
        var transfersPerStep: [[Int]] = []
        var editsPerStep: [[Int]] = []
        var meanConfPerStep: [[Float]] = []
        var trailingStarvedPerBlock: [Int] = []
        var acceptedPerStep: [[Int]] = []
        var eosBlockIndex: Int? = nil
        var logicalStepsTotal = 0
        var dualActiveSteps = 0
        var activationSteps: [Int] = []
        var tokensProcessed = 0

        var slots: [SlotRun] = []
        var nextBlockIndex = prefillBlocks
        var stopped = false

        // Initial active block: prompt tail where it exists, mask id elsewhere. Only the
        // first generated block can hold prompt positions; the general build costs nothing.
        func activateSlot() {
            let numBlock = nextBlockIndex
            let blockStart = numBlock * B
            var initialActive = [Int32](repeating: maskId, count: B)
            var promptMaskLocal = [Bool](repeating: false, count: B)
            if let temp = iceTemplate, numBlock == prefillBlocks {
                precondition(temp.count == B, "iceTemplate count must equal blockLength B")
                for i in 0 ..< B {
                    initialActive[i] = Int32(temp[i])
                    // Real prompt positions are prompt positions — not "frozen generated" tokens.
                    // The scaffold template ids (e.g. "Step 1:") are the frozen ones (handled
                    // below); keeping the two distinct makes `promptCount` — and therefore every
                    // generated-position denominator (τ_add, τ_semi, dynamic τ) — correct if ICE
                    // is ever composed past the single-block config.
                    let global = blockStart + i
                    if global < promptLength {
                        promptMaskLocal[i] = true
                    }
                }
            } else {
                for i in 0 ..< B {
                    let global = blockStart + i
                    if global < promptLength {
                        initialActive[i] = Int32(prompt[global])
                        promptMaskLocal[i] = true
                    }
                }
            }
            let bufferSlot = buffer.activate(blockIndex: numBlock)
            let activeArray = MLXArray(initialActive).reshaped(1, B)
            let frozenMaskArray: MLXArray
            if iceTemplate != nil, numBlock == prefillBlocks {
                // Freeze the scaffold (non-mask, non-prompt template ids). Prompt positions are
                // protected via `promptMask`; both are excluded from Δ-editing, but only the
                // scaffold is a "frozen" slot for JOT / capacity-gather purposes.
                let isFrozen = zip(initialActive, promptMaskLocal).map { $0 != maskId && !$1 }
                frozenMaskArray = MLXArray(isFrozen).reshaped(1, B)
            } else {
                frozenMaskArray = MLXArray.zeros([1, B], dtype: .bool)
            }
            slots.append(SlotRun(
                blockIndex: numBlock,
                bufferSlot: bufferSlot,
                active: activeArray,
                promptMask: MLXArray(promptMaskLocal).reshaped(1, B),
                promptCount: promptMaskLocal.lazy.filter { $0 }.count,
                postSteps: MLXArray(Int32(0)),
                jotStableCount: MLXArray.zeros([1, B], dtype: .int32),
                prevPredictions: activeArray,
                frozenMask: frozenMaskArray,
                isAnswerPhase: false))
            nextBlockIndex += 1
        }

        activateSlot()

        var singleActiveDenoiseSeconds = 0.0
        var dualActiveDenoiseSeconds = 0.0

        while !slots.isEmpty {
            if logicalStepsTotal > 1000 {
                print("WARNING: Guard triggered! Exceeded 1000 logical steps. Breaking loop to prevent infinite run.")
                break
            }
            let hasNextBlock = nextBlockIndex < numBlocks
            let phaseWidth = slots.count
            let denoiseStart = Date()
            let phase = denoisePhase(
                prefix: prefixArray, slots: &slots, hasNextBlock: hasNextBlock,
                params: params, activeCache: activeCache, boundaryHolder: boundaryHolder,
                forward: forward, verifierForward: verifierForward,
                globalStep: &logicalStepsTotal, dualActiveSteps: &dualActiveSteps)
            syncPoints += phase.syncPoints
            denoiseForwards += phase.forwards
            tokensProcessed += phase.tokensProcessed
            let phaseSeconds = Date().timeIntervalSince(denoiseStart)
            denoiseSeconds += phaseSeconds
            if phaseWidth == 2 { dualActiveDenoiseSeconds += phaseSeconds }
            else { singleActiveDenoiseSeconds += phaseSeconds }

            switch phase.event {
            case .iceEarlyExit:
                break
            case .activation:
                activationSteps.append(logicalStepsTotal - 1)
                activateSlot()

            case .prefixCommit:
                // Option C (WP-3a §11): commit the settled frozen prefix [0, n) of the front block
                // early and shrink the active window to the suffix [n, L). The block STAYS the
                // front slot (same blockIndex, accumulating stepsTaken) — its per-block metrics
                // emit only when it fully settles (`.frontBreak`), so a sub-committed block still
                // counts as one block. Streaming emits the prefix now (in-order), the suffix later.
                let n = phase.prefixCommitN
                let front = slots[0]
                let full = front.active                          // [1, L]
                let L = full.dim(1)
                let prefixIds = full[0..., 0 ..< n]              // [1, n]
                let suffixIds = full[0..., n ..< L]              // [1, L-n]
                let commitStart = Date()

                // Commit the prefix KV into the ExactPrefixCache at its absolute start (== the
                // current committed length) and drop the held JOT columns for those positions.
                onPrefixSettled?(committedIds.count, prefixIds)

                let prefixIdInts = prefixIds.asArray(Int32.self).map { Int($0) }  // memcpy (batch synced)
                syncPoints += 1
                committedIds.append(contentsOf: prefixIdInts)
                prefixArray = concatenated([prefixArray, prefixIds], axis: 1)
                streamBlock?(prefixIdInts)
                commitSeconds += Date().timeIntervalSince(commitStart)

                // Reshape the front slot to the suffix, preserving block identity and progress.
                // A committed frozen prefix is never a prompt position (prompt tokens are never
                // frozen), so the block carries no leading prompt once n > 0 — the suffix prompt
                // count is whatever prompt remains after n (0 in every case a commit can fire).
                var reshaped = SlotRun(
                    blockIndex: front.blockIndex,
                    bufferSlot: front.bufferSlot,
                    active: suffixIds,
                    promptMask: front.promptMask[0..., n ..< L],
                    promptCount: max(0, front.promptCount - n),
                    postSteps: front.postSteps,
                    jotStableCount: front.jotStableCount[0..., n ..< L],
                    prevPredictions: front.prevPredictions[0..., n ..< L],
                    frozenMask: front.frozenMask[0..., n ..< L])
                reshaped.credit = front.credit?[0..., n ..< L, 0...]
                reshaped.stepsTaken = front.stepsTaken
                reshaped.trajectory = front.trajectory
                reshaped.trailingStarvedSteps = front.trailingStarvedSteps
                slots[0] = reshaped

                // eos_early_stop may fire off a prefix that already contains eos (the committed
                // generated region now includes it). Mirror the frontBreak check.
                if params.eosEarlyStop {
                    let generated = committedIds[promptLength...]
                    if generated.contains(params.eosId) {
                        for slot in slots { buffer.cancel(slotIndex: slot.bufferSlot) }
                        slots.removeAll()
                        stopped = true
                    }
                }

            case .frontBreak:
                // Commit the front block in order (streaming contract). The trailing slot,
                // if any, is promoted and keeps its in-flight tokens and post-steps counter.
                let front = slots.removeFirst()
                let committedActive = front.active
                let commitStart = Date()
                buffer.markSettled(slotIndex: front.bufferSlot)
                onBlockSettled?(front.blockIndex, committedActive)

                // One readback per commit: materialize the block's ids for streaming + prefix growth.
                let blockIds = committedActive.asArray(Int32.self).map { Int($0) }
                syncPoints += 1

                committedIds.append(contentsOf: blockIds)
                prefixArray = concatenated([prefixArray, committedActive], axis: 1)
                buffer.markCommitted(slotIndex: front.bufferSlot)

                blockCommits.append(committedIds)
                stepsPerBlock.append(front.stepsTaken)
                postStepsPerBlock.append(phase.frontPostAtBreak)
                transfersPerStep.append(front.trajectory.transfers)
                editsPerStep.append(front.trajectory.edits)
                meanConfPerStep.append(front.trajectory.meanConfidence)
                trailingStarvedPerBlock.append(front.trailingStarvedSteps)
                acceptedPerStep.append(front.trajectory.accepted)

                // First generated block whose generated-region positions contain eos (E1). Use the
                // committed-sequence offset of these ids (== blockIndex*B for a full block, but
                // blockIndex*B + prefix for a block that already sub-committed a prefix — Option C).
                let blockStart = committedIds.count - blockIds.count
                if eosBlockIndex == nil {
                    let generatedInBlock = blockIds.enumerated().filter {
                        blockStart + $0.offset >= promptLength
                    }
                    if generatedInBlock.contains(where: { $0.element == params.eosId }) {
                        eosBlockIndex = blockCommits.count - 1
                    }
                }
                streamBlock?(blockIds)
                commitSeconds += Date().timeIntervalSince(commitStart)

                // eos_early_stop: a committed block is always mask-free, so only test whether
                // the generated region so far contains eos. A live trailing slot is cancelled —
                // its tokens are discarded and its KV was never captured.
                if params.eosEarlyStop {
                    let generated = committedIds[promptLength...]
                    if generated.contains(params.eosId) {
                        for slot in slots { buffer.cancel(slotIndex: slot.bufferSlot) }
                        slots.removeAll()
                        stopped = true
                    }
                }
                if !stopped && slots.isEmpty && hasNextBlock {
                    activateSlot()
                }
            }
        }

        // Reference trim: first eos in the generated region [promptLength, promptLength+genLength),
        // inclusive; else the full genLength.
        let genEnd = min(promptLength + params.genLength, committedIds.count)
        let generated = Array(committedIds[promptLength ..< genEnd])
        let firstEos = generated.firstIndex(of: params.eosId) ?? params.genLength
        let tokens = Array(generated.prefix(firstEos + 1))

        if params.temporalVotingEnabled {
            self.onTrace = originalOnTrace
        }

        return Output(
            tokens: tokens,
            finalSequence: committedIds,
            blockCommits: blockCommits,
            stepsPerBlock: stepsPerBlock,
            syncPoints: syncPoints,
            metrics: Metrics(
                prefillSeconds: stats?.prefillSeconds ?? 0,
                denoiseSeconds: denoiseSeconds,
                commitSeconds: commitSeconds,
                postStepsPerBlock: postStepsPerBlock,
                forwardsEvaluated: denoiseForwards + (stats?.extraForwards ?? 0),
                transfersPerStep: transfersPerStep,
                editsPerStep: editsPerStep,
                meanTransferConfidencePerStep: meanConfPerStep,
                eosBlockIndex: eosBlockIndex,
                logicalStepsTotal: logicalStepsTotal,
                dualActiveSteps: dualActiveSteps,
                activationSteps: activationSteps,
                trailingStarvedStepsPerBlock: trailingStarvedPerBlock,
                singleActiveDenoiseSeconds: singleActiveDenoiseSeconds,
                dualActiveDenoiseSeconds: dualActiveDenoiseSeconds,
                acceptedPerStep: acceptedPerStep,
                tokensProcessedInForwards: tokensProcessed + (stats?.extraForwards ?? 0) * B,
                effectiveNBuf: params.nBuf,
                effectiveTauAdd: params.tauAdd,
                effectiveTauSemi: params.tauSemi,
                effectiveSpeculationK: speculationK,
                effectiveSpeculation: params.speculation.rawValue,
                effectiveTauSpan: params.tauSpan,
                effectiveDynamicTauAlpha: params.dynamicTauAlpha,
                effectiveEosEarlyExit: params.eosEarlyExit,
                effectiveJotEnabled: params.jotEnabled,
                effectiveJotK: params.jotK,
                effectiveJotThreshold: params.jotThreshold,
                effectiveJotFaithful: params.jotEnabled && params.jotFaithful,
                effectiveMoeCapacityRatio: (params.jotEnabled && params.jotFaithful)
                    ? params.moeCapacityRatio : 0,
                effectiveIceEnabled: params.iceEnabled,
                effectiveIceTau: params.iceTau,
                effectiveIceNt: params.iceNt,
                effectiveIceThinkingLength: params.iceThinkingLength,
                effectiveCreditDecodingEnabled: params.creditDecodingEnabled,
                effectiveCreditAlpha: params.creditAlpha,
                effectiveCreditBeta: params.creditBeta,
                effectiveCreditGamma: params.creditGamma),
            trajectorySequences: params.temporalVotingEnabled ? trajectorySequences : nil)
    }

    // MARK: - Per-phase denoising with K-step speculative readback

    /// Per-block trajectory diagnostics (M8 E1 + WP-2a), accumulated by the owning slot.
    struct BlockTrajectory {
        var transfers: [Int] = []
        var edits: [Int] = []
        var meanConfidence: [Float] = []
        /// Tokens written by the speculation policy per step (WP-2a; 0 when not speculating).
        var accepted: [Int] = []
    }

    /// Why a phase ended. A *phase* is a stretch of logical steps over a fixed set of active
    /// slots; any scheduling event ends the phase at a speculative-batch boundary (overshoot
    /// past the event is discarded, so results are exact and K-invariant).
    enum PhaseEvent {
        /// The front block settled or hit its post-steps budget — commit it.
        case frontBreak
        /// τ_add fired — activate the next block.
        case activation
        /// Option C (WP-3a §11): a contiguous frozen prefix of the front block reached the
        /// threshold — commit the prefix early and shrink the active window to the suffix.
        case prefixCommit
        /// ICE early exit transition from reasoning to answer phase.
        case iceEarlyExit
    }

    struct PhaseResult {
        let event: PhaseEvent
        let syncPoints: Int
        let forwards: Int
        /// Active-window tokens processed across the phase's evaluated forwards (width-aware:
        /// target forwards count A, verifier forwards 2B — WP-2a accounting).
        let tokensProcessed: Int
        /// The front slot's post-steps counter at the break step (valid on `.frontBreak`).
        let frontPostAtBreak: Int
        /// The frozen-prefix length to commit early (valid on `.prefixCommit`; 0 otherwise).
        var prefixCommitN: Int = 0
    }

    /// Run the current slot set until a scheduling event. Builds up to K speculative
    /// `windowStep`s into one graph; a single blocking readback of the stacked `[2K]`
    /// break/activation flags per batch (M5(c) budget: ≤1 readback per K steps + 1 per commit).
    func denoisePhase(
        prefix: MLXArray,
        slots: inout [SlotRun],
        hasNextBlock: Bool,
        params: GenerationParams,
        activeCache: ActiveBlockCache? = nil,
        boundaryHolder: BoundaryHolder? = nil,
        forward: Forward,
        verifierForward: VerifierForward? = nil,
        globalStep: inout Int,
        dualActiveSteps: inout Int
    ) -> PhaseResult {
        precondition(params.numToTransfer == 1,
            "Phase 2 parity implements numToTransfer == 1 (the live reference value); "
            + "num_to_transfer > 1 needs index-exact top-k tie handling (M6 bench extension)")
        let prefixLen = prefix.dim(1)
        let S = slots.count
        let B = params.blockLength
        let K = speculationK
        let specActive = speculating(params, slots: S, verifierForward: verifierForward)
        // Min-span routing (S2D2 §4.3, batch-boundary variant): a batch is verified iff the
        // previous batch ended with ≥ τ_span masks remaining (proxy for the span length — exact
        // while the masked region is one contiguous run, which holds until Γ punches holes).
        // Routing at batch boundaries is a REAL compute saving (the verifier forward is not
        // built at all), at the price of K-dependence for τ_span > 1 — same trade as the
        // elastic Proposal-B pattern; K-invariance is guaranteed (and tested) at τ_span == 1.
        var verifyThisBatch = specActive   // first batch of a phase: masks == B ≥ any τ_span
        var syncPoints = 0
        var forwardsEvaluated = 0
        var tokensProcessed = 0

        var windowActive = S == 1 ? slots[0].active
            : concatenated(slots.map(\.active), axis: 1)
        let promptMasks = S == 1 ? slots[0].promptMask
            : concatenated(slots.map(\.promptMask), axis: 1)
        let slotPromptCounts = slots.map(\.promptCount)
        var posts = slots.map(\.postSteps)

        var jotStableCount = S == 1 ? slots[0].jotStableCount
            : concatenated(slots.map(\.jotStableCount), axis: 1)
        var prevPredictions = S == 1 ? slots[0].prevPredictions
            : concatenated(slots.map(\.prevPredictions), axis: 1)
        var frozenMask = S == 1 ? slots[0].frozenMask
            : concatenated(slots.map(\.frozenMask), axis: 1)

        var credit: MLXArray? = nil
        if params.creditDecodingEnabled {
            if slots.allSatisfy({ $0.credit != nil }) {
                credit = S == 1 ? slots[0].credit : concatenated(slots.map { $0.credit! }, axis: 1)
            }
        }

        while true {
            // Build up to K speculative steps into one graph.
            var specWindow = windowActive
            var specPosts = posts
            var specJotStableCount = jotStableCount
            var specPrevPredictions = prevPredictions
            var specFrozenMask = frozenMask
            var specCredit = credit
            var snapshots: [MLXArray] = []     // result window if the break lands here
            var nextWindows: [MLXArray] = []   // window carried to the next step
            var breakFlags: [MLXArray] = []    // [1] Bool per step (front break)
            var activationFlags: [MLXArray] = [] // [1] Bool per step (τ_add)
            var postsPerStep: [[MLXArray]] = []
            var statsPerStep: [MLXArray] = []  // [5*S] per step
            var tracesPerStep: [MLXArray] = [] // [5, B] per step (offline tracing only)
            let tracing = onTrace != nil && S == 1

            var jotStableCountsPerStep: [MLXArray] = []
            var prevPredictionsPerStep: [MLXArray] = []
            var frozenMasksPerStep: [MLXArray] = []
            var frozenPrefixLensPerStep: [MLXArray] = []   // Option C candidate prefix per step
            var creditPerStep: [MLXArray?] = []

            var avgConfsPerStep: [MLXArray] = []
            let isAnswerPhase = slots[0].isAnswerPhase

            for _ in 0 ..< K {
                let s = windowStep(
                    prefix: prefix, prefixLen: prefixLen, windowActive: specWindow,
                    posts: specPosts, promptMasks: promptMasks,
                    slotPromptCounts: slotPromptCounts,
                    hasNextBlock: hasNextBlock, params: params, forward: forward,
                    verifierForward: verifyThisBatch ? verifierForward : nil,
                    tracing: tracing,
                    jotStableCount: specJotStableCount,
                    prevPredictions: specPrevPredictions,
                    frozenMask: specFrozenMask,
                    isAnswerPhase: isAnswerPhase,
                    credit: specCredit)
                snapshots.append(s.resultWindow)
                avgConfsPerStep.append(s.avgConfAnswer)
                nextWindows.append(s.nextWindow)
                breakFlags.append(s.breakFlag)
                activationFlags.append(s.activationFlag)
                postsPerStep.append(s.nextPosts)
                statsPerStep.append(s.stats)
                if let t = s.trace { tracesPerStep.append(t) }
                
                jotStableCountsPerStep.append(s.nextJotStableCount)
                prevPredictionsPerStep.append(s.nextPrevPredictions)
                frozenMasksPerStep.append(s.nextFrozenMask)
                frozenPrefixLensPerStep.append(s.frozenPrefixLen)
                creditPerStep.append(s.nextCredit)

                specWindow = s.nextWindow
                specPosts = s.nextPosts
                specJotStableCount = s.nextJotStableCount
                specPrevPredictions = s.nextPrevPredictions
                specFrozenMask = s.nextFrozenMask
                specCredit = s.nextCredit
            }

            // Single blocking readback of the 2K stacked event flags + ICE confidences.
            let flags = concatenated(breakFlags + activationFlags, axis: 0)  // [2K] Bool
            let confs = concatenated(avgConfsPerStep, axis: 0)              // [K] Float
            eval(flags)
            eval(confs)
            eval(snapshots + nextWindows + postsPerStep.flatMap { $0 } + statsPerStep
                 + tracesPerStep + jotStableCountsPerStep + prevPredictionsPerStep + frozenMasksPerStep
                 + (params.subBlockCommit ? frozenPrefixLensPerStep : [])
                 + creditPerStep.compactMap { $0 })

            // Proposal B (WP-1a elastic, nBuf == 1 only): refresh the delayed boundary from
            // the batch's last evaluated drift similarities.
            if let activeCache, let boundaryHolder,
               params.elasticCacheEnabled && params.elasticStaticBoundary == nil && K > 1 {
                boundaryHolder.value = readBoundary(
                    activeCache: activeCache, layerCount: model.layerCount,
                    gamma: params.elasticGamma)
            }

            let flagVals = flags.asArray(Bool.self)
            syncPoints += 1
            forwardsEvaluated += K * (verifyThisBatch ? 2 : 1)
            tokensProcessed += K * (S * B + (verifyThisBatch ? 2 * B : 0))

            let firstBreak = flagVals[0 ..< K].firstIndex(of: true)
            let firstActivation = flagVals[K ..< 2 * K].firstIndex(of: true).map { $0 - K }

            var firstEarlyExit: Int? = nil
            if params.iceEnabled && !slots[0].isAnswerPhase {
                let confVals = confs.asArray(Float.self)
                for idx in 0 ..< K {
                    let confVal = confVals[idx]
                    let statVals = statsPerStep[idx].asArray(Float.self)
                    let thinkingMasksLeft = statVals[3]
                    if confVal >= params.iceTau || thinkingMasksLeft == 0 {
                        firstEarlyExit = idx
                        break
                    }
                }
            }

            // Route the already-evaluated per-step stats of `count` logical steps to their slots.
            func applyStats(_ count: Int) {
                for r in 0 ..< count {
                    let vals = statsPerStep[r].asArray(Float.self)  // materialized — memcpy
                    for s in slots.indices {
                        slots[s].trajectory.transfers.append(Int(vals[5 * s]))
                        slots[s].trajectory.edits.append(Int(vals[5 * s + 1]))
                        slots[s].trajectory.meanConfidence.append(vals[5 * s + 2])
                        slots[s].trajectory.accepted.append(Int(vals[5 * s + 4]))
                        if s > 0 && vals[5 * s] == 0 && vals[5 * s + 3] > 0 {
                            slots[s].trailingStarvedSteps += 1
                        }
                    }
                    if tracing, r < tracesPerStep.count, let onTrace {
                        let t = tracesPerStep[r].asArray(Float.self)  // [5*B] memcpy
                        onTrace(StepTrace(
                            blockIndex: slots[0].blockIndex,
                            stepInBlock: slots[0].stepsTaken + r,
                            confidence: Array(t[0 ..< B]),
                            transferred: t[B ..< 2 * B].map { $0 > 0.5 },
                            edited: t[2 * B ..< 3 * B].map { $0 > 0.5 },
                            argmaxToken: t[3 * B ..< 4 * B].map { Int($0) },
                            masked: t[4 * B ..< 5 * B].map { $0 > 0.5 }))
                    }
                }
            }
            func applyWindow(_ window: MLXArray, posts stepPosts: [MLXArray], jotStableCount: MLXArray, prevPredictions: MLXArray, frozenMask: MLXArray, credit: MLXArray?) {
                for s in slots.indices {
                    slots[s].active = S == 1
                        ? window : window[0..., (s * B) ..< ((s + 1) * B)]
                    slots[s].postSteps = stepPosts[s]
                    slots[s].jotStableCount = S == 1
                        ? jotStableCount : jotStableCount[0..., (s * B) ..< ((s + 1) * B)]
                    slots[s].prevPredictions = S == 1
                        ? prevPredictions : prevPredictions[0..., (s * B) ..< ((s + 1) * B)]
                    slots[s].frozenMask = S == 1
                        ? frozenMask : frozenMask[0..., (s * B) ..< ((s + 1) * B)]
                    if let credit {
                        slots[s].credit = S == 1
                            ? credit : credit[0..., (s * B) ..< ((s + 1) * B), 0...]
                    }
                }
            }
            func addSteps(_ n: Int) {
                for s in slots.indices { slots[s].stepsTaken += n }
                globalStep += n
                if S == 2 { dualActiveSteps += n }
            }

            // Early exit transition
            if let j = firstEarlyExit, j <= (firstBreak ?? Int.max) && j <= (firstActivation ?? Int.max) {
                applyStats(j + 1)
                applyWindow(nextWindows[j], posts: postsPerStep[j], jotStableCount: jotStableCountsPerStep[j], prevPredictions: prevPredictionsPerStep[j], frozenMask: frozenMasksPerStep[j], credit: creditPerStep[j])
                addSteps(j + 1)
                slots[0].isAnswerPhase = true
                return PhaseResult(
                    event: .iceEarlyExit, syncPoints: syncPoints,
                    forwards: forwardsEvaluated, tokensProcessed: tokensProcessed,
                    frontPostAtBreak: 0)
            }

            // Break wins a same-step tie: the commit re-derives activation from fresh state.
            if let j = firstBreak, j <= (firstActivation ?? Int.max) {
                applyStats(j + 1)
                let frontPost = Int(postsPerStep[j][0].item(Int32.self))  // memcpy, evaluated above
                let isBudget = frontPost > params.maxPostSteps
                let finalCredit = isBudget ? (j > 0 ? creditPerStep[j - 1] : credit) : creditPerStep[j]
                // On a budget break the whole window's step-j write is discarded (simplest
                // K-invariant rule; the trailing slot's step-j write goes with it — recorded
                // deviation, WP-1b logbook). On a settle break `resultWindow == nextWindow`.
                applyWindow(snapshots[j], posts: postsPerStep[j], jotStableCount: jotStableCountsPerStep[j], prevPredictions: prevPredictionsPerStep[j], frozenMask: frozenMasksPerStep[j], credit: finalCredit)
                addSteps(j + 1)
                return PhaseResult(
                    event: .frontBreak, syncPoints: syncPoints,
                    forwards: forwardsEvaluated, tokensProcessed: tokensProcessed,
                    frontPostAtBreak: frontPost)
            }
            if let j = firstActivation {
                applyStats(j + 1)
                applyWindow(nextWindows[j], posts: postsPerStep[j], jotStableCount: jotStableCountsPerStep[j], prevPredictions: prevPredictionsPerStep[j], frozenMask: frozenMasksPerStep[j], credit: creditPerStep[j])
                addSteps(j + 1)
                return PhaseResult(
                    event: .activation, syncPoints: syncPoints,
                    forwards: forwardsEvaluated, tokensProcessed: tokensProcessed,
                    frontPostAtBreak: 0)
            }

            // Option C (WP-3a §11): sub-block prefix commit. Reached only when no break/activation
            // fired this batch. Commit the first step whose frozen prefix reaches the threshold and
            // is a *proper* prefix (n < front length; n == full length would have settled above).
            // The prefix-length reads are memcpies of the already-evaluated batch — no new sync.
            if params.subBlockCommit {
                let frontLen = windowActive.dim(1)   // S == 1 under Option C
                for j in 0 ..< K {
                    let n = Int(frozenPrefixLensPerStep[j].item(Int32.self))
                    if n >= params.subBlockMinPrefix && n < frontLen {
                        applyStats(j + 1)
                        applyWindow(nextWindows[j], posts: postsPerStep[j], jotStableCount: jotStableCountsPerStep[j], prevPredictions: prevPredictionsPerStep[j], frozenMask: frozenMasksPerStep[j], credit: creditPerStep[j])
                        addSteps(j + 1)
                        return PhaseResult(
                            event: .prefixCommit, syncPoints: syncPoints,
                            forwards: forwardsEvaluated, tokensProcessed: tokensProcessed,
                            frontPostAtBreak: 0, prefixCommitN: n)
                    }
                }
            }

            // No event within this batch — advance by K steps and continue.
            applyStats(K)
            applyWindow(specWindow, posts: specPosts, jotStableCount: specJotStableCount, prevPredictions: specPrevPredictions, frozenMask: specFrozenMask, credit: specCredit)
            addSteps(K)
            windowActive = specWindow
            posts = specPosts
            jotStableCount = specJotStableCount
            prevPredictions = specPrevPredictions
            frozenMask = specFrozenMask
            credit = specCredit

            // Min-span routing update from the batch's last evaluated step (memcpy, no sync):
            // masks remaining in the front block is the span-length proxy.
            if specActive && params.tauSpan > 1 {
                let vals = statsPerStep[K - 1].asArray(Float.self)
                verifyThisBatch = Int(vals[3]) >= params.tauSpan
            }
        }
    }

    /// Result of one speculative window step.
    private struct WindowStepResult {
        let nextWindow: MLXArray     // [1, S*B] window carried forward (post-update)
        let resultWindow: MLXArray   // window if the loop breaks at this step (budget → pre-update)
        let breakFlag: MLXArray      // [1] Bool — front settle or front budget
        let activationFlag: MLXArray // [1] Bool — τ_add fired (single-active phases only)
        let nextPosts: [MLXArray]    // per-slot post-steps accumulators carried forward
        /// Diagnostics `[5*S]` Float32 per slot: |Γ|, |Δ|, mean x0_p over written positions,
        /// masks remaining post-update, speculation-accepted count (WP-2a; slot 0 only).
        /// Read back only from the already-evaluated batch.
        let stats: MLXArray
        /// Per-position trace `[5, B]` for the front slot (x0_p, Γ, Δ, argmax token, start-of-
        /// step mask), built only when the engine's `onTrace` hook is set. Evaluated with the
        /// batch; read back in `applyStats`.
        let trace: MLXArray?
        let nextJotStableCount: MLXArray
        let nextPrevPredictions: MLXArray
        let nextFrozenMask: MLXArray
        /// Length of the front slot's leading contiguous frozen run after this step (Option C,
        /// WP-3a §11) as a `[1]` Int32 — the candidate sub-block-commit prefix. `0` when JOT is off.
        let frozenPrefixLen: MLXArray
        let avgConfAnswer: MLXArray  // [1] Float — ICE answer confidence
        let nextCredit: MLXArray?    // [1, A, V] Credit Decoding scores carried forward
    }

    /// One denoising step over the concatenated active window, fully in-graph. Generalizes the
    /// Phase-2 single-block step to S ∈ {1, 2} slots (WP-1b, arXiv:2606.29215 Alg. 5): per-slot
    /// Γ with the τ_semi-gated top-1 fallback, window-wide Δ, per-slot post counters, front-only
    /// break flags, and the τ_add activation flag. At S == 1 this is exactly the Phase-2 step.
    private func windowStep(
        prefix: MLXArray, prefixLen: Int, windowActive: MLXArray,
        posts: [MLXArray], promptMasks: MLXArray, slotPromptCounts: [Int],
        hasNextBlock: Bool, params: GenerationParams, forward: Forward,
        verifierForward: VerifierForward? = nil,
        tracing: Bool = false,
        jotStableCount: MLXArray,
        prevPredictions: MLXArray,
        frozenMask: MLXArray,
        isAnswerPhase: Bool,
        credit: MLXArray? = nil
    ) -> WindowStepResult {
        let B = params.blockLength
        let S = posts.count
        // Active window length. Normally S*B, but a single slot may be SHORTER than B after an
        // Option-C sub-block prefix commit — so read it from the window rather than assume S*B.
        // `slotLen` is the per-slot length used for all front/slot indexing below: for S==1 it is
        // the (possibly shrunk) window; for S==2 (MultiBD, never combined with Option C) each slot
        // is a full block B. When A==B (no sub-block commit) every use is identical to before.
        let A = windowActive.dim(1)
        let slotLen = S == 1 ? A : B
        let maskId = Int32(params.maskId)

        let activeMask = windowActive .== maskId            // [1, A] Bool
        // Per-slot post counters: each increments only on that block's mask-free iteration
        // (reference semantics per block; the trailing block's budget only bites once promoted).
        var nextPosts: [MLXArray] = []
        for s in 0 ..< S {
            let slotMask = activeMask[0..., (s * slotLen) ..< ((s + 1) * slotLen)]
            let slotMaskAny: MLXArray
            if params.iceEnabled && !isAnswerPhase {
                let thinkingMask = slotMask[0..., 0 ..< params.iceThinkingLength]
                slotMaskAny = thinkingMask.any()
            } else {
                slotMaskAny = slotMask.any()
            }
            nextPosts.append(posts[s] + which(slotMaskAny, MLXArray(Int32(0)), MLXArray(Int32(1))))
        }
        let anyMaskFront: MLXArray
        if params.iceEnabled && !isAnswerPhase {
            anyMaskFront = activeMask[0..., 0 ..< params.iceThinkingLength].any()
        } else {
            anyMaskFront = activeMask[0..., 0 ..< slotLen].any()
        }
        let budgetBreak = nextPosts[0] .> MLXArray(Int32(params.maxPostSteps))  // scalar Bool

        // One forward over the full window, logits for the active columns only.
        let window = prefixLen > 0 ? concatenated([prefix, windowActive], axis: 1) : windowActive
        let logits = forward(window, A, params.jotEnabled ? frozenMask : nil)                     // [1, A, V] FP32

        // Raw model predictions. These govern EDITING (Δ), JOT freezing, the ICE answer-
        // confidence signal, and every diagnostic — none of which credit is allowed to bias.
        let probs = softmax(logits, axis: -1)
        let x0 = argMax(logits, axis: -1).asType(.int32)
        let x0p = probs.max(axis: -1)

        // Credit Decoding (WP-4d, dInfer arXiv:2510.01239). Accumulate stability credit for each
        // position's top candidate and boost that candidate's logit. Per the source, the boost
        // governs the UNMASKING (Γ) pathway ONLY: `unmaskConf` sets a masked position's confidence
        // for the τ_mask test and `unmaskTok` is the token written when it unmasks. Δ-editing,
        // JOT freezing, the ICE answer signal, and diagnostics stay on the raw predictions above,
        // so credit never re-biases an already-decoded position (a deliberate deviation from a
        // naive all-logits boost — recorded in phase-4 §WP-4d).
        let unmaskConf: MLXArray
        let unmaskTok: MLXArray
        let nextCredit: MLXArray?
        if params.creditDecodingEnabled {
            let currentCredit = credit ?? MLXArray.zeros([1, A, model.config.vocabSize], dtype: .float32)
            let decayed = currentCredit * params.creditBeta
            let boost = pow(x0p, params.creditGamma)
            let vocabIndices = arange(model.config.vocabSize, dtype: .int32).reshaped([1, 1, model.config.vocabSize])
            // Localized one-hot broadcast — never materialize a [V, V] identity (V ≈ 157k).
            let oneHot = (x0.expandedDimensions(axis: -1) .== vocabIndices).asType(.float32)
            let updatedCredit = decayed + oneHot * boost.expandedDimensions(axis: -1)
            nextCredit = updatedCredit
            // f_tilde = f + alpha * log1p(C). The enhanced argmax may flip to an earlier-consensus
            // token (the point of credit), so unmaskTok is not necessarily the raw argmax.
            let enhancedLogits = logits + params.creditAlpha * updatedCredit.log1p()
            let enhancedProbs = softmax(enhancedLogits, axis: -1)
            unmaskTok = argMax(enhancedLogits, axis: -1).asType(.int32)
            unmaskConf = enhancedProbs.max(axis: -1)
        } else {
            nextCredit = nil
            unmaskConf = x0p
            unmaskTok = x0
        }

        let avgConfAnswer: MLXArray
        if params.iceEnabled {
            let answerStart = params.iceThinkingLength
            if answerStart < A {
                let answerConf = x0p[0..., answerStart...]
                avgConfAnswer = answerConf.mean().reshaped([1])
            } else {
                avgConfAnswer = MLXArray([Float(0)])
            }
        } else {
            avgConfAnswer = MLXArray([Float(0)])
        }

        let negInf = MLXArray(-Float.infinity)
        // Γ (masked-position) confidence uses the credit-enhanced value; every other consumer of
        // confidence below uses raw x0p (unmaskConf == x0p when credit is disabled).
        let maskConf = which(activeMask, unmaskConf, negInf)       // [1, A]
        let positionsB = MLXArray(0 ..< Int32(slotLen)).reshaped(1, slotLen)

        let gamma: MLXArray            // [1, A] positions written from writeTok this step
        let writeTok: MLXArray         // [1, A] token source (x0, or verifier correction)
        var specAcceptedCount = MLXArray(Float(0))

        // Γ writes the (credit-enhanced) unmask token on masked positions; Δ/JOT writes on
        // unmasked positions keep the raw x0 (the two sets are disjoint). Identity when credit
        // is off, since unmaskTok == x0 there.
        let gammaWriteTok = which(activeMask, unmaskTok, x0)

        if isAnswerPhase {
            gamma = activeMask
            writeTok = gammaWriteTok
        } else if let verifierForward, S == 1 {
            // WP-2a S2D2 self-verification (arXiv:2603.25702 Alg. 3, temp-0 greedy): draft the
            // first contiguous masked span C_t from this forward's x0, verify with one 2B-wide
            // block-size-1-AR forward (M_ver), accept the matching prefix, and take the
            // verifier's token at the first mismatch. Non-span masked positions keep the plain
            // threshold Γ; the top-1 fallback is subsumed (the span always yields ≥1 token
            // while masks exist), and Δ below is untouched (accepted tokens stay editable).
            let firstMasked = argMax(activeMask.asType(.int32), axis: -1)
                .asType(.int32).reshaped(1, 1)                        // [1,1] span start
            let unmaskedCum = cumsum((.!activeMask).asType(.int32), axis: -1)
            // Span membership: masked, and every position before it up to the span start is
            // unmasked (inclusive unmasked-count equals the span start index).
            let spanMask = activeMask .&& (unmaskedCum .== firstMasked)
            let draftFull = which(spanMask, x0, windowActive)         // [1, B]
            let pairWindow = concatenated([draftFull, windowActive], axis: -1)  // [1, 2B]
            let vLogits = verifierForward(pairWindow)                 // [1, 2B, V]
            let qTok = argMax(vLogits[0..., B..., 0...], axis: -1).asType(.int32)  // [1, B]

            let match = (qTok .== x0) .&& spanMask
            let fail = spanMask .&& (.!match)
            let failCum = cumsum(fail.asType(.int32), axis: -1)       // [1, B] inclusive
            let acceptDraft = spanMask .&& (failCum .== MLXArray(Int32(0)))
            let correction = fail .&& (failCum .== MLXArray(Int32(1)))
            let specGamma = acceptDraft .|| correction

            let nonSpan = activeMask .&& (.!spanMask)
            let highConfNonSpan = (maskConf .> MLXArray(params.threshold)) .&& nonSpan
            gamma = specGamma .|| highConfNonSpan
            writeTok = which(correction, qTok, x0)
            specAcceptedCount = specGamma.asType(.float32).sum()
        } else {
            // Γ (M2T) per slot: threshold acceptances plus the top-1 fallback. The fallback is
            // gated on the *preceding* block being semi-complete (progress > τ_semi) or committed
            // (Alg. 5 lines 11–14) — the front's predecessor is committed, so the front keeps the
            // unconditional fallback (== the nBuf=1 semantics). Blocks are evaluated front-to-back
            // within the step, so the gate sees the front's post-Γ progress (Alg. 5 loop order).
            var slotGammas: [MLXArray] = []
            var precedingSemiComplete = MLXArray(true)
            for s in 0 ..< S {
                let r = (s * slotLen) ..< ((s + 1) * slotLen)
                let slotMask = activeMask[0..., r]
                let slotConf = maskConf[0..., r]
                // WP-2b-2 dynamic τ (arXiv:2601.17917): τ(t) = τ0·(1 − α(1 − r_mask)), with
                // r_mask the start-of-step masked fraction over this slot's *generated*
                // positions (prompt positions are never masked, so the raw masked count is
                // already generated-only). α == 0 keeps the static scalar — exact parity.
                // Note: the threshold LOOSENS as the block fills (τ0 at r_mask=1 → τ0(1−α)
                // at r_mask=0) — the formula is authoritative over the source note's prose
                // (logbook note, WP-2b).
                let tauMask: MLXArray
                if params.dynamicTauAlpha > 0 {
                    let genCount = Float(slotLen - slotPromptCounts[s])
                    let rMask = slotMask.asType(.float32).sum() / MLXArray(genCount)
                    tauMask = MLXArray(params.threshold)
                        * (MLXArray(Float(1)) - MLXArray(params.dynamicTauAlpha)
                            * (MLXArray(Float(1)) - rMask))
                } else {
                    tauMask = MLXArray(params.threshold)
                }
                
                let slotConfRestricted: MLXArray
                if params.iceEnabled {
                    let thinkingMask = positionsB .< Int32(params.iceThinkingLength)
                    slotConfRestricted = which(thinkingMask, slotConf, negInf)
                } else {
                    slotConfRestricted = slotConf
                }
                
                let highConf = (slotConfRestricted .> tauMask) .&& slotMask
                let numHigh = highConf.asType(.int32).sum()               // scalar
                let forceTop1 = (numHigh .== MLXArray(Int32(0))) .&& precedingSemiComplete
                let top1 = argMax(slotConfRestricted, axis: -1)                     // [1]
                let oneHotTop1 = positionsB .== top1.reshaped(1, 1)       // [1, B] Bool
                let slotGamma = highConf .|| (forceTop1 .&& oneHotTop1 .&& slotMask)
                slotGammas.append(slotGamma)

                if s + 1 < S {
                    let genCount = Float(slotLen - slotPromptCounts[s])
                    let masksAfter = (slotMask .&& (.!slotGamma)).asType(.float32).sum()
                    let decodedAfter = (MLXArray(genCount) - masksAfter) / MLXArray(genCount)
                    precedingSemiComplete = decodedAfter .> MLXArray(params.tauSemi)
                }
            }
            gamma = S == 1 ? slotGammas[0] : concatenated(slotGammas, axis: -1)  // [1, A]
            writeTok = gammaWriteTok
        }

        // Δ (T2T) over the whole window: unmasked, non-prompt positions clearing τ_edit whose
        // prediction changed. Applies to both active blocks — every position is uncommitted
        // (Alg. 5 line 16; the dual-block edit behavior is a measured open question).
        let editable = (.!activeMask) .&& (.!promptMasks) .&& (.!frozenMask)
        let editConf = which(editable, x0p, negInf)
        let highConfEdit = (editConf .> MLXArray(params.editingThreshold)) .&& editable
        let tokenChanged = x0 .!= windowActive
        let delta = highConfEdit .&& tokenChanged
        let deltaAnyFront = delta[0..., 0 ..< slotLen].any()

        // JOT evaluation (before update):
        let nextJotStableCount: MLXArray
        let nextFrozenMask: MLXArray
        let nextPrevPredictions: MLXArray
        let newlyFrozen: MLXArray

        if params.jotEnabled {
            // stable if prediction unchanged, confidence clears threshold, and not a prompt position
            let isStable = (x0 .== prevPredictions) .&& (x0p .> MLXArray(params.jotThreshold)) .&& (.!promptMasks)
            nextJotStableCount = which(x0 .== prevPredictions, jotStableCount + MLXArray(Int32(1)), MLXArray(Int32(1)))
            newlyFrozen = (nextJotStableCount .>= MLXArray(Int32(params.jotK))) .&& isStable
            nextFrozenMask = frozenMask .|| newlyFrozen
            nextPrevPredictions = x0
        } else {
            nextJotStableCount = jotStableCount
            newlyFrozen = MLXArray.zeros([1, A], dtype: .bool)
            nextFrozenMask = frozenMask
            nextPrevPredictions = prevPredictions
        }

        let finalTransfer = gamma .|| delta .|| newlyFrozen
        var nextWindow = which(finalTransfer, writeTok, windowActive)

        // JOT collision resolution with Δ-editing (after update):
        let nextJotStableCountFinal: MLXArray
        let nextFrozenMaskFinal: MLXArray
        if params.jotEnabled {
            // A collision occurs if Δ edits a position that is currently frozen
            let collision = delta .&& nextFrozenMask
            nextFrozenMaskFinal = nextFrozenMask .&& (.!collision)
            nextJotStableCountFinal = which(collision, MLXArray(Int32(0)), nextJotStableCount)
            // Set the position back to maskId in nextWindow:
            nextWindow = which(collision, MLXArray(maskId), nextWindow)
        } else {
            nextJotStableCountFinal = nextJotStableCount
            nextFrozenMaskFinal = nextFrozenMask
        }

        // Option C (WP-3a §11): length of the FRONT slot's leading contiguous frozen run — the
        // candidate prefix to commit early. A position is in the run iff it and every position
        // before it (within the front slot) are frozen; equivalently, the cumulative count of
        // non-frozen positions up to it is still 0. Read back with the batch (no new sync).
        let frozenPrefixLen: MLXArray
        if params.jotEnabled {
            let frontFrozen = nextFrozenMaskFinal[0..., 0 ..< slotLen]
            let notFrozenCum = cumsum((.!frontFrozen).asType(.int32), axis: -1)
            frozenPrefixLen = (notFrozenCum .== MLXArray(Int32(0))).asType(.int32).sum().reshaped([1])
        } else {
            frozenPrefixLen = MLXArray([Int32(0)])
        }

        // WP-2b-3 EOS early exit (arXiv:2601.17917): once a settled, non-prompt EOS exists in
        // the FRONT block, fill every still-masked window position after the first EOS with
        // EOS in-graph — the block (and any trailing slot) then settles via the normal
        // mask-free break. Trim is inclusive of the first EOS, so output text is unchanged
        // unless Δ would later have edited that EOS away (the measured text-change-rate risk).
        if params.eosEarlyExit {
            let eosTok = MLXArray(Int32(params.eosId))
            let settledEos = (nextWindow .== eosTok) .&& (.!promptMasks)     // [1, A]
            let frontHasEos = settledEos[0..., 0 ..< slotLen].any()                // scalar Bool
            let afterFirstEos = cumsum(settledEos.asType(.int32), axis: -1) .>= MLXArray(Int32(1))
            let filled = which(afterFirstEos .&& (nextWindow .== maskId), eosTok, nextWindow)
            nextWindow = which(frontHasEos, filled, nextWindow)
        }

        // Front break only (in-order commit): settle = front mask-free with no front edits
        // this step; budget = front post-steps over budget (no update applied that iteration —
        // the pre-update window is the result).
        let settleBreak = (.!anyMaskFront) .&& (.!deltaAnyFront)
        let breakFlag = (budgetBreak .|| settleBreak).reshaped([1])
        let resultWindow = which(budgetBreak, windowActive, nextWindow)

        // τ_add activation (Alg. 4/5): in single-active phases with a next block available,
        // fire when the block's post-update decoded fraction over generated positions strictly
        // exceeds τ_add; suppressed while an in-flight eos is present under eos_early_stop.
        let activationFlag: MLXArray
        if S == 1 && hasNextBlock {
            let nextMaskCount = (nextWindow .== maskId).asType(.float32).sum()
            let genCount = Float(slotLen - slotPromptCounts[0])
            let decodedFrac = (MLXArray(genCount) - nextMaskCount) / MLXArray(genCount)
            var activation = decodedFrac .> MLXArray(params.tauAdd)
            if params.eosEarlyStop {
                let eosInFlight = (nextWindow .== MLXArray(Int32(params.eosId))).any()
                activation = activation .&& (.!eosInFlight)
            }
            activationFlag = activation.reshaped([1])
        } else {
            activationFlag = MLXArray([false])
        }

        // Trajectory diagnostics per slot (M8 E1 + τ_semi starvation + WP-2a acceptance).
        var statsParts: [MLXArray] = []
        for s in 0 ..< S {
            let r = (s * slotLen) ..< ((s + 1) * slotLen)
            let slotDelta = delta[0..., r]
            let slotGamma = gamma[0..., r]
            let written = (slotGamma .|| slotDelta).asType(.float32)
            let writtenCount = written.sum()
            let meanConf = (x0p[0..., r].asType(.float32) * written).sum()
                / MLX.maximum(writtenCount, MLXArray(Float(1)))
            let masksRemaining: MLXArray
            if params.iceEnabled && !isAnswerPhase {
                masksRemaining = (nextWindow[0..., r][0..., 0 ..< params.iceThinkingLength] .== maskId).asType(.float32).sum()
            } else {
                masksRemaining = (nextWindow[0..., r] .== maskId).asType(.float32).sum()
            }
            statsParts.append(slotGamma.asType(.float32).sum().reshaped([1]))
            statsParts.append(slotDelta.asType(.float32).sum().reshaped([1]))
            statsParts.append(meanConf.reshaped([1]))
            statsParts.append(masksRemaining.reshaped([1]))
            statsParts.append((s == 0 ? specAcceptedCount : MLXArray(Float(0))).reshaped([1]))
        }
        let stats = concatenated(statsParts, axis: 0)       // [5*S]

        // Offline per-position trace (front slot; WP-2a calibration + JOT pre-experiment).
        let trace: MLXArray? = tracing
            ? concatenated([
                x0p[0..., 0 ..< slotLen].asType(.float32),
                gamma[0..., 0 ..< slotLen].asType(.float32),
                delta[0..., 0 ..< slotLen].asType(.float32),
                x0[0..., 0 ..< slotLen].asType(.float32),
                activeMask[0..., 0 ..< slotLen].asType(.float32),
            ], axis: 0)
            : nil

        return WindowStepResult(
            nextWindow: nextWindow, resultWindow: resultWindow,
            breakFlag: breakFlag, activationFlag: activationFlag,
            nextPosts: nextPosts, stats: stats, trace: trace,
            nextJotStableCount: nextJotStableCountFinal,
            nextPrevPredictions: nextPrevPredictions,
            nextFrozenMask: nextFrozenMaskFinal,
            frozenPrefixLen: frozenPrefixLen,
            avgConfAnswer: avgConfAnswer,
            nextCredit: nextCredit)
    }
}
