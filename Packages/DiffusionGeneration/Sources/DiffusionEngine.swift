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
        /// Effective parameters as the engine actually ran them (provenance rule: benches must
        /// record these echoes, never the CLI inputs — elastic-cache logbook F7).
        public let effectiveNBuf: Int
        public let effectiveTauAdd: Float
        public let effectiveTauSemi: Float
        public let effectiveSpeculationK: Int
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
    public typealias Forward = (_ windowIds: MLXArray, _ activeLen: Int) -> MLXArray

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
        streamBlock: (([Int]) -> Void)? = nil
    ) -> Output {
        let forward: Forward = { [model] windowIds, activeLen in
            let W = windowIds.dim(windowIds.ndim - 1)
            let logits = model.logits(
                forTokens: windowIds, blockLength: params.blockLength,
                maskSemantics: maskSemantics)
            return logits[0..., (W - activeLen)..., 0...]
        }
        return run(prompt: prompt, params: params, forward: forward, streamBlock: streamBlock)
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
        streamBlock: (([Int]) -> Void)? = nil
    ) -> Output {
        let B = params.blockLength
        let cache = ExactPrefixCache(layerCount: model.layerCount)
        let activeCache = ActiveBlockCache(layerCount: model.layerCount)
        let stats = RunStats()

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
        let forward: Forward = { [self, model, cache, activeCache, boundaryHolder] windowIds, activeLen in
            let W = windowIds.dim(windowIds.ndim - 1)
            let activeIds = windowIds[0..., (W - activeLen)...]
            let positionIds = MLXArray(Int32(W - activeLen) ..< Int32(W)).expandedDimensions(axis: 0)

            // Elastic off (the served default): plain cached forward. The elastic overload
            // materializes full attention weights per layer for the drift test — that
            // instrumentation must never run on the serving path.
            guard params.elasticCacheEnabled else {
                // Single active block: mask nil (every committed key is allowed — the
                // ExactPrefixCache argument). Two active blocks: block-causal active mask
                // (the trailing block sees the front, never vice versa — WP-1b).
                guard activeLen > B else {
                    return model(activeIds, positionIds: positionIds, caches: cache.layers)
                }
                let prefixLen = W - activeLen
                let mask = maskMemo[prefixLen] ?? {
                    let m = BlockDiffusionMask.activeWindowMask(
                        prefixLen: prefixLen, activeLen: activeLen, blockLength: B)
                    maskMemo[prefixLen] = m
                    return m
                }()
                return model(activeIds, positionIds: positionIds, caches: cache.layers, mask: mask)
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

        // On settle, run the capture forward over the committed tokens and commit clean K/V.
        let onSettled: (Int, MLXArray) -> Void = { blockIndex, committedActive in
            captureAndCommit(committedActive, startPos: blockIndex * B)
            activeCache.clear()
            stepIndex = 0
            boundaryHolder.value = 0
        }

        return run(prompt: prompt, params: params, activeCache: activeCache, boundaryHolder: boundaryHolder,
                   forward: forward, streamBlock: streamBlock, onBlockSettled: onSettled, stats: stats)
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
        activeCache: ActiveBlockCache? = nil,
        boundaryHolder: BoundaryHolder? = nil,
        forward: Forward,
        streamBlock: (([Int]) -> Void)?,
        onBlockSettled: ((_ blockIndex: Int, _ committedActive: MLXArray) -> Void)? = nil,
        stats: RunStats? = nil
    ) -> Output {
        let B = params.blockLength
        precondition(params.nBuf >= 1 && params.nBuf <= 2, "WP-1b implements nBuf in 1...2")
        precondition(!(params.elasticCacheEnabled && params.nBuf > 1),
            "Elastic-Cache x MultiBD composability is untested (roadmap §5 cell open); "
            + "run one at a time")
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
        var commitSeconds = 0.0
        var transfersPerStep: [[Int]] = []
        var editsPerStep: [[Int]] = []
        var meanConfPerStep: [[Float]] = []
        var trailingStarvedPerBlock: [Int] = []
        var eosBlockIndex: Int? = nil
        var logicalStepsTotal = 0
        var dualActiveSteps = 0
        var activationSteps: [Int] = []

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
            for i in 0 ..< B {
                let global = blockStart + i
                if global < promptLength {
                    initialActive[i] = Int32(prompt[global])
                    promptMaskLocal[i] = true
                }
            }
            let bufferSlot = buffer.activate(blockIndex: numBlock)
            slots.append(SlotRun(
                blockIndex: numBlock,
                bufferSlot: bufferSlot,
                active: MLXArray(initialActive).reshaped(1, B),
                promptMask: MLXArray(promptMaskLocal).reshaped(1, B),
                promptCount: promptMaskLocal.lazy.filter { $0 }.count,
                postSteps: MLXArray(Int32(0))))
            nextBlockIndex += 1
        }

        activateSlot()

        var singleActiveDenoiseSeconds = 0.0
        var dualActiveDenoiseSeconds = 0.0

        while !slots.isEmpty {
            let hasNextBlock = nextBlockIndex < numBlocks
            let phaseWidth = slots.count
            let denoiseStart = Date()
            let phase = denoisePhase(
                prefix: prefixArray, slots: &slots, hasNextBlock: hasNextBlock,
                params: params, activeCache: activeCache, boundaryHolder: boundaryHolder,
                forward: forward,
                globalStep: &logicalStepsTotal, dualActiveSteps: &dualActiveSteps)
            syncPoints += phase.syncPoints
            denoiseForwards += phase.forwards
            let phaseSeconds = Date().timeIntervalSince(denoiseStart)
            denoiseSeconds += phaseSeconds
            if phaseWidth == 2 { dualActiveDenoiseSeconds += phaseSeconds }
            else { singleActiveDenoiseSeconds += phaseSeconds }

            switch phase.event {
            case .activation:
                activationSteps.append(logicalStepsTotal - 1)
                activateSlot()

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

                // First generated block whose generated-region positions contain eos (E1).
                let blockStart = front.blockIndex * B
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
                effectiveNBuf: params.nBuf,
                effectiveTauAdd: params.tauAdd,
                effectiveTauSemi: params.tauSemi,
                effectiveSpeculationK: speculationK))
    }

    // MARK: - Per-phase denoising with K-step speculative readback

    /// Per-block trajectory diagnostics (M8 E1), accumulated by the owning slot.
    struct BlockTrajectory {
        var transfers: [Int] = []
        var edits: [Int] = []
        var meanConfidence: [Float] = []
    }

    /// Why a phase ended. A *phase* is a stretch of logical steps over a fixed set of active
    /// slots; any scheduling event ends the phase at a speculative-batch boundary (overshoot
    /// past the event is discarded, so results are exact and K-invariant).
    enum PhaseEvent {
        /// The front block settled or hit its post-steps budget — commit it.
        case frontBreak
        /// τ_add fired — activate the next block.
        case activation
    }

    struct PhaseResult {
        let event: PhaseEvent
        let syncPoints: Int
        let forwards: Int
        /// The front slot's post-steps counter at the break step (valid on `.frontBreak`).
        let frontPostAtBreak: Int
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
        var syncPoints = 0
        var forwardsEvaluated = 0

        var windowActive = S == 1 ? slots[0].active
            : concatenated(slots.map(\.active), axis: 1)
        let promptMasks = S == 1 ? slots[0].promptMask
            : concatenated(slots.map(\.promptMask), axis: 1)
        let slotPromptCounts = slots.map(\.promptCount)
        var posts = slots.map(\.postSteps)

        while true {
            // Build up to K speculative steps into one graph.
            var specWindow = windowActive
            var specPosts = posts
            var snapshots: [MLXArray] = []     // result window if the break lands here
            var nextWindows: [MLXArray] = []   // window carried to the next step
            var breakFlags: [MLXArray] = []    // [1] Bool per step (front break)
            var activationFlags: [MLXArray] = [] // [1] Bool per step (τ_add)
            var postsPerStep: [[MLXArray]] = []
            var statsPerStep: [MLXArray] = []  // [4*S] per step

            for _ in 0 ..< K {
                let s = windowStep(
                    prefix: prefix, prefixLen: prefixLen, windowActive: specWindow,
                    posts: specPosts, promptMasks: promptMasks,
                    slotPromptCounts: slotPromptCounts,
                    hasNextBlock: hasNextBlock, params: params, forward: forward)
                snapshots.append(s.resultWindow)
                nextWindows.append(s.nextWindow)
                breakFlags.append(s.breakFlag)
                activationFlags.append(s.activationFlag)
                postsPerStep.append(s.nextPosts)
                statsPerStep.append(s.stats)
                specWindow = s.nextWindow
                specPosts = s.nextPosts
            }

            // Single blocking readback of the 2K stacked event flags.
            let flags = concatenated(breakFlags + activationFlags, axis: 0)  // [2K] Bool
            eval(flags)
            eval(snapshots + nextWindows + postsPerStep.flatMap { $0 } + statsPerStep)

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
            forwardsEvaluated += K

            let firstBreak = flagVals[0 ..< K].firstIndex(of: true)
            let firstActivation = flagVals[K ..< 2 * K].firstIndex(of: true).map { $0 - K }

            // Route the already-evaluated per-step stats of `count` logical steps to their slots.
            func applyStats(_ count: Int) {
                for r in 0 ..< count {
                    let vals = statsPerStep[r].asArray(Float.self)  // materialized — memcpy
                    for s in slots.indices {
                        slots[s].trajectory.transfers.append(Int(vals[4 * s]))
                        slots[s].trajectory.edits.append(Int(vals[4 * s + 1]))
                        slots[s].trajectory.meanConfidence.append(vals[4 * s + 2])
                        if s > 0 && vals[4 * s] == 0 && vals[4 * s + 3] > 0 {
                            slots[s].trailingStarvedSteps += 1
                        }
                    }
                }
            }
            func applyWindow(_ window: MLXArray, posts stepPosts: [MLXArray]) {
                for s in slots.indices {
                    slots[s].active = S == 1
                        ? window : window[0..., (s * B) ..< ((s + 1) * B)]
                    slots[s].postSteps = stepPosts[s]
                }
            }
            func addSteps(_ n: Int) {
                for s in slots.indices { slots[s].stepsTaken += n }
                globalStep += n
                if S == 2 { dualActiveSteps += n }
            }

            // Break wins a same-step tie: the commit re-derives activation from fresh state.
            if let j = firstBreak, j <= (firstActivation ?? Int.max) {
                applyStats(j + 1)
                // On a budget break the whole window's step-j write is discarded (simplest
                // K-invariant rule; the trailing slot's step-j write goes with it — recorded
                // deviation, WP-1b logbook). On a settle break `resultWindow == nextWindow`.
                applyWindow(snapshots[j], posts: postsPerStep[j])
                addSteps(j + 1)
                let frontPost = Int(postsPerStep[j][0].item(Int32.self))  // memcpy, evaluated above
                return PhaseResult(
                    event: .frontBreak, syncPoints: syncPoints,
                    forwards: forwardsEvaluated, frontPostAtBreak: frontPost)
            }
            if let j = firstActivation {
                applyStats(j + 1)
                applyWindow(nextWindows[j], posts: postsPerStep[j])
                addSteps(j + 1)
                return PhaseResult(
                    event: .activation, syncPoints: syncPoints,
                    forwards: forwardsEvaluated, frontPostAtBreak: 0)
            }

            // No event within this batch — advance by K steps and continue.
            applyStats(K)
            applyWindow(specWindow, posts: specPosts)
            addSteps(K)
            windowActive = specWindow
            posts = specPosts
        }
    }

    /// Result of one speculative window step.
    private struct WindowStepResult {
        let nextWindow: MLXArray     // [1, S*B] window carried forward (post-update)
        let resultWindow: MLXArray   // window if the loop breaks at this step (budget → pre-update)
        let breakFlag: MLXArray      // [1] Bool — front settle or front budget
        let activationFlag: MLXArray // [1] Bool — τ_add fired (single-active phases only)
        let nextPosts: [MLXArray]    // per-slot post-steps accumulators carried forward
        /// Diagnostics `[4*S]` Float32 per slot: |Γ|, |Δ|, mean x0_p over written positions,
        /// masks remaining post-update. Read back only from the already-evaluated batch.
        let stats: MLXArray
    }

    /// One denoising step over the concatenated active window, fully in-graph. Generalizes the
    /// Phase-2 single-block step to S ∈ {1, 2} slots (WP-1b, arXiv:2606.29215 Alg. 5): per-slot
    /// Γ with the τ_semi-gated top-1 fallback, window-wide Δ, per-slot post counters, front-only
    /// break flags, and the τ_add activation flag. At S == 1 this is exactly the Phase-2 step.
    private func windowStep(
        prefix: MLXArray, prefixLen: Int, windowActive: MLXArray,
        posts: [MLXArray], promptMasks: MLXArray, slotPromptCounts: [Int],
        hasNextBlock: Bool, params: GenerationParams, forward: Forward
    ) -> WindowStepResult {
        let B = params.blockLength
        let S = posts.count
        let A = S * B
        let maskId = Int32(params.maskId)

        let activeMask = windowActive .== maskId            // [1, A] Bool
        // Per-slot post counters: each increments only on that block's mask-free iteration
        // (reference semantics per block; the trailing block's budget only bites once promoted).
        var nextPosts: [MLXArray] = []
        for s in 0 ..< S {
            let slotMaskAny = activeMask[0..., (s * B) ..< ((s + 1) * B)].any()
            nextPosts.append(posts[s] + which(slotMaskAny, MLXArray(Int32(0)), MLXArray(Int32(1))))
        }
        let anyMaskFront = activeMask[0..., 0 ..< B].any()
        let budgetBreak = nextPosts[0] .> MLXArray(Int32(params.maxPostSteps))  // scalar Bool

        // One forward over the full window, logits for the active columns only.
        let window = prefixLen > 0 ? concatenated([prefix, windowActive], axis: 1) : windowActive
        let logits = forward(window, A)                     // [1, A, V] FP32
        let probs = softmax(logits, axis: -1)
        let x0 = argMax(logits, axis: -1).asType(.int32)    // [1, A]
        let x0p = probs.max(axis: -1)                       // [1, A]

        let negInf = MLXArray(-Float.infinity)
        let maskConf = which(activeMask, x0p, negInf)       // [1, A]
        let positionsB = MLXArray(0 ..< Int32(B)).reshaped(1, B)

        // Γ (M2T) per slot: threshold acceptances plus the top-1 fallback. The fallback is
        // gated on the *preceding* block being semi-complete (progress > τ_semi) or committed
        // (Alg. 5 lines 11–14) — the front's predecessor is committed, so the front keeps the
        // unconditional fallback (== the nBuf=1 semantics). Blocks are evaluated front-to-back
        // within the step, so the gate sees the front's post-Γ progress (Alg. 5 loop order).
        var slotGammas: [MLXArray] = []
        var precedingSemiComplete = MLXArray(true)
        for s in 0 ..< S {
            let r = (s * B) ..< ((s + 1) * B)
            let slotMask = activeMask[0..., r]
            let slotConf = maskConf[0..., r]
            let highConf = (slotConf .> MLXArray(params.threshold)) .&& slotMask
            let numHigh = highConf.asType(.int32).sum()               // scalar
            let forceTop1 = (numHigh .== MLXArray(Int32(0))) .&& precedingSemiComplete
            let top1 = argMax(slotConf, axis: -1)                     // [1]
            let oneHotTop1 = positionsB .== top1.reshaped(1, 1)       // [1, B] Bool
            let gamma = highConf .|| (forceTop1 .&& oneHotTop1 .&& slotMask)
            slotGammas.append(gamma)

            if s + 1 < S {
                let genCount = Float(B - slotPromptCounts[s])
                let masksAfter = (slotMask .&& (.!gamma)).asType(.float32).sum()
                let decodedAfter = (MLXArray(genCount) - masksAfter) / MLXArray(genCount)
                precedingSemiComplete = decodedAfter .> MLXArray(params.tauSemi)
            }
        }
        let gamma = S == 1 ? slotGammas[0] : concatenated(slotGammas, axis: -1)  // [1, A]

        // Δ (T2T) over the whole window: unmasked, non-prompt positions clearing τ_edit whose
        // prediction changed. Applies to both active blocks — every position is uncommitted
        // (Alg. 5 line 16; the dual-block edit behavior is a measured open question).
        let editable = (.!activeMask) .&& (.!promptMasks)
        let editConf = which(editable, x0p, negInf)
        let highConfEdit = (editConf .> MLXArray(params.editingThreshold)) .&& editable
        let tokenChanged = x0 .!= windowActive
        let delta = highConfEdit .&& tokenChanged
        let deltaAnyFront = delta[0..., 0 ..< B].any()

        let finalTransfer = gamma .|| delta
        let nextWindow = which(finalTransfer, x0, windowActive)

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
            let genCount = Float(B - slotPromptCounts[0])
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

        // Trajectory diagnostics per slot (M8 E1 + τ_semi starvation).
        var statsParts: [MLXArray] = []
        for s in 0 ..< S {
            let r = (s * B) ..< ((s + 1) * B)
            let slotDelta = delta[0..., r]
            let written = (slotGammas[s] .|| slotDelta).asType(.float32)
            let writtenCount = written.sum()
            let meanConf = (x0p[0..., r].asType(.float32) * written).sum()
                / MLX.maximum(writtenCount, MLXArray(Float(1)))
            let masksRemaining = (nextWindow[0..., r] .== maskId).asType(.float32).sum()
            statsParts.append(slotGammas[s].asType(.float32).sum().reshaped([1]))
            statsParts.append(slotDelta.asType(.float32).sum().reshaped([1]))
            statsParts.append(meanConf.reshaped([1]))
            statsParts.append(masksRemaining.reshaped([1]))
        }
        let stats = concatenated(statsParts, axis: 0)       // [4*S]

        return WindowStepResult(
            nextWindow: nextWindow, resultWindow: resultWindow,
            breakFlag: breakFlag, activationFlag: activationFlag,
            nextPosts: nextPosts, stats: stats)
    }
}
