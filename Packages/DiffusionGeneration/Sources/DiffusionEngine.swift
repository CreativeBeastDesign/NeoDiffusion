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

        // Active-only forward: slice the active block out of the window; positions are absolute
        // from the committed length (== window length − active length).
        let forward: Forward = { [model] windowIds, activeLen in
            let W = windowIds.dim(windowIds.ndim - 1)
            let activeIds = windowIds[0..., (W - activeLen)...]
            let positionIds = MLXArray(Int32(W - activeLen) ..< Int32(W)).expandedDimensions(axis: 0)
            return model(activeIds, positionIds: positionIds, caches: cache.layers)
        }

        // On settle, run the capture forward over the committed tokens and commit clean K/V.
        let onSettled: (Int, MLXArray) -> Void = { blockIndex, committedActive in
            captureAndCommit(committedActive, startPos: blockIndex * B)
        }

        return run(prompt: prompt, params: params, forward: forward,
                   streamBlock: streamBlock, onBlockSettled: onSettled, stats: stats)
    }

    // MARK: - Shared loop

    /// The block schedule shared by the cache-disabled and cached paths. `forward` abstracts how
    /// logits over the active block are produced.
    func run(
        prompt: [Int],
        params: GenerationParams,
        forward: Forward,
        streamBlock: (([Int]) -> Void)?,
        onBlockSettled: ((_ blockIndex: Int, _ committedActive: MLXArray) -> Void)? = nil,
        stats: RunStats? = nil
    ) -> Output {
        let B = params.blockLength
        let promptLength = prompt.count
        let numBlocks = (promptLength + params.genLength + B - 1) / B
        let totalLength = numBlocks * B
        let prefillBlocks = promptLength / B
        let maskId = Int32(params.maskId)

        var buffer = BlockBuffer(nBuf: 1)
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
        var eosBlockIndex: Int? = nil

        for numBlock in prefillBlocks ..< numBlocks {
            let slot = buffer.activate(blockIndex: numBlock)
            let blockStart = numBlock * B

            // Initial active block: prompt tail where it exists, mask id elsewhere.
            var initialActive = [Int32](repeating: maskId, count: B)
            var promptMaskLocal = [Bool](repeating: false, count: B)
            for i in 0 ..< B {
                let global = blockStart + i
                if global < promptLength {
                    initialActive[i] = Int32(prompt[global])
                    promptMaskLocal[i] = true
                }
            }
            let activeInit = MLXArray(initialActive).reshaped(1, B)
            let promptMask = MLXArray(promptMaskLocal).reshaped(1, B)  // Bool [1, B]

            let denoiseStart = Date()
            let (committedActive, steps, syncs, postSteps, forwards, trajectory) = denoiseBlock(
                prefix: prefixArray,
                initialActive: activeInit,
                promptMask: promptMask,
                params: params,
                forward: forward)
            syncPoints += syncs
            denoiseForwards += forwards
            denoiseSeconds += Date().timeIntervalSince(denoiseStart)
            transfersPerStep.append(trajectory.transfers)
            editsPerStep.append(trajectory.edits)
            meanConfPerStep.append(trajectory.meanConfidence)

            // Commit-cleanliness / KV capture hook (cached path); no-op for cache-disabled.
            let commitStart = Date()
            buffer.markSettled(slotIndex: slot)
            onBlockSettled?(numBlock, committedActive)

            // One readback per commit: materialize the block's ids for streaming + prefix growth.
            let blockIds = committedActive.asArray(Int32.self).map { Int($0) }
            syncPoints += 1

            committedIds.append(contentsOf: blockIds)
            prefixArray = concatenated([prefixArray, committedActive], axis: 1)
            buffer.markCommitted(slotIndex: slot)

            blockCommits.append(committedIds)
            stepsPerBlock.append(steps)
            postStepsPerBlock.append(postSteps)
            // First generated block whose generated-region positions contain eos (E1).
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

            // eos_early_stop: a committed block is always mask-free (both break conditions imply
            // no masks), so we only test whether the generated region so far contains eos.
            if params.eosEarlyStop {
                let generated = committedIds[promptLength...]
                if generated.contains(params.eosId) { break }
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
                eosBlockIndex: eosBlockIndex))
    }

    // MARK: - Per-block denoising with K-step speculative readback

    /// Denoise one active block to settlement. Returns the committed active block `[1, B]`, the
    /// number of denoising steps taken (matching the reference's per-block step count), the
    /// number of blocking readbacks (1 per speculative batch of ≤ K steps), the `post_steps`
    /// counter at loop exit (M6 diagnostic — read from the already-evaluated batch, so it adds
    /// no synchronization), and the number of forwards evaluated (K per batch, honestly
    /// counting speculative overshoot).
    struct BlockTrajectory {
        var transfers: [Int] = []
        var edits: [Int] = []
        var meanConfidence: [Float] = []
    }

    func denoiseBlock(
        prefix: MLXArray,
        initialActive: MLXArray,
        promptMask: MLXArray,
        params: GenerationParams,
        forward: Forward
    ) -> (committed: MLXArray, steps: Int, syncPoints: Int, postSteps: Int, forwards: Int,
          trajectory: BlockTrajectory) {
        precondition(params.numToTransfer == 1,
            "Phase 2 parity implements numToTransfer == 1 (the live reference value); "
            + "num_to_transfer > 1 needs index-exact top-k tie handling (M6 bench extension)")
        let prefixLen = prefix.dim(1)

        var active = initialActive          // [1, B] Int32
        var postSteps = MLXArray(Int32(0))  // in-graph accumulator, no per-step readback
        var stepsTaken = 0
        var syncPoints = 0
        var forwardsEvaluated = 0
        var trajectory = BlockTrajectory()

        // Append the already-evaluated per-step stats of the `count` logical steps.
        func recordStats(_ stats: [MLXArray], count: Int) {
            for s in stats.prefix(count) {
                let vals = s.asArray(Float.self)  // materialized by the batch eval — memcpy
                trajectory.transfers.append(Int(vals[0]))
                trajectory.edits.append(Int(vals[1]))
                trajectory.meanConfidence.append(vals[2])
            }
        }

        while true {
            // Build up to K speculative steps into one graph.
            var specActive = active
            var specPost = postSteps
            var snapshots: [MLXArray] = []   // result active block if the break is here
            var nextActives: [MLXArray] = [] // active block carried to the next step
            var breakFlags: [MLXArray] = []  // scalar Bool per step
            var posts: [MLXArray] = []       // post_steps after each step (M6 diagnostic)
            var stats: [MLXArray] = []       // [3] per step (M8 E1 trajectory diagnostic)
            for _ in 0 ..< speculationK {
                let s = step(
                    prefix: prefix, prefixLen: prefixLen, active: specActive, postSteps: specPost,
                    promptMask: promptMask, params: params, forward: forward)
                snapshots.append(s.resultActive)
                nextActives.append(s.nextActive)
                breakFlags.append(s.breakFlag)
                posts.append(s.nextPost)
                stats.append(s.stats)
                specActive = s.nextActive
                specPost = s.nextPost
            }

            // Single blocking readback of the K stacked break flags.
            let flags = concatenated(breakFlags, axis: 0)  // [K] Bool
            eval(flags)
            eval(snapshots + nextActives + posts + stats)
            let flagVals = flags.asArray(Bool.self)
            syncPoints += 1
            forwardsEvaluated += speculationK

            if let firstBreak = flagVals.firstIndex(of: true) {
                stepsTaken += firstBreak + 1
                // `posts[firstBreak]` / `stats[...]` were materialized by the eval above —
                // reading them is a memcpy, not a graph sync (M5(c) budget unaffected).
                let postAtBreak = Int(posts[firstBreak].item(Int32.self))
                recordStats(stats, count: firstBreak + 1)
                return (snapshots[firstBreak], stepsTaken, syncPoints, postAtBreak,
                        forwardsEvaluated, trajectory)
            }

            // No break within this batch — advance by K steps and continue.
            stepsTaken += speculationK
            recordStats(stats, count: speculationK)
            active = specActive
            postSteps = specPost
        }
    }

    /// Result of one speculative step.
    private struct StepResult {
        let nextActive: MLXArray   // active block carried forward (post-update)
        let resultActive: MLXArray // committed active block *if the loop breaks at this step*
        let breakFlag: MLXArray    // scalar Bool
        let nextPost: MLXArray     // post_steps accumulator carried forward
        /// Trajectory diagnostics `[3]` Float32: |Γ|, |Δ|, mean x0_p over written positions
        /// (0 when nothing written). Read back only from the already-evaluated batch.
        let stats: MLXArray
    }

    /// One denoising step, fully in-graph. Mirrors the reference loop body: recompute
    /// `active_mask`, increment `post_steps` on a mask-free iteration, forward once, sample
    /// (argmax + confidence), build Γ ∪ Δ, and apply. Both break conditions are computed as
    /// scalar flags so the caller can find the break point without a per-step readback.
    private func step(
        prefix: MLXArray, prefixLen: Int, active: MLXArray, postSteps: MLXArray,
        promptMask: MLXArray, params: GenerationParams, forward: Forward
    ) -> StepResult {
        let B = params.blockLength
        let maskId = Int32(params.maskId)

        let activeMask = active .== maskId                 // [1, B] Bool
        let anyMask = activeMask.any()                     // scalar Bool
        // post_steps increments only on a mask-free iteration (reference semantics).
        let post = postSteps + which(anyMask, MLXArray(Int32(0)), MLXArray(Int32(1)))
        let budgetBreak = post .> MLXArray(Int32(params.maxPostSteps))  // scalar Bool

        // One forward, over the full window, logits for the active block only.
        let window = prefixLen > 0 ? concatenated([prefix, active], axis: 1) : active
        let logits = forward(window, B)                    // [1, B, V] FP32
        let probs = softmax(logits, axis: -1)
        let x0 = argMax(logits, axis: -1).asType(.int32)   // [1, B]
        let x0p = probs.max(axis: -1)                      // [1, B] confidence of argmax token

        // Γ (M2T): masked positions clearing τ_mask; if none, force the single most-confident
        // masked position (numToTransfer == 1). Branchless — no readback of the high-conf count.
        let negInf = MLXArray(-Float.infinity)
        let maskConf = which(activeMask, x0p, negInf)
        let highConfMask = (maskConf .> MLXArray(params.threshold)) .&& activeMask
        let numHigh = highConfMask.asType(.int32).sum()               // scalar
        let forceTop1 = numHigh .== MLXArray(Int32(0))                // scalar Bool
        let top1 = argMax(maskConf, axis: -1)                         // [1]
        let positions = MLXArray(0 ..< Int32(B)).reshaped(1, B)
        let oneHotTop1 = positions .== top1.reshaped(1, 1)            // [1, B] Bool
        let gamma = highConfMask .|| (forceTop1 .&& oneHotTop1 .&& activeMask)

        // Δ (T2T): unmasked, non-prompt positions clearing τ_edit whose prediction changed.
        let editable = (.!activeMask) .&& (.!promptMask)
        let editConf = which(editable, x0p, negInf)
        let highConfEdit = (editConf .> MLXArray(params.editingThreshold)) .&& editable
        let tokenChanged = x0 .!= active
        let delta = highConfEdit .&& tokenChanged
        let deltaAny = delta.any()

        let finalTransfer = gamma .|| delta
        let nextActive = which(finalTransfer, x0, active)

        // Settle break: no masks and no edits this step. Budget break: post_steps over budget
        // (no update applied that iteration → result is the pre-update window).
        let settleBreak = (.!anyMask) .&& (.!deltaAny)
        // Reshape to [1] so the caller can `concatenated(..., axis: 0)` the K flags into one
        // array for a single readback (0-dim scalars cannot be concatenated).
        let breakFlag = (budgetBreak .|| settleBreak).reshaped([1])
        let resultActive = which(budgetBreak, active, nextActive)

        // Trajectory diagnostics (M8 E1): counts + mean confidence of what this step wrote.
        // On a budget-break step the write is discarded by `resultActive`, but the diagnostic
        // still reports what the step *computed* — callers see the discarded step's stats
        // only if it is the break step, matching stepsPerBlock's +1 semantics.
        let gammaCount = gamma.asType(.float32).sum()
        let deltaCount = delta.asType(.float32).sum()
        let written = finalTransfer.asType(.float32)
        let writtenCount = written.sum()
        let meanConf = (x0p.asType(.float32) * written).sum()
            / MLX.maximum(writtenCount, MLXArray(Float(1)))
        let stats = concatenated(
            [gammaCount.reshaped([1]), deltaCount.reshaped([1]), meanConf.reshaped([1])],
            axis: 0)

        return StepResult(
            nextActive: nextActive, resultActive: resultActive,
            breakFlag: breakFlag, nextPost: post, stats: stats)
    }
}
