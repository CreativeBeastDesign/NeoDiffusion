import Foundation
import MLX
import MLXNN
import DiffusionCore
import DiffusionModel

/// The WP-1b slot scheduler: the block schedule shared by the cached and cache-disabled
/// paths, its in-flight slot state, and the speculation predicate. Split out of the engine
/// core for readability (WP-4 refactor); same module, no behavioural change.
extension DiffusionEngine {
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
    /// and a verifier closure available (cached path). Internal (not private): the scheduler and
    /// the step function live in separate files after the WP-4 refactor.
    func speculating(
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
}
