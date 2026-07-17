import Foundation
import MLX
import MLXNN
import DiffusionCore
import DiffusionModel

/// The per-phase denoising loop and the single fused in-graph step (windowStep). This is
/// the hot path: one MLX graph per K-step speculative batch, one readback per batch. Split
/// out of the engine core for readability (WP-4 refactor); same module, no graph change.
extension DiffusionEngine {
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

        // Credit Decoding governs the Γ (unmasking) pathway only; raw predictions above are left
        // untouched for Δ/JOT/ICE/diagnostics. `nextCredit` is the carried-forward matrix.
        let (unmaskConf, unmaskTok, nextCredit) = creditDecode(
            logits: logits, x0: x0, x0p: x0p, activeLen: A, credit: credit, params: params)

        let avgConfAnswer = iceAnswerConfidence(x0p: x0p, activeLen: A, params: params)

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

        // JOT evaluation (before the window update): grow per-position stability counts and
        // freeze positions that have been stable for `jotK` steps above threshold.
        let jot = jotEvaluate(
            x0: x0, x0p: x0p, prevPredictions: prevPredictions, jotStableCount: jotStableCount,
            frozenMask: frozenMask, promptMasks: promptMasks, activeLen: A, params: params)
        let nextPrevPredictions = jot.prevPredictions

        let finalTransfer = gamma .|| delta .|| jot.newlyFrozen
        var nextWindow = which(finalTransfer, writeTok, windowActive)

        // JOT collision resolution (after the update): if Δ edited a frozen position, unfreeze it,
        // reset its stability count, and re-mask it. Returns the possibly-remasked window.
        let jotFinal = jotResolveCollision(
            nextWindow: nextWindow, delta: delta, frozenMask: jot.frozenMask,
            jotStableCount: jot.stableCount, maskId: maskId, params: params)
        nextWindow = jotFinal.window
        let nextJotStableCountFinal = jotFinal.stableCount
        let nextFrozenMaskFinal = jotFinal.frozenMask

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

        // WP-2b-3 EOS early exit: fill still-masked positions after a settled front-block EOS.
        nextWindow = eosEarlyExitFill(
            nextWindow: nextWindow, promptMasks: promptMasks, slotLen: slotLen,
            maskId: maskId, params: params)

        // Front break only (in-order commit): settle = front mask-free with no front edits
        // this step; budget = front post-steps over budget (no update applied that iteration —
        // the pre-update window is the result).
        let settleBreak = (.!anyMaskFront) .&& (.!deltaAnyFront)
        let breakFlag = (budgetBreak .|| settleBreak).reshaped([1])
        let resultWindow = which(budgetBreak, windowActive, nextWindow)

        // τ_add activation (Alg. 4/5): single-active phases only; fires off the post-update
        // decoded fraction, suppressed while an in-flight eos is present under eos_early_stop.
        let activationFlag = activationFlagFor(
            nextWindow: nextWindow, slotLen: slotLen, slotPromptCount: slotPromptCounts[0],
            slotCount: S, hasNextBlock: hasNextBlock, maskId: maskId, params: params)

        // Trajectory diagnostics per slot (M8 E1 + τ_semi starvation + WP-2a acceptance).
        let stats = stepStats(
            gamma: gamma, delta: delta, x0p: x0p, nextWindow: nextWindow, slotLen: slotLen,
            slotCount: S, specAcceptedCount: specAcceptedCount, maskId: maskId,
            isAnswerPhase: isAnswerPhase, params: params)

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

    // MARK: - windowStep lazy helpers
    //
    // Each returns un-`eval`'d MLXArrays that compose into windowStep's single fused graph — no
    // per-helper eval, no dynamic dispatch (`private final`-context, inlined). They exist only to
    // keep the step readable; behaviour is identical to the pre-refactor inline blocks.

    /// Credit Decoding (WP-4d, dInfer arXiv:2510.01239). Accumulates stability credit for each
    /// position's top candidate and boosts that candidate's logit. The boost governs the UNMASKING
    /// (Γ) pathway ONLY: the returned `unmaskConf` sets a masked position's confidence for the
    /// τ_mask test and `unmaskTok` is the token written when it unmasks. Callers keep the raw
    /// predictions for Δ-editing, JOT, the ICE answer signal, and diagnostics, so credit never
    /// re-biases an already-decoded position (deliberate deviation — phase-4 §WP-4d). When disabled
    /// this returns `(x0p, x0, nil)`, i.e. exact parity.
    private func creditDecode(
        logits: MLXArray, x0: MLXArray, x0p: MLXArray, activeLen A: Int,
        credit: MLXArray?, params: GenerationParams
    ) -> (unmaskConf: MLXArray, unmaskTok: MLXArray, nextCredit: MLXArray?) {
        guard params.creditDecodingEnabled else { return (x0p, x0, nil) }
        let currentCredit = credit ?? MLXArray.zeros([1, A, model.config.vocabSize], dtype: .float32)
        let decayed = currentCredit * params.creditBeta
        let boost = pow(x0p, params.creditGamma)
        let vocabIndices = arange(model.config.vocabSize, dtype: .int32).reshaped([1, 1, model.config.vocabSize])
        // Localized one-hot broadcast — never materialize a [V, V] identity (V ≈ 157k).
        let oneHot = (x0.expandedDimensions(axis: -1) .== vocabIndices).asType(.float32)
        let updatedCredit = decayed + oneHot * boost.expandedDimensions(axis: -1)
        // f_tilde = f + alpha * log1p(C). The enhanced argmax may flip to an earlier-consensus
        // token (the point of credit), so unmaskTok is not necessarily the raw argmax.
        let enhancedLogits = logits + params.creditAlpha * updatedCredit.log1p()
        let enhancedProbs = softmax(enhancedLogits, axis: -1)
        return (enhancedProbs.max(axis: -1),
                argMax(enhancedLogits, axis: -1).asType(.int32),
                updatedCredit)
    }

    /// ICE (WP-4c) answer-region convergence signal: mean raw confidence over the answer window
    /// `[iceThinkingLength, A)`. `[0]` when ICE is off or the answer window is empty.
    private func iceAnswerConfidence(x0p: MLXArray, activeLen A: Int, params: GenerationParams) -> MLXArray {
        guard params.iceEnabled, params.iceThinkingLength < A else { return MLXArray([Float(0)]) }
        return x0p[0..., params.iceThinkingLength...].mean().reshaped([1])
    }

    /// JOT (WP-3a) stability evaluation, before the window update. Grows each position's stable
    /// count (reset on a prediction change), and freezes positions stable for `jotK` steps above
    /// threshold that are not prompt positions. When JOT is off, returns the inputs unchanged with
    /// an all-false `newlyFrozen`.
    private func jotEvaluate(
        x0: MLXArray, x0p: MLXArray, prevPredictions: MLXArray, jotStableCount: MLXArray,
        frozenMask: MLXArray, promptMasks: MLXArray, activeLen A: Int, params: GenerationParams
    ) -> (stableCount: MLXArray, frozenMask: MLXArray, prevPredictions: MLXArray, newlyFrozen: MLXArray) {
        guard params.jotEnabled else {
            return (jotStableCount, frozenMask, prevPredictions, MLXArray.zeros([1, A], dtype: .bool))
        }
        let isStable = (x0 .== prevPredictions) .&& (x0p .> MLXArray(params.jotThreshold)) .&& (.!promptMasks)
        let stableCount = which(x0 .== prevPredictions, jotStableCount + MLXArray(Int32(1)), MLXArray(Int32(1)))
        let newlyFrozen = (stableCount .>= MLXArray(Int32(params.jotK))) .&& isStable
        return (stableCount, frozenMask .|| newlyFrozen, x0, newlyFrozen)
    }

    /// JOT collision resolution, after the window update: where Δ edited a currently-frozen
    /// position, unfreeze it, reset its stable count, and re-mask that cell in the window. When JOT
    /// is off, returns the window and states unchanged.
    private func jotResolveCollision(
        nextWindow: MLXArray, delta: MLXArray, frozenMask: MLXArray, jotStableCount: MLXArray,
        maskId: Int32, params: GenerationParams
    ) -> (window: MLXArray, frozenMask: MLXArray, stableCount: MLXArray) {
        guard params.jotEnabled else { return (nextWindow, frozenMask, jotStableCount) }
        let collision = delta .&& frozenMask
        return (which(collision, MLXArray(maskId), nextWindow),
                frozenMask .&& (.!collision),
                which(collision, MLXArray(Int32(0)), jotStableCount))
    }

    /// WP-2b-3 EOS early exit (arXiv:2601.17917): once a settled, non-prompt EOS exists in the
    /// FRONT block, fill every still-masked position after the first EOS with EOS so the block
    /// settles via the normal mask-free break. Trim is inclusive of the first EOS, so output text
    /// is unchanged unless Δ would later have edited that EOS away. No-op when disabled.
    private func eosEarlyExitFill(
        nextWindow: MLXArray, promptMasks: MLXArray, slotLen: Int, maskId: Int32,
        params: GenerationParams
    ) -> MLXArray {
        guard params.eosEarlyExit else { return nextWindow }
        let eosTok = MLXArray(Int32(params.eosId))
        let settledEos = (nextWindow .== eosTok) .&& (.!promptMasks)     // [1, A]
        let frontHasEos = settledEos[0..., 0 ..< slotLen].any()               // scalar Bool
        let afterFirstEos = cumsum(settledEos.asType(.int32), axis: -1) .>= MLXArray(Int32(1))
        let filled = which(afterFirstEos .&& (nextWindow .== maskId), eosTok, nextWindow)
        return which(frontHasEos, filled, nextWindow)
    }

    /// τ_add activation flag (WP-1b Alg. 4/5): fires in single-active phases with a next block
    /// available when the front block's post-update decoded fraction (over generated positions)
    /// strictly exceeds τ_add; suppressed while an in-flight eos is present under eos_early_stop.
    /// `[false]` otherwise.
    private func activationFlagFor(
        nextWindow: MLXArray, slotLen: Int, slotPromptCount: Int, slotCount S: Int,
        hasNextBlock: Bool, maskId: Int32, params: GenerationParams
    ) -> MLXArray {
        guard S == 1 && hasNextBlock else { return MLXArray([false]) }
        let nextMaskCount = (nextWindow .== maskId).asType(.float32).sum()
        let genCount = Float(slotLen - slotPromptCount)
        let decodedFrac = (MLXArray(genCount) - nextMaskCount) / MLXArray(genCount)
        var activation = decodedFrac .> MLXArray(params.tauAdd)
        if params.eosEarlyStop {
            let eosInFlight = (nextWindow .== MLXArray(Int32(params.eosId))).any()
            activation = activation .&& (.!eosInFlight)
        }
        return activation.reshaped([1])
    }

    /// Per-slot trajectory diagnostics `[5*S]` (M8 E1 + τ_semi starvation + WP-2a acceptance):
    /// for each slot |Γ|, |Δ|, mean written-position confidence, masks remaining post-update, and
    /// (slot 0 only) the speculation-accepted count.
    private func stepStats(
        gamma: MLXArray, delta: MLXArray, x0p: MLXArray, nextWindow: MLXArray, slotLen: Int,
        slotCount S: Int, specAcceptedCount: MLXArray, maskId: Int32, isAnswerPhase: Bool,
        params: GenerationParams
    ) -> MLXArray {
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
        return concatenated(statsParts, axis: 0)       // [5*S]
    }
}
