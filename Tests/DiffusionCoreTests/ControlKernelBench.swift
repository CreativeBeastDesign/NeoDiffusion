import XCTest
import MLX
import MLXRandom
@testable import DiffusionCore

/// Task T1.5 of the GPU-control campaign (`Plans/gpu-control-logbook.md` §3): the standalone
/// "stock arm" microbench that T7 will later compare a fused control kernel against. This file
/// does NOT call `DiffusionEngine`'s private `stockSelectionUpdate` (Swift `private` methods in
/// an extension are file-scoped and unreachable from here) — it independently reconstructs the
/// **plain-Γ path only** (S=1 slot, no ICE/JOT/credit/S2D2, dynamic-τ α=0) of that op chain on
/// synthetic `[1,32]`-shaped inputs, mirroring `DiffusionEngine+Step.swift`'s
/// `stockSelectionUpdate` (see that file, roughly lines 469-710) op-for-op:
/// activeMask → nextPosts/budgetBreak → maskConf/threshold+top-1-fallback (Γ) → Δ →
/// finalTransfer → nextWindow → settleBreak/resultWindow → activationFlag → stats.
///
/// Values are irrelevant (random seed, arbitrary thresholds) — this is a wall-clock attribution
/// microbench only, never a correctness check (correctness of the real chain is T3/T4/T6's job).
///
/// Opt-in (env-gated per the task's ask, though the op chain is cheap and CPU-issue-bound so
/// this runs in well under a second): `NEODIFFUSION_CONTROL_BENCH=1 swift test --filter
/// ControlKernelBench`.
final class ControlKernelBench: XCTestCase {

    static let A = 32   // active window width (block length B, S=1 so A == B)

    private func skipUnlessOptedIn() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_CONTROL_BENCH"] == "1",
            "opt-in: set NEODIFFUSION_CONTROL_BENCH=1")
    }

    /// Wall-clock a closure: `warmup` un-timed reps, then `reps` timed (each fully eval'd).
    /// Mirrors `LLaDAMoEDispatchBench.swift:37-46`'s pattern exactly (repeated-median-adjacent:
    /// mean of `reps` timed calls after `warmup` untimed ones).
    private func time(_ label: String, warmup: Int = 5, reps: Int = 200,
                       _ body: () -> MLXArray) -> Double {
        for _ in 0 ..< warmup { eval(body()) }
        let start = Date()
        for _ in 0 ..< reps { eval(body()) }
        let mean = Date().timeIntervalSince(start) / Double(reps)
        print(String(format: "[control-bench] %@: %.6f s/iter (%.1f µs)",
                     label, mean, mean * 1_000_000))
        return mean
    }

    /// The stock plain-Γ selection segment, reconstructed from
    /// `DiffusionEngine+Step.swift`'s `stockSelectionUpdate`, specialized to S=1 (single slot,
    /// so `slotLen == A`), no ICE (`iceEnabled=false`), no JOT (`jotEnabled=false`, so
    /// `frozenMask` is all-false and `newlyFrozen` is always false — dropped from
    /// `finalTransfer`), no credit decoding (`unmaskConf == x0p`, `unmaskTok == x0`, so
    /// `gammaWriteTok == x0`), no S2D2 verifier, dynamic-τ α=0 (static `tauMask`), no EOS
    /// early-exit fill (`eosEarlyExit=false`, identity), `hasNextBlock=true` (activationFlag
    /// takes the live branch). Returns the same discrete/float outputs the real chain does
    /// (`nextWindow`, `resultWindow`, `breakFlag`, `nextPosts`, `activationFlag`, `stats`)
    /// concatenated into one array so the whole segment is timed by one `eval`.
    private func stockPlainSegment(
        windowActive: MLXArray, x0: MLXArray, x0p: MLXArray,
        promptMasks: MLXArray, post: MLXArray,
        maskId: Int32, eosId: Int32,
        tauMaskScalar: Float, tauEditScalar: Float, tauAddScalar: Float, maxPostSteps: Int32
    ) -> MLXArray {
        let A = Self.A
        let negInf = MLXArray(-Float.infinity)
        let positionsB = MLXArray(0 ..< Int32(A)).reshaped(1, A)

        // activeMask + per-slot post counter + budget flag (S=1: slotMaskAny == anyMaskFront).
        let activeMask = windowActive .== maskId                      // [1, A] Bool
        let anyMaskFront = activeMask.any()
        let nextPost = post + which(anyMaskFront, MLXArray(Int32(0)), MLXArray(Int32(1)))
        let budgetBreak = nextPost .> MLXArray(maxPostSteps)           // scalar Bool

        // Γ: threshold + top-1 fallback (front slot always has the unconditional fallback at
        // S=1 — no preceding-block gate to evaluate).
        let maskConf = which(activeMask, x0p, negInf)                  // [1, A]
        let highConf = (maskConf .> MLXArray(tauMaskScalar)) .&& activeMask
        let numHigh = highConf.asType(.int32).sum()
        let forceTop1 = numHigh .== MLXArray(Int32(0))
        let top1 = argMax(maskConf, axis: -1)
        let oneHotTop1 = positionsB .== top1.reshaped(1, 1)
        let gamma = highConf .|| (forceTop1 .&& oneHotTop1 .&& activeMask)
        let writeTok = x0                                              // credit off ⇒ identity

        // Δ: unmasked, non-prompt positions clearing τ_edit whose prediction changed.
        // frozenMask is all-false (JOT disabled), so `editable` drops that term.
        let editable = (.!activeMask) .&& (.!promptMasks)
        let editConf = which(editable, x0p, negInf)
        let highConfEdit = (editConf .> MLXArray(tauEditScalar)) .&& editable
        let tokenChanged = x0 .!= windowActive
        let delta = highConfEdit .&& tokenChanged
        let deltaAnyFront = delta.any()

        // finalTransfer (no JOT newlyFrozen term) → nextWindow. This is the real chain's
        // selection-boundary `eval` point (`selectionSeconds` in `stockSelectionUpdate`).
        let finalTransfer = gamma .|| delta
        let nextWindow = which(finalTransfer, writeTok, windowActive)

        // settleBreak/resultWindow (no eosEarlyExitFill — identity when disabled).
        let settleBreak = (.!anyMaskFront) .&& (.!deltaAnyFront)
        let breakFlag = (budgetBreak .|| settleBreak).reshaped([1])
        let resultWindow = which(budgetBreak, windowActive, nextWindow)

        // τ_add activation flag (S==1 && hasNextBlock ⇒ live branch; eos_early_stop suppression
        // included since the served Q-mode default has it on).
        let nextMaskCount = (nextWindow .== maskId).asType(.float32).sum()
        let genCount = Float(A)   // no prompt positions in this slot for the synthetic bench
        let decodedFrac = (MLXArray(genCount) - nextMaskCount) / MLXArray(genCount)
        var activation = decodedFrac .> MLXArray(tauAddScalar)
        let eosInFlight = (nextWindow .== MLXArray(eosId)).any()
        activation = activation .&& (.!eosInFlight)
        let activationFlag = activation.reshaped([1])

        // stats: [Γ|, |Δ|, meanConf, masksRemaining, specAccepted(=0)] — S=1 ⇒ 5 floats.
        let written = (gamma .|| delta).asType(.float32)
        let writtenCount = written.sum()
        let meanConf = (x0p.asType(.float32) * written).sum()
            / MLX.maximum(writtenCount, MLXArray(Float(1)))
        let masksRemaining = (nextWindow .== maskId).asType(.float32).sum()
        let stats = concatenated([
            gamma.asType(.float32).sum().reshaped([1]),
            delta.asType(.float32).sum().reshaped([1]),
            meanConf.reshaped([1]),
            masksRemaining.reshaped([1]),
            MLXArray(Float(0)).reshaped([1]),
        ], axis: 0)

        // Concatenate every output into one array so one `eval()` forces the whole segment,
        // matching how the real chain's timers force `nextWindow` (and the caller forces the
        // rest via the `[2K]` flags readback + block-commit readback). All flattened to 1-D
        // ([1,A] -> [A]) since `concatenated` requires matching rank across inputs.
        return concatenated([
            nextWindow.asType(.float32).reshaped([A]),
            resultWindow.asType(.float32).reshaped([A]),
            breakFlag.asType(.float32),
            nextPost.asType(.float32).reshaped([1]),
            activationFlag.asType(.float32),
            stats,
        ], axis: 0)
    }

    func testStockPlainSegmentMicrobench() throws {
        try skipUnlessOptedIn()
        let A = Self.A
        MLXRandom.seed(42)

        // Synthetic [1,32] inputs: a realistic partially-decoded block — some masked positions,
        // some already-written positions eligible for Δ, plausible confidence values.
        let maskId: Int32 = 156895
        let eosId: Int32 = 156892
        let isMasked = MLXRandom.uniform(0 ..< 1, [1, A]) .< MLXArray(Float(0.4))
        let randomTok = MLXRandom.randInt(0 ..< Int32(157184), [1, A])
        let windowActive = which(isMasked, MLXArray(maskId), randomTok).asType(.int32)
        let x0 = MLXRandom.randInt(0 ..< Int32(157184), [1, A])
        let x0p = MLXRandom.uniform(0.3 ..< 1.0, [1, A]).asType(.float32)
        let promptMasks = MLXArray([Bool](repeating: false, count: A)).reshaped(1, A)
        let post = MLXArray(Int32(3)).reshaped([1])
        eval(windowActive, x0, x0p, promptMasks, post)

        let mean = time("stock plain-Γ segment [1,32]") {
            self.stockPlainSegment(
                windowActive: windowActive, x0: x0, x0p: x0p, promptMasks: promptMasks,
                post: post, maskId: maskId, eosId: eosId,
                tauMaskScalar: 0.7, tauEditScalar: 0.5, tauAddScalar: 2.0, maxPostSteps: 16)
        }

        print(String(format: "[control-bench] stock plain-Γ segment: %.2f µs/iter",
                     mean * 1_000_000))
        // Sanity: cheap microbench, not a hang. No correctness assertion — this is a wall-clock
        // attribution bench (see file doc comment); T3/T4/T6 own correctness.
        XCTAssertLessThan(mean, 0.05, "stock segment unexpectedly slow (>50ms/iter) — investigate before citing as the T7 baseline")
    }
}
