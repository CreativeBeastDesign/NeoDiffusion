import Foundation
import MLX
import MLXRandom
import DiffusionModel

/// Generation request for the Sumi uniform-diffusion engine, mirroring
/// `SumiGenerationMixin.generate`'s resolved parameters (generation_sumi.py).
public struct SumiGenerationRequest {
    public enum Sampler: String, Sendable {
        case ancestral
        case adaptive
        case greedy
    }

    public var promptIds: [Int32]
    public var maxNewTokens: Int
    /// The model is trained on a packed fixed-length canvas: generation always runs on a
    /// full `canvasLength` canvas (clamped to `max_position_embeddings`), not
    /// prompt+maxNewTokens. `maxNewTokens` is the content budget only.
    public var canvasLength: Int
    public var numDenoisingSteps: Int
    public var sampler: Sampler
    public var schedule: LogSNRSchedule
    public var temperature: Float
    public var tokensPerStep: Int
    /// Explicit pinned anchors. `nil` + `anchorEOSBOS` → the default `[EOS, BOS]` delimiter
    /// at `promptLength + budget` (an explicit value takes precedence, matching the reference).
    public var frozen: [FrozenAnchor]?
    public var denoiseEnd: Int?
    public var anchorEOSBOS: Bool
    public var trimAtEOS: Bool
    public var seed: UInt64
    /// Adaptive only. When set, a position is frozen after its first commit — no revision
    /// passes. Motivated by the Sumi paper's revision-budget finding (extra passes beyond
    /// first commit change ≤1% of tokens, mostly A→B→A round trips, and never improve
    /// accuracy), and makes `steps = ceil(window / tokensPerStep)` exact full coverage.
    /// **Deviation from the reference sampler** — off by default; gated by the
    /// SumiLeverExperiments suite, not by parity tests.
    public var freezeCommitted: Bool

    /// SchED-style early exit (S4.2, arXiv:2512.02892 adapted to uniform-state): stop when
    /// the argmax predictions over the active window have been **unchanged for
    /// `stableSteps` consecutive steps** (and at least `minSteps` ran). `numDenoisingSteps`
    /// becomes a cap. On exit the active window takes a final greedy commit of the stable
    /// argmax — required for ancestral, whose sampled canvas still carries posterior noise
    /// mid-schedule; a no-op for greedy. **Deviation** — off by default (`nil`), quality
    /// gated by bench arms, not parity tests. Adds one tiny (`Bool`) readback per step;
    /// not composed with `pipelined` (the flag read would stall the pipeline — cap-only
    /// runs there).
    public struct EarlyExit {
        public var stableSteps: Int
        public var minSteps: Int
        /// A step counts as "stable" when at least this fraction of active-window
        /// predictions is unchanged. 1.0 = exact stability — measured to **never fire** on
        /// real weights (Sumi keeps flipping 1–4 tokens indefinitely, matching the
        /// revision-churn finding); ~0.9–0.95 is the practical setting.
        public var stableFraction: Float

        public init(stableSteps: Int = 2, minSteps: Int = 0, stableFraction: Float = 1.0) {
            self.stableSteps = stableSteps
            self.minSteps = minSteps
            self.stableFraction = stableFraction
        }
    }

    public var earlyExit: EarlyExit?

    public init(
        promptIds: [Int32],
        maxNewTokens: Int,
        canvasLength: Int = 2048,
        numDenoisingSteps: Int = 128,
        sampler: Sampler = .ancestral,
        schedule: LogSNRSchedule = LogSNRSchedule(),
        temperature: Float = 1.0,
        tokensPerStep: Int = 1,
        frozen: [FrozenAnchor]? = nil,
        denoiseEnd: Int? = nil,
        anchorEOSBOS: Bool = true,
        trimAtEOS: Bool = true,
        seed: UInt64 = 0,
        freezeCommitted: Bool = false,
        earlyExit: EarlyExit? = nil
    ) {
        self.promptIds = promptIds
        self.maxNewTokens = maxNewTokens
        self.canvasLength = canvasLength
        self.numDenoisingSteps = numDenoisingSteps
        self.sampler = sampler
        self.schedule = schedule
        self.temperature = temperature
        self.tokensPerStep = tokensPerStep
        self.frozen = frozen
        self.denoiseEnd = denoiseEnd
        self.anchorEOSBOS = anchorEOSBOS
        self.trimAtEOS = trimAtEOS
        self.seed = seed
        self.freezeCommitted = freezeCommitted
        self.earlyExit = earlyExit
    }
}

/// The Sumi denoising engine (sumi-plan.md §3 S1.5): fixed-step uniform-state loop over a
/// full canvas, fully bidirectional attention every step, no KV cache
/// (`disableCacheDuringDenoise` — decision S0.1-3; the reference passes `use_cache=False`
/// on every forward).
///
/// Loop-control note: unlike the LLaDA draft-and-edit loop there is **no break condition** —
/// the step count is fixed, so no speculative execution or per-step readback is needed. The
/// only blocking readback is the final EOS trim.
public final class SumiEngine {
    public struct Output {
        /// Full untrimmed denoised canvas `[1, canvasLength]`.
        public let canvas: MLXArray
        /// Prompt + generation, cut at the first EOS in the generated region
        /// (EOS/anchor/denoised tail dropped) when `trimAtEOS` is set.
        public let sequences: [Int32]
        /// Denoising steps actually executed (< `numDenoisingSteps` on early exit).
        public let stepsExecuted: Int
    }

    public let model: SumiModel

    public init(model: SumiModel) {
        self.model = model
    }

    /// Runs the denoising loop.
    ///
    /// - Parameters:
    ///   - request: resolved generation parameters.
    ///   - initialCanvas: optional full canvas override `[1, canvasLength]` (prompt included,
    ///     anchors not yet written). Used by parity tests — `torch.randint` cannot be
    ///     reproduced across RNGs, so trace fixtures ship the reference's initial canvas.
    ///   - pipelined: use `asyncEval` instead of a blocking `eval` per step, letting the
    ///     CPU build step t+1's graph while the GPU runs step t (S3.2 occupancy A/B).
    ///     Per-step `onStep` readbacks would defeat it — prefer `onStep: nil` here.
    ///   - onStep: called after each denoising step with `(stepIndex, canvas)`; canvases are
    ///     handed out for trace diffing, not mutated afterwards.
    public func generate(
        _ request: SumiGenerationRequest,
        initialCanvas: MLXArray? = nil,
        pipelined: Bool = false,
        onStep: ((Int, MLXArray) -> Void)? = nil
    ) -> Output {
        let config = model.config
        let promptLength = request.promptIds.count
        precondition(promptLength > 0, "prompt must be non-empty (bos-only at minimum)")
        precondition(request.maxNewTokens > 0, "maxNewTokens must be positive")
        precondition(request.numDenoisingSteps >= 1, "numDenoisingSteps must be positive")

        // Canvas geometry (reference: generate's budget/anchor resolution).
        let canvasLength = min(request.canvasLength, config.maxPositionEmbeddings)
        let reserve = request.anchorEOSBOS ? 2 : 0
        precondition(
            promptLength + reserve < canvasLength,
            "prompt (\(promptLength)) leaves no room in canvas (\(canvasLength))")
        let budget = max(1, min(request.maxNewTokens, canvasLength - promptLength - reserve))

        // Anchors: explicit `frozen` takes precedence over the default EOS,BOS delimiter.
        let anchors: [FrozenAnchor]
        if let frozen = request.frozen {
            anchors = frozen
        } else if request.anchorEOSBOS {
            let eosPos = promptLength + budget
            anchors = [
                FrozenAnchor(position: eosPos, tokenId: Int32(config.eosTokenId)),
                FrozenAnchor(position: eosPos + 1, tokenId: Int32(config.bosTokenId)),
            ]
        } else {
            anchors = []
        }

        // Canvas init: prompt + uniform-random completion (the uniform-state prior).
        var key = MLXRandom.key(request.seed)
        var z: MLXArray
        if let initialCanvas {
            precondition(
                initialCanvas.dim(-1) == canvasLength,
                "initialCanvas length \(initialCanvas.dim(-1)) != canvas \(canvasLength)")
            z = initialCanvas.asType(.int32).reshaped(1, canvasLength)
        } else {
            let (initKey, nextKey) = splitKey(key)
            key = nextKey
            let completion = MLXRandom.randInt(
                0 ..< Int32(config.vocabSize), [1, canvasLength - promptLength], key: initKey)
            let prompt = MLXArray(request.promptIds).reshaped(1, promptLength)
            z = concatenated([prompt, completion.asType(.int32)], axis: 1)
        }

        // Write anchor tokens into the canvas, then freeze them via the noise mask.
        for anchor in anchors {
            z[0..., anchor.position ..< (anchor.position + 1)] =
                MLXArray([anchor.tokenId]).reshaped(1, 1)
        }
        // `activeMask` shrinks under `freezeCommitted`; otherwise it stays the noise mask.
        var activeMask = SumiNoiseMask.build(
            totalLength: canvasLength,
            promptLength: promptLength,
            anchors: anchors,
            denoiseEnd: request.denoiseEnd)

        let logSNRs = request.sampler == .ancestral
            ? request.schedule.values(steps: request.numDenoisingSteps) : []

        // SchED-style early exit state (S4.2): argmax-prediction stability over the
        // active window, checked per step via one tiny Bool readback.
        var previousPred: MLXArray? = nil
        var stableCount = 0
        var stepsExecuted = 0

        for step in 0 ..< request.numDenoisingSteps {
            // FP32, vocab-truncated logits; mask=nil is the bidirectional generation path.
            let logits = model.logits(forTokens: z)

            // Early-exit signal must come from the same forward the step consumes.
            var currentPred: MLXArray? = nil
            if request.earlyExit != nil {
                currentPred = logits.argMax(axis: -1).asType(.int32)
            }

            switch request.sampler {
            case .greedy:
                z = UniformStateSampler.greedyStep(z: z, logits: logits, noiseMask: activeMask)
            case .adaptive:
                let (stepKey, nextKey) = splitKey(key)
                key = nextKey
                let (zNew, selected) = UniformStateSampler.adaptiveStep(
                    z: z, logits: logits, noiseMask: activeMask,
                    tokensPerStep: request.tokensPerStep,
                    temperature: request.temperature, key: stepKey)
                z = zNew
                if request.freezeCommitted {
                    let iota = MLXArray(0 ..< Int32(canvasLength))
                    let selMask = (selected.expandedDimensions(axis: -1) .== iota)
                        .sum(axis: -2) .> 0
                    activeMask = activeMask .&& .!selMask
                }
            case .ancestral:
                var xHat = logits
                if request.temperature != 1.0 {
                    xHat = xHat / max(request.temperature, 1e-6)
                }
                let (stepKey, nextKey) = splitKey(key)
                key = nextKey
                let zNew = UniformStateSampler.ancestralStep(
                    z: z, xHat: softmax(xHat, axis: -1),
                    logSNRt: logSNRs[step], logSNRs: logSNRs[step + 1],
                    vocabSize: config.vocabSize, key: stepKey)
                z = which(activeMask, zNew, z)
            }

            if pipelined {
                asyncEval(z)
            } else {
                eval(z)
            }
            stepsExecuted = step + 1
            onStep?(step, z)

            // Early exit: predictions unchanged over the active window for `stableSteps`
            // consecutive steps → final greedy commit of the (stable) argmax and stop.
            // The commit is required for ancestral (its sampled canvas still carries
            // posterior noise mid-schedule) and a no-op for greedy at convergence.
            if let policy = request.earlyExit, let pred = currentPred, !pipelined {
                if let prev = previousPred {
                    let changed = ((pred .!= prev) .&& activeMask).sum().item(Int.self)
                    let activeN = activeMask.sum().item(Int.self)
                    let allowed = Int(((1 - policy.stableFraction) * Float(activeN)).rounded())
                    stableCount = changed <= allowed ? stableCount + 1 : 0
                }
                previousPred = pred
                if stableCount >= policy.stableSteps, step + 1 >= policy.minSteps {
                    z = which(activeMask, pred.asType(z.dtype), z)
                    eval(z)
                    break
                }
            }
        }

        let sequences = request.trimAtEOS
            ? Self.trimAtEOS(canvas: z, promptLength: promptLength, eosId: Int32(config.eosTokenId))
            : z.reshaped(-1).asArray(Int32.self)
        return Output(canvas: z, sequences: sequences, stepsExecuted: stepsExecuted)
    }

    /// `_trim_at_eos` (B = 1): cut at the first EOS in the generated region, dropping the
    /// EOS and everything after (anchor BOS, denoised tail). Prompt tokens are kept.
    static func trimAtEOS(canvas: MLXArray, promptLength: Int, eosId: Int32) -> [Int32] {
        let row = canvas.reshaped(-1).asArray(Int32.self)  // single final readback
        let generated = row[promptLength...]
        if let hit = generated.firstIndex(of: eosId) {
            return Array(row[..<hit])
        }
        return row
    }

    private func splitKey(_ key: MLXArray) -> (MLXArray, MLXArray) {
        let parts = MLXRandom.split(key: key, into: 2)
        return (parts[0], parts[1])
    }
}
