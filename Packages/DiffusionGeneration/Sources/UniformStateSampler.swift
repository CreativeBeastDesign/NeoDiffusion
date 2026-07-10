import Foundation
import MLX
import MLXRandom

/// The Sumi uniform-state sampler family (generation_sumi.py): three step functions over a
/// canvas `z [B, S]` of token ids and FP32 logits `[B, S, V]` (already vocab-truncated and
/// upcast by `SumiModel`). No mask token exists — every non-frozen position is always
/// eligible for change.
///
/// Parity notes (decision, sumi-plan.md §2 S0.1-1): greedy and adaptive (temperature 0) are
/// deterministic and gated token-for-token; ancestral is stochastic — `torch.multinomial`
/// cannot be reproduced across RNGs, so it is realised as **Gumbel-max over the analytic
/// posterior** (statistically equivalent categorical sampling; the guide's §4.4 form, which
/// also avoids materialising a normalised FP32 posterior) and gated only distributionally.
public enum UniformStateSampler {

    // MARK: - Greedy

    /// `_greedy_step`: overwrite every denoise position with the argmax prediction.
    public static func greedyStep(
        z: MLXArray, logits: MLXArray, noiseMask: MLXArray
    ) -> MLXArray {
        which(noiseMask, logits.argMax(axis: -1).asType(z.dtype), z)
    }

    // MARK: - Adaptive (von Rütte confidence commit)

    /// `_adaptive_step`: confidence = `p_max − p_curr` from the **un-tempered** softmax
    /// (the uniform prior 1/V drops out of the ranking); commit the top `tokensPerStep`
    /// positions to the tempered prediction (argmax at temperature 0); frozen positions
    /// guaranteed unchanged.
    ///
    /// The softmax is never materialised (S3.2 structural fix, André 2026-07-09):
    /// `p_v = exp(l_v − LSE)`, so `conf = exp(l_max − LSE) − exp(l_{z_t} − LSE)` needs only
    /// max/LSE/gather reductions over the `[S, V]` logits — no `[S, V]` intermediate, and
    /// no fp16 blurring of the decision margins the top-k ranking relies on. The tempered
    /// commit path likewise samples via Gumbel-max on `logits/T` directly
    /// (`argmax(l/T + g)` ≡ sampling from `softmax(l/T)`).
    ///
    /// - Returns: `(z_s [B, S], selectedPositions [B, k])`
    public static func adaptiveStep(
        z: MLXArray,
        logits: MLXArray,
        noiseMask: MLXArray,
        tokensPerStep: Int,
        temperature: Float,
        key: MLXArray? = nil
    ) -> (z: MLXArray, selectedPositions: MLXArray) {
        let S = z.dim(-1)

        // Selection confidence from reductions only (see doc comment).
        let lse = logits.logSumExp(axis: -1)                                  // [B, S]
        let lMax = logits.max(axis: -1)
        let lCurr = takeAlong(logits, z.asType(.int32).expandedDimensions(axis: -1), axis: -1)
            .squeezed(axis: -1)
        var conf = exp(lMax - lse) - exp(lCurr - lse)
        conf = which(noiseMask, conf, MLXArray(-Float.infinity))

        let k = max(1, min(tokensPerStep, S))
        // Descending argsort; first k are the top-k (reference torch.topk).
        let selected = argSort(-conf, axis: -1)[.ellipsis, ..<k]  // [B, k]

        // Commit distribution: tempered (temperature 0 → greedy argmax).
        let pred: MLXArray
        if temperature > 0 {
            let scaled = logits / max(temperature, 1e-6)
            let u = MLXRandom.uniform(
                low: Float.leastNormalMagnitude, high: 1.0, logits.shape,
                key: key ?? MLXRandom.key(0))
            pred = (scaled + (-log(-log(u)))).argMax(axis: -1).asType(z.dtype)
        } else {
            pred = logits.argMax(axis: -1).asType(z.dtype)
        }

        // Scatter pred into z at the selected positions: one-hot over S, no CPU readback.
        let iota = MLXArray(0 ..< Int32(S))                                   // [S]
        let selMask = (selected.expandedDimensions(axis: -1) .== iota)        // [B, k, S]
            .sum(axis: -2) .> 0                                               // [B, S]
        let zS = which(selMask, pred, z)
        return (which(noiseMask, zS, z), selected)
    }

    // MARK: - Ancestral (analytic posterior)

    /// The deterministic part of `_ancestral_step`: the unnormalised analytic posterior
    /// `p(z_s | z_t, x̂) ∝ q(z_s | x̂) · q(z_t | z_s)` under the uniform forward process.
    /// Split out so parity tests can gate it exactly (the subsequent sample is stochastic).
    public static func ancestralPosterior(
        z: MLXArray,
        xHat: MLXArray,
        logSNRt: Float,
        logSNRs: Float,
        vocabSize: Int,
        eps: Float = 1e-12
    ) -> MLXArray {
        let alphaT = 1.0 / (1.0 + exp(-logSNRt))
        let alphaS = 1.0 / (1.0 + exp(-logSNRs))
        let alphaTS = alphaT / max(alphaS, eps)
        let betaT = 1.0 - alphaT
        let betaS = 1.0 - alphaS
        let betaTS = max(1.0 - alphaTS, 0.0)

        let invV = 1.0 / Float(vocabSize)
        let uT = betaT * invV
        let uS = betaS * invV
        let uTS = betaTS * invV

        let qS = alphaS * xHat + uS                                            // [B, S, V]

        let iota = MLXArray(0 ..< Int32(vocabSize))
        let oneHotZt = (z.asType(.int32).expandedDimensions(axis: -1) .== iota)
            .asType(xHat.dtype)                                                // [B, S, V]
        let qTGivenS = alphaTS * oneHotZt + uTS

        let xHatAtZt = takeAlong(xHat, z.asType(.int32).expandedDimensions(axis: -1), axis: -1)
            .squeezed(axis: -1)                                                // [B, S]
        let qTAtZt = maximum(alphaT * xHatAtZt + uT, MLXArray(eps))

        let posterior = qS * qTGivenS / qTAtZt.expandedDimensions(axis: -1)
        return maximum(posterior, MLXArray(Float(0)))
    }

    /// One ancestral denoising step `z_t → z_s`: categorical sample from the analytic
    /// posterior via Gumbel-max, in **reduced form**.
    ///
    /// Derivation (**sourced** by algebra; exact-equivalence gated in
    /// `testAncestralReducedFormEquivalence`): with `w_v ∝ q_s(v) · q_{t|s}(v)` the
    /// per-position divisor `q_t(z_t)` is constant over `v` and drops out of the argmax;
    /// `q_{t|s}(v) = α_{t|s}·1[v=z_t] + u_{t|s}` contributes `log u_{t|s}` everywhere
    /// (another droppable constant) plus a scalar spike `log((α_{t|s}+u_{t|s})/u_{t|s})`
    /// at `v = z_t` only. So Gumbel-max over the full posterior reduces to
    ///   `best = argmax_v(log q_s(v) + g_v)` vs the single spiked candidate `z_t`,
    /// a two-candidate comparison. This avoids materialising the `[B, S, V]` one-hot,
    /// `q_{t|s}`, and the posterior division — the tensors that made the naive sampler
    /// cost ~15 s/step at canvas 1536 on real weights (measured 2026-07-09).
    public static func ancestralStep(
        z: MLXArray,
        xHat: MLXArray,
        logSNRt: Float,
        logSNRs: Float,
        vocabSize: Int,
        key: MLXArray,
        eps: Float = 1e-12
    ) -> MLXArray {
        let alphaT = 1.0 / (1.0 + exp(-logSNRt))
        let alphaS = 1.0 / (1.0 + exp(-logSNRs))
        let alphaTS = alphaT / max(alphaS, eps)
        let uS = (1.0 - alphaS) / Float(vocabSize)
        let uTS = max(1.0 - alphaTS, 0.0) / Float(vocabSize)
        // Spike at z_t from the q_{t|s} factor. uTS > 0 for any ascending schedule
        // (α_t < α_s ⇒ α_{t|s} < 1); the eps floor guards a degenerate flat schedule.
        let spike = Foundation.logf(alphaTS + max(uTS, eps)) - Foundation.logf(max(uTS, eps))

        let zIdx = z.asType(.int32).expandedDimensions(axis: -1)
        let logQS = log(maximum(alphaS * xHat + uS, MLXArray(eps)))          // [B, S, V]
        let u = MLXRandom.uniform(
            low: Float.leastNormalMagnitude, high: 1.0, xHat.shape, key: key)
        let total = logQS + (-log(-log(u)))                                   // [B, S, V]

        let best = total.argMax(axis: -1)                                     // [B, S]
        let bestVal = total.max(axis: -1)
        let ztVal = takeAlong(total, zIdx, axis: -1).squeezed(axis: -1) + spike
        return which(ztVal .>= bestVal, z, best.asType(z.dtype))
    }

    // MARK: - Gumbel-max categorical

    /// Samples an index along the last axis with probability ∝ `weights` (unnormalised is
    /// fine): `argmax(log(max(w, floor)) + Gumbel(0,1))`. Zero-weight entries get a large
    /// negative log — selectable only with vanishing probability (matches `torch.multinomial`
    /// semantics distributionally, not bit-for-bit).
    static func gumbelMaxSample(_ weights: MLXArray, key: MLXArray) -> MLXArray {
        let logW = log(maximum(weights, MLXArray(Float(1e-12))))
        let u = MLXRandom.uniform(
            low: Float.leastNormalMagnitude, high: 1.0, weights.shape, key: key)
        let gumbel = -log(-log(u))
        return (logW + gumbel).argMax(axis: -1)
    }
}
