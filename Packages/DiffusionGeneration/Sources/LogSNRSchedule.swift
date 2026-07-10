import Foundation

/// Log-SNR schedule for the Sumi ancestral sampler, matching `_make_log_snr_schedule`
/// (generation_sumi.py) exactly: the schedule is built in **alpha space** and converted,
/// not sampled in log-SNR space directly.
///
///   linear:  α = linspace(1e-4, 1 − 1e-4, steps + 1)
///   cosine:  t = linspace(1 − 1e-3, 1e-3, steps + 1); α = 0.5 + 0.5·cos(t·π)
///   then     α clamped to [1e-6, 1 − 1e-6];  logSNR = log(α) − log1p(−α),
///            clamped to [minLogSNR, maxLogSNR]  (defaults ±9).
///
/// Values ascend noisy → clean; entry `steps` is the terminal (clean) level. Only the
/// ancestral sampler consumes it — adaptive/greedy ignore the schedule entirely.
public struct LogSNRSchedule: Equatable, Sendable {
    public enum Kind: String, Sendable {
        case linear
        case cosine
    }

    public let kind: Kind
    public let minLogSNR: Float
    public let maxLogSNR: Float

    public init(kind: Kind = .linear, minLogSNR: Float = -9.0, maxLogSNR: Float = 9.0) {
        self.kind = kind
        self.minLogSNR = minLogSNR
        self.maxLogSNR = maxLogSNR
    }

    /// `steps + 1` log-SNR values, ascending (noisy → clean).
    ///
    /// Arithmetic note: torch runs the elementwise chain (cos, clamp, log, log1p) in FP32;
    /// linspace itself is effectively double-precision. Mirror that split exactly — doing the
    /// whole chain in Double gives *more* accurate values that differ from the reference by
    /// ~1e-4 near the clamp edges and fail the 1e-5 parity gate.
    public func values(steps: Int) -> [Float] {
        precondition(steps >= 1, "num_denoising_steps must be a positive integer")
        let n = steps + 1

        func linspace(_ start: Double, _ end: Double, _ i: Int) -> Double {
            start + (end - start) * Double(i) / Double(steps)
        }

        var alphas = [Float](repeating: 0, count: n)
        switch kind {
        case .linear:
            for i in 0 ..< n {
                alphas[i] = Float(linspace(1e-4, 1.0 - 1e-4, i))
            }
        case .cosine:
            for i in 0 ..< n {
                let t = Float(linspace(1.0 - 1e-3, 1e-3, i))
                alphas[i] = 0.5 + 0.5 * Foundation.cosf(t * Float(Double.pi))
            }
        }
        return alphas.map { alpha in
            let clamped = Swift.min(Swift.max(alpha, 1e-6), 1.0 - 1e-6)
            let logSNR = Foundation.logf(clamped) - Foundation.log1pf(-clamped)
            return Swift.min(Swift.max(logSNR, minLogSNR), maxLogSNR)
        }
    }
}
