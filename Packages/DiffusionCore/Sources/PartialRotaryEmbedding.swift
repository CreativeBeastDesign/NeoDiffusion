import Foundation
import MLX

/// Partial rotary position embedding (phase-2 §2.3): rotary on the first
/// `headDim * partialRotaryFactor` dims of each head, passthrough on the rest.
///
/// Frequencies are computed in FP32 (`1 / θ^(2i/dim)`); cos/sin are produced in FP32
/// and cast to the activation dtype by the caller / `apply` (matches the reference,
/// which computes under `autocast(enabled=False)` and casts to `x.dtype` after).
///
/// Deliberately a struct, not a `Module`: it has no loadable weights and must not
/// appear in the parameter tree (`inv_freq` is a non-persistent buffer in the reference).
public struct PartialRotaryEmbedding {
    public let rotaryDim: Int
    public let invFreq: MLXArray  // FP32 [rotaryDim / 2]

    public init(headDim: Int, partialRotaryFactor: Float = 0.5, ropeTheta: Float) {
        self.rotaryDim = Int(Float(headDim) * partialRotaryFactor)
        let exponents = MLXArray(stride(from: 0, to: rotaryDim, by: 2).map { Float($0) })
            / Float(rotaryDim)
        self.invFreq = 1.0 / pow(MLXArray(ropeTheta), exponents)
    }

    /// cos/sin tables in FP32, shape `[B, L, rotaryDim]`, for absolute `positionIds` `[B, L]`.
    public func cosSin(positionIds: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        // freqs[b, l, i] = position[b, l] * invFreq[i]
        let freqs = positionIds.asType(.float32).expandedDimensions(axis: -1) * invFreq
        let emb = concatenated([freqs, freqs], axis: -1)
        return (cos(emb), sin(emb))
    }

    /// Applies rotary embedding to the first `cos.dim(-1)` dims of `x` `[B, H, L, D]`
    /// (llama-style rotate-half on the rotary slice, passthrough on the rest).
    /// Full rotary (`rotaryDim == headDim`, e.g. Sumi) takes the no-passthrough fast path.
    public static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let rotaryDim = cos.dim(-1)
        let fullRotary = rotaryDim == x.dim(-1)
        // [B, L, rot] -> [B, 1, L, rot] to broadcast over heads; cast to activation dtype
        let cosE = cos.expandedDimensions(axis: 1).asType(x.dtype)
        let sinE = sin.expandedDimensions(axis: 1).asType(x.dtype)

        let xRot = fullRotary ? x : x[.ellipsis, ..<rotaryDim]

        let half = rotaryDim / 2
        let x1 = xRot[.ellipsis, ..<half]
        let x2 = xRot[.ellipsis, half...]
        let rotated = concatenated([-x2, x1], axis: -1)

        let xEmbed = (xRot * cosE) + (rotated * sinE)
        if fullRotary { return xEmbed }
        let xPass = x[.ellipsis, rotaryDim...]
        return concatenated([xEmbed, xPass], axis: -1)
    }
}
