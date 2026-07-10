import Foundation
import MLX
import MLXNN

/// RMSNorm matching the reference `LLaDA2MoeRMSNorm` exactly (phase-2 §2.2):
/// compute in FP32 internally, cast back to the input dtype, then scale by weight.
///
/// The operation order is load-bearing for parity: the normalized value is cast to the
/// activation dtype *before* the weight multiply (`weight * normed.to(input_dtype)`).
/// Do not replace with a fused variant that skips the FP32 upcast.
public final class LLaDA2RMSNorm: Module {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    public let eps: Float

    public init(dimensions: Int, eps: Float = 1e-6) {
        self._weight = ParameterInfo(wrappedValue: MLXArray.ones([dimensions]), key: "weight")
        self.eps = eps
        super.init()
    }

    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let inputDType = x.dtype
        let xf = x.asType(.float32)
        let variance = mean(square(xf), axis: -1, keepDims: true)
        let normed = xf * rsqrt(variance + eps)
        return weight * normed.asType(inputDType)
    }
}
