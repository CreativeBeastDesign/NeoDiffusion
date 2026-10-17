import Foundation
import MLX

/// KV Cache Quantization mode for Inter-Block Prefix Storage (2026 Dual Caching Tier 1).
///
/// - `none`: Full-precision storage (FP16 / BF16).
/// - `fp8`: 8-bit quantized KV with per-group scaling (50% memory reduction, < 0.02 max deviation).
/// - `fp4`: 4-bit packed KV with per-group scaling (75% memory reduction).
public enum KVCacheQuantization: String, Sendable, Equatable, CaseIterable {
    case none
    case fp8
    case fp4
    
    public var bits: Int {
        switch self {
        case .none: return 16
        case .fp8: return 8
        case .fp4: return 4
        }
    }
}

/// Container for compressed (quantized) prefix key or value tensors.
public struct CompressedKVTensor {
    public let packed: MLXArray       // uint32 packed tensor
    public let scales: MLXArray       // per-group float scales
    public let biases: MLXArray?      // per-group float biases
    public let groupSize: Int
    public let bits: Int
    public let originalShape: [Int]
    
    public init(packed: MLXArray, scales: MLXArray, biases: MLXArray?, groupSize: Int, bits: Int, originalShape: [Int]) {
        self.packed = packed
        self.scales = scales
        self.biases = biases
        self.groupSize = groupSize
        self.bits = bits
        self.originalShape = originalShape
    }
    
    /// Compresses a full-precision tensor using fine-grained group quantization.
    public static func compress(_ x: MLXArray, mode: KVCacheQuantization, groupSize: Int = 64) -> CompressedKVTensor? {
        guard mode != .none else { return nil }
        let (q, scales, biases) = MLX.quantized(x, groupSize: groupSize, bits: mode.bits)
        return CompressedKVTensor(
            packed: q,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: mode.bits,
            originalShape: x.shape
        )
    }
    
    /// Dequantizes the compressed tensor back to full-precision FP16.
    public func dequantize(dtype: DType = .float16) -> MLXArray {
        return dequantized(
            packed,
            scales: scales,
            biases: biases,
            groupSize: groupSize,
            bits: bits,
            dtype: dtype
        )
    }
}
