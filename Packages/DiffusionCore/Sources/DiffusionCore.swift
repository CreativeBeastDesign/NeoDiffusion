import Foundation
import MLX
import Numerics
import DiffusionKernels

/// DiffusionCore: Core neural blocks (attention, embeddings, MoE routing/dispatch, layers).
public struct DiffusionCore {
    public static let description = "Core neural network blocks and MoE layers"
    
    public init() {}
    
    /// Basic attention routing block template using MLX Arrays
    public func attentionBlock(query: MLXArray, key: MLXArray, value: MLXArray) -> MLXArray {
        // Standard dot product attention wrapper using MLX
        let scale = MLXArray(1.0 / Double.sqrt(Double(query.dim(-1))))
        let scores = matmul(query * scale, key.transposed(axes: [0, 2, 1]))
        let weights = softmax(scores, axis: -1)
        return matmul(weights, value)
    }
}
