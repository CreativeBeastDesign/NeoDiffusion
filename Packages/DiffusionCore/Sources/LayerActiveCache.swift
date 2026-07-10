import Foundation
import MLX

/// Per-layer active-block cache for Elastic-Cache (WP-1a).
///
/// Stores the active block's keys and values from the previous step,
/// as well as the attention vector of the most-attended token to calculate drift.
public final class LayerActiveCache {
    /// Active block keys from previous step `[1, nKV, B, D]`
    public var keys: MLXArray?
    
    /// Active block values from previous step `[1, nKV, B, D]`
    public var values: MLXArray?
    
    /// Attention vector of the most-attended token from previous step `[numHeads, B]`
    public var previousAttentionVector: MLXArray?
    
    /// Index of the most-attended token from previous step
    public var previousMostAttendedIndex: Int?

    /// Computed attention similarity drift from the previous step (σ_t^ℓ) as a lazy MLXArray
    public var lastDriftSimilarity: MLXArray?

    public init() {}

    public func clear() {
        self.keys = nil
        self.values = nil
        self.previousAttentionVector = nil
        self.previousMostAttendedIndex = nil
        self.lastDriftSimilarity = nil
    }
}
