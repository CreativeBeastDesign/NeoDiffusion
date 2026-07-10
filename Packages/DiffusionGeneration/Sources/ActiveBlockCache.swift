import Foundation
import MLX
import DiffusionCore

/// The model-level wrapper for the active-block KV caches (WP-1a).
///
/// Under Phase 2, the active block's KV was recomputed every step.
/// Under Phase 3, this class holds a `LayerActiveCache` for each layer,
/// allowing selective reuse when logit/attention drift is low.
public final class ActiveBlockCache {
    public let layers: [LayerActiveCache]
    
    /// Track if we want to recompute every step (can be set to false for Elastic-Cache)
    public var recomputeEveryStep: Bool

    public init(layerCount: Int) {
        self.layers = (0 ..< layerCount).map { _ in LayerActiveCache() }
        self.recomputeEveryStep = true
    }

    public func clear() {
        for layer in layers {
            layer.clear()
        }
    }
}
