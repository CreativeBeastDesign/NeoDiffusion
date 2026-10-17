import Foundation

/// Static Layer Classification for Compressed Sparse Attention 2 (DeepSeek-V4.1-Flash CSA2, 2026).
///
/// **Why CSA2 vs WP-1a Elastic-Cache**:
/// - WP-1a failed because runtime attention-column drift checking:
///   1. Paid ~20 CPU-GPU blocking readbacks per step, breaking graph pipelining.
///   2. Reused deep semantic layers, causing confidence starvation and doubling step counts.
///   3. Added full-attention score matrix materialization that cost more than QKV savings.
/// - CSA2 classifies layers **statically** into Full and Reuse modes:
///   1. Zero runtime drift testing. Zero GPU-CPU sync points.
///   2. Step 0 (block entry) and commit captures always run Full mode.
///   3. Deep confidence-driving layers remain Full mode; shallow/middle lexical layers can reuse representations.
public struct CSA2Config: Sendable, Equatable {
    public enum LayerMode: Sendable, Equatable {
        /// All layers recompute on every step (standard baseline).
        case none
        /// Layers below boundary reuse active KV during refinement steps; layers >= boundary recompute.
        case staticBoundary(boundary: Int)
        /// Alternating layers: layers with `(index % stride != 0)` reuse; stride-aligned layers recompute.
        case alternating(stride: Int)
    }
    
    public var mode: LayerMode
    public var enabled: Bool
    
    public init(mode: LayerMode = .none, enabled: Bool = false) {
        self.mode = mode
        self.enabled = enabled
    }

    public static func boundary(_ boundary: Int) -> CSA2Config {
        CSA2Config(mode: .staticBoundary(boundary: boundary), enabled: true)
    }
    
    public static func alternating(stride: Int) -> CSA2Config {
        CSA2Config(mode: .alternating(stride: stride), enabled: true)
    }

    
    /// Determines whether the given layer should reuse active KV states on `stepInBlock`.
    ///
    /// - Parameters:
    ///   - layerIndex: Index of the transformer layer (0 ..< numLayers).
    ///   - stepInBlock: Logical step within the active block (0 = first step).
    ///   - totalLayers: Total number of decoder layers.
    /// - Returns: `true` if active KV should be reused; `false` if freshly recomputed.
    public func shouldReuse(layerIndex: Int, stepInBlock: Int, totalLayers: Int) -> Bool {
        guard enabled else { return false }
        // Step 0 of a block ALWAYS recomputes all layers to initialize the block representation cleanly.
        if stepInBlock == 0 { return false }
        
        switch mode {
        case .none:
            return false
        case .staticBoundary(let boundary):
            // Layers < boundary reuse; deeper layers (which drive logits and edits) always recompute
            return layerIndex < boundary
        case .alternating(let stride):
            // Keep stride-aligned layers fresh; reuse intermediate layers
            return (layerIndex % stride != 0)
        }
    }
}
