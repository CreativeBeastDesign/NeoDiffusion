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
    /// Temporal refresh stride across denoising steps within a block (`1` = reuse every step >= minStep;
    /// `2` = alternate refresh and reuse so active K/V is never >1 step stale and never stuck on Step-0 masks).
    public var stepStride: Int
    /// First step in block eligible for reuse (`1` = legacy; `2` = ensure Step 1 post-mask K/V is captured first).
    public var minStep: Int
    
    public init(mode: LayerMode = .none, enabled: Bool = false, stepStride: Int = 1, minStep: Int = 1) {
        self.mode = mode
        self.enabled = enabled
        self.stepStride = max(1, stepStride)
        self.minStep = max(1, minStep)
    }

    public static func boundary(_ boundary: Int, stepStride: Int = 1, minStep: Int = 1) -> CSA2Config {
        CSA2Config(mode: .staticBoundary(boundary: boundary), enabled: true, stepStride: stepStride, minStep: minStep)
    }
    
    public static func alternating(stride: Int, stepStride: Int = 1, minStep: Int = 1) -> CSA2Config {
        CSA2Config(mode: .alternating(stride: stride), enabled: true, stepStride: stepStride, minStep: minStep)
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
        // Steps before minStep (always including Step 0) recompute all layers to initialize post-mask representations.
        if stepInBlock < minStep { return false }
        // Periodic temporal refresh: since step (minStep - 1) recomputed, recompute every `stepStride` steps thereafter
        // (e.g., for minStep=2, stepStride=2: Steps 0,1 recompute; Step 2 reuses; Step 3 recomputes; Step 4 reuses).
        if stepStride > 1 && ((stepInBlock - minStep + 1) % stepStride == 0) {
            return false
        }
        
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
