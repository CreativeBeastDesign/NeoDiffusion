import Foundation
import MLX

/// Per-layer frozen-token K/V hold for **faithful JOT** (WP-3a v2).
///
/// The v1 JOT implementation (`LLaDA2SparseMoEBlock`'s active-only dispatch) *only* zeroed the
/// MoE output of frozen tokens while continuing to recompute their attention every step. That
/// makes a frozen token's hidden state — and therefore its K/V — drift step-to-step, and through
/// bidirectional attention that drift perturbs its non-frozen neighbours (the "representation
/// perturbation cascade" recorded in `Plans/jot-logbook.md`).
///
/// The source paper (`Resources/possibly new/just-on-time-jot.md`) does not zero anything: it
/// *finalizes* a converged token and reuses its **stable KV** ("finalized tokens have stable KV
/// (easy to cache)… once token is converged, cache aggressively"). Faithful JOT therefore holds a
/// frozen column's post-qk-norm/post-RoPE keys and values at their pre-freeze value, so every
/// other position attends to a representation *identical* to the last full-compute step. This
/// class is that hold, one instance per decoder layer.
///
/// Rollback note: the hold mutates in-graph once per denoising step and is **not** snapshotted /
/// rolled back across a `K > 1` speculative batch, so faithful JOT is gated to `speculationK == 1`
/// (`DiffusionEngine.generateCached`). K=1 measures the (K-invariant) logical trajectory, which is
/// exactly the algorithmic quantity the WP-3a verdict turns on.
public final class LayerJotCache {
    /// Held active-window keys `[1, nKV, A, D]` (frozen columns pinned, active columns refreshed).
    public var keys: MLXArray?
    /// Held active-window values `[1, nKV, A, D]`.
    public var values: MLXArray?

    public init() {}

    public func clear() {
        keys = nil
        values = nil
    }
}

/// Model-level wrapper holding one ``LayerJotCache`` per decoder layer (faithful JOT, WP-3a v2).
/// Cleared at every block commit — each new block's active window starts with no frozen columns.
public final class JotFreezeCache {
    public let layers: [LayerJotCache]

    public init(layerCount: Int) {
        self.layers = (0 ..< layerCount).map { _ in LayerJotCache() }
    }

    public func clear() {
        for layer in layers { layer.clear() }
    }
}
