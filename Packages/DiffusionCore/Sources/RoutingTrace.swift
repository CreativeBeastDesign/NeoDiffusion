import Foundation
import MLX

/// **Diagnostic only — off in serving, and it forces a GPU→CPU readback per MoE layer per step.**
/// A run with this enabled has meaningless wall-clock; only the *distribution* it records is valid.
///
/// ## Why this exists
///
/// The `gather_qmm` roofline in `Plans/gather_qmm_handoff.md` §5.7 hinges on one unmeasured
/// quantity: **how many distinct experts a forward actually touches**. Everything downstream moves
/// with it —
///
/// | model of routing | distinct/layer | bytes/forward | achieved BW | kernel ceiling |
/// |---|---|---|---|---|
/// | uniform over 256 | ~162 | 5.45 GB | 58% of peak | 17.9% |
/// | concentrated (4 of 8 groups) | ~111 | 3.73 GB | 40% of peak | ~26% |
///
/// Both are *models*, not measurements. This records the truth. It also tests the
/// noise-aware-routing hypothesis (routing varies by denoising phase) for free, since each record
/// is tagged with its block/step/mask-ratio.
///
/// Records arrive **in layer order, one per MoE layer per forward** (19 for llada2.1-mini — layer 0
/// is dense and has no gate). Drain once per step and the array index is the layer index.
public final class RoutingTrace: @unchecked Sendable {
    private let lock = NSLock()
    private var pending: [[Int32]] = []

    public init() {}

    /// Append one gate's expert selection. `indices` is `[T, topK]`.
    public func record(_ indices: MLXArray) {
        let flat = indices.asArray(Int32.self)  // forces a sync — diagnostic only
        lock.lock()
        pending.append(flat)
        lock.unlock()
    }

    /// Everything recorded since the last drain, in layer order.
    public func drain() -> [[Int32]] {
        lock.lock()
        defer { pending.removeAll(); lock.unlock() }
        return pending
    }
}

/// Set to a `RoutingTrace` to enable collection. `nil` (the default) costs nothing — the gate's
/// check is a single nil test, and no readback happens.
nonisolated(unsafe) public var moeRoutingTrace: RoutingTrace? = nil
