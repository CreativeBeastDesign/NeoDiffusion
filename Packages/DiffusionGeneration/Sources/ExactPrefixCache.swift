import Foundation
import MLX
import DiffusionCore

/// The **exact** KV tier: prompt + committed-block K/V (phase-1 §5 "two caches, two contracts").
///
/// Exact by block-causality under the `.strict` mask (phase-2 §5 deviation 1) — a committed
/// block's K/V is what an uncached full-window forward would compute for those positions, so
/// reusing it is mathematically equivalent, not approximate. **Append-only, never
/// policy-managed**: Phase 3 staleness policies touch only ``ActiveBlockCache``. Kept a distinct
/// type from the active tier on purpose (phase-1 decision).
///
/// Holds one ``LayerKVCache`` per decoder layer. Growth is driven exclusively by the loop's
/// commit-cleanliness capture (a forward over the *final* committed tokens), so the stored K/V
/// are always consistent with the committed tokens (gotcha 6).
public final class ExactPrefixCache {
    public let layers: [LayerKVCache]

    public init(layerCount: Int) {
        self.layers = (0 ..< layerCount).map { _ in LayerKVCache() }
    }

    /// Committed token count (all layers advance in lockstep; layer 0 is representative).
    public var committedLength: Int { layers.first?.committedLength ?? 0 }

    /// Commit the pending (last-captured) active-block K/V on every layer — the block-commit
    /// transition. Callers must have run a capture forward over the final committed tokens first.
    public func commitBlock() {
        for layer in layers { layer.commitPending() }
    }
}
