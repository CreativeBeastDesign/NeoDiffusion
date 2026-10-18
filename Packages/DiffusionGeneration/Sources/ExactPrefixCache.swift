import Foundation
import MLX
import DiffusionCore

/// The **exact** KV tier: prompt + committed-block K/V (phase-1 §5 "two caches, two contracts")
/// enhanced with Tier-1 Preallocated Memory Pool and Low-Bit KV Quantization.
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
    public let usePool: Bool
    public let quantization: KVCacheQuantization

    public init(
        layerCount: Int,
        usePool: Bool = false,
        initialPoolCapacity: Int = 256,
        quantization: KVCacheQuantization = .none
    ) {
        self.usePool = usePool
        self.quantization = quantization
        self.layers = (0 ..< layerCount).map { l in
            let layerQuant: KVCacheQuantization
            if quantization == .k8v4s4 {
                layerQuant = (l < 4) ? .k8v4 : .fp4
            } else {
                layerQuant = quantization
            }
            return LayerKVCache(usePool: usePool, initialPoolCapacity: initialPoolCapacity, quantization: layerQuant)
        }
    }

    /// Committed token count (all layers advance in lockstep; layer 0 is representative).
    public var committedLength: Int { layers.first?.committedLength ?? 0 }

    /// Total logical bytes across all layers of the committed prefix KV cache.
    public var logicalBytes: Int { layers.reduce(0) { $0 + $1.logicalBytes } }

    /// Total allocated bytes across all layers of the committed prefix KV cache (including pool slack and FlashBlock accumulators).
    public var allocatedBytes: Int { layers.reduce(0) { $0 + $1.allocatedBytes } }

    /// Total FlashBlock cache refreshes (layer 0 is representative; all layers run the same step kind).
    public var flashBlockRefreshes: Int { layers.first?.flashBlockRefreshes ?? 0 }

    /// Total FlashBlock cache reuses (layer 0 is representative).
    public var flashBlockReuses: Int { layers.first?.flashBlockReuses ?? 0 }

    /// Commit the pending (last-captured) active-block K/V on every layer — the block-commit
    /// transition. Callers must have run a capture forward over the final committed tokens first.
    public func commitBlock(usePool: Bool? = nil, quantization: KVCacheQuantization? = nil) {
        let poolFlag = usePool ?? self.usePool
        for layer in layers {
            let quantMode = quantization ?? layer.quantizationMode
            layer.commitPending(usePool: poolFlag, quantization: quantMode)
        }
    }
}
