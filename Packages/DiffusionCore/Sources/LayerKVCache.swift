import Foundation
import MLX
import Metal

/// Per-layer key/value store for the cached attention path (phase-2 §5 M5, deviation 1)
/// enhanced with 2026 Unified Dual Caching Tier 1 (Pre-allocated Memory Pool + Quantization).
///
/// Holds one layer's **committed-block** K/V — the keys already have qk-norm + partial RoPE
/// applied at their own absolute positions, exactly as an uncached forward would produce them,
/// so a cached forward's active-block queries attend to `[committed ++ active]` with *no mask*
/// (every committed key is in an allowed, ≤-current block). This is what makes
/// `ExactPrefixCache` mathematically exact under the `.strict` mask.
///
/// `pending*` holds the most recent forward's active-block K/V; `commitPending()` appends it to
/// the committed store. The commit is driven by the loop's commit-cleanliness rule (a dedicated
/// capture forward over the final committed tokens), never by a raw denoising step — see
/// `DiffusionEngine`.
public final class LayerKVCache {
    /// Committed keys `[1, nKV, committedLen, D]` (post qk-norm + RoPE), or `nil` when empty.
    public var keys: MLXArray? {
        if let pool = pool, pool.committedLength > 0 {
            return pool.keys
        }
        if let comp = compressedKeys {
            return comp.dequantize()
        }
        return rawKeys
    }
    
    /// Committed values `[1, nKV, committedLen, D]`, or `nil` when empty.
    public var values: MLXArray? {
        if let pool = pool, pool.committedLength > 0 {
            return pool.values
        }
        if let comp = compressedValues {
            return comp.dequantize()
        }
        return rawValues
    }

    private var rawKeys: MLXArray?
    private var rawValues: MLXArray?

    /// Pre-allocated memory pool for zero-reallocation block commits (Tier 1).
    public private(set) var pool: KVCachePool?
    
    /// Compact/quantized KV representation (FP8 / FP4) for reduced unified memory footprint.
    public private(set) var compressedKeys: CompressedKVTensor?
    public private(set) var compressedValues: CompressedKVTensor?
    public var quantizationMode: KVCacheQuantization = .none

    /// Last forward's active-block keys `[1, nKV, B, D]` (candidate to commit).
    public var pendingKeys: MLXArray?
    /// Last forward's active-block values `[1, nKV, B, D]`.
    public var pendingValues: MLXArray?

    // FlashBlock runner and auxiliary buffers (WP-3b)
    public var flashBlockRunner: FlashBlockRunner?
    public var blockTables: MLXArray?
    public var ctxLens: MLXArray?
    
    // Persistent FlashBlock cache tensors
    public var attnOutPast: MLXArray?
    public var logsumexp: MLXArray?

    public init(usePool: Bool = false, initialPoolCapacity: Int = 512, quantization: KVCacheQuantization = .none) {
        if usePool {
            self.pool = KVCachePool(initialCapacity: initialPoolCapacity)
        }
        self.quantizationMode = quantization
    }

    /// Number of committed key/value positions.
    public var committedLength: Int {
        if let pool = pool { return pool.committedLength }
        if let comp = compressedKeys { return comp.originalShape[2] }
        return rawKeys?.dim(2) ?? 0
    }

    /// Append `pendingKeys`/`pendingValues` to the committed store (commit-cleanliness point).
    ///
    /// Supports pre-allocated memory pool and low-bit (FP8/FP4) compaction.
    public func commitPending(usePool: Bool = false, quantization: KVCacheQuantization = .none) {
        guard let pk = pendingKeys, let pv = pendingValues else {
            preconditionFailure("commitPending called with no pending K/V — a capture forward "
                + "must run over the final committed tokens before commit")
        }
        
        let shouldPool = usePool || self.pool != nil
        if shouldPool {
            if pool == nil {
                pool = KVCachePool(initialCapacity: max(512, pk.dim(2) * 4))
            }
            pool!.commit(pendingKeys: pk, pendingValues: pv)
        } else {
            rawKeys = rawKeys.map { concatenated([$0, pk], axis: 2) } ?? pk
            rawValues = rawValues.map { concatenated([$0, pv], axis: 2) } ?? pv
        }
        
        let targetQuant = quantization != .none ? quantization : self.quantizationMode
        if targetQuant != .none {
            let curK = shouldPool ? pool!.keys! : rawKeys!
            let curV = shouldPool ? pool!.values! : rawValues!
            self.compressedKeys = CompressedKVTensor.compress(curK, mode: targetQuant)
            self.compressedValues = CompressedKVTensor.compress(curV, mode: targetQuant)
            self.quantizationMode = targetQuant
        }

        pendingKeys = nil
        pendingValues = nil
    }
}
