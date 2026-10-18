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
/// the committed store. The commit is driven by the loop's commit-cleanliness rule: either a
/// dedicated capture forward over the final committed tokens, or — since 2026-10-17, when
/// `GenerationParams.cleanCommitSkipEligible` holds — the settle step's own pending K/V, which
/// is exact only because that step was a plain fresh forward over exactly the committed tokens.
/// See `DiffusionEngine+Entry.swift` (`onSettled`).
public final class LayerKVCache {
    /// Committed keys `[1, nKV, committedLen, D]` (post qk-norm + RoPE), or `nil` when empty.
    public var keys: MLXArray? {
        if let comp = compressedKeys {
            return comp.dequantize()
        }
        if let pool = pool, pool.committedLength > 0 {
            return pool.keys
        }
        return rawKeys
    }
    
    /// Committed values `[1, nKV, committedLen, D]`, or `nil` when empty.
    public var values: MLXArray? {
        if let comp = compressedValues {
            return comp.dequantize()
        }
        if let pool = pool, pool.committedLength > 0 {
            return pool.values
        }
        return rawValues
    }

    private var rawKeys: MLXArray?
    private var rawValues: MLXArray?

    /// Pre-allocated memory pool for zero-reallocation block commits (Tier 1, unquantized).
    public private(set) var pool: KVCachePool?
    
    /// Compact/quantized KV representation (FP8 / K8V4 / FP4) for reduced unified memory footprint.
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
    /// Host-tracked committed length for `ctxLens`, avoiding per-layer `.item()` GPU-CPU readbacks.
    public var cachedCtxLen: Int = -1
    /// Diagnostic counters for FlashBlock cache refresh vs reuse steps on this layer.
    public var flashBlockRefreshes: Int = 0
    public var flashBlockReuses: Int = 0
    
    // Persistent FlashBlock cache tensors
    public var attnOutPast: MLXArray?
    public var logsumexp: MLXArray?

    public init(usePool: Bool = false, initialPoolCapacity: Int = 256, quantization: KVCacheQuantization = .none) {
        self.quantizationMode = quantization
        if usePool && quantization == .none {
            self.pool = KVCachePool(initialCapacity: initialPoolCapacity)
        }
    }

    /// Number of committed key/value positions.
    public var committedLength: Int {
        if let comp = compressedKeys { return comp.originalShape[2] }
        if let pool = pool { return pool.committedLength }
        return rawKeys?.dim(2) ?? 0
    }

    /// Logical byte footprint of the active committed prefix KV in this layer.
    public var logicalBytes: Int {
        if let ck = compressedKeys, let cv = compressedValues {
            return ck.byteCount + cv.byteCount
        }
        if let pool = pool {
            return pool.logicalBytes
        }
        return (rawKeys?.nbytes ?? 0) + (rawValues?.nbytes ?? 0)
    }

    /// Total allocated byte footprint in this layer (including pool slack and FlashBlock accumulators).
    public var allocatedBytes: Int {
        var total = 0
        if let ck = compressedKeys, let cv = compressedValues {
            total += ck.byteCount + cv.byteCount
        }
        if let pool = pool {
            total += pool.allocatedBytes
        }
        total += (rawKeys?.nbytes ?? 0) + (rawValues?.nbytes ?? 0)
        total += (attnOutPast?.nbytes ?? 0) + (logsumexp?.nbytes ?? 0)
        return total
    }

    /// Append `pendingKeys`/`pendingValues` to the committed store (commit-cleanliness point).
    ///
    /// Supports pre-allocated memory pool and incremental low-bit (FP8 / K8V4 / FP4) compaction
    /// without retaining duplicate full-precision buffers.
    public func commitPending(usePool: Bool = false, quantization: KVCacheQuantization = .none) {
        guard let pk = pendingKeys, let pv = pendingValues else {
            preconditionFailure("commitPending called with no pending K/V — a capture forward "
                + "must run over the final committed tokens before commit")
        }
        
        let targetQuant = quantization != .none ? quantization : self.quantizationMode
        if targetQuant != .none {
            // Incremental block-level quantization: quantize only the newly committed block [1, nKV, B, D]
            // and append its packed uint32 + scales + biases. Never retain uncompressed rawKeys/pool!
            self.compressedKeys = compressedKeys.map { $0.append(pk) }
                ?? CompressedKVTensor.compress(
                    pk,
                    bits: targetQuant.keyBits,
                    groupSize: targetQuant.keyGroupSize,
                    perChannel: targetQuant.keyPerChannel
                )
            self.compressedValues = compressedValues.map { $0.append(pv) }
                ?? CompressedKVTensor.compress(
                    pv,
                    bits: targetQuant.valueBits,
                    groupSize: targetQuant.valueGroupSize,
                    perChannel: false
                )
            self.quantizationMode = targetQuant
            self.rawKeys = nil
            self.rawValues = nil
            self.pool = nil
        } else {
            let shouldPool = usePool || self.pool != nil
            if shouldPool {
                if pool == nil {
                    pool = KVCachePool(initialCapacity: max(256, pk.dim(2) * 4))
                }
                pool!.commit(pendingKeys: pk, pendingValues: pv)
                self.rawKeys = nil
                self.rawValues = nil
            } else {
                rawKeys = rawKeys.map { concatenated([$0, pk], axis: 2) } ?? pk
                rawValues = rawValues.map { concatenated([$0, pv], axis: 2) } ?? pv
            }
        }

        pendingKeys = nil
        pendingValues = nil
    }

    // MARK: - Session prefix-cache support (exp-5B)

    /// Drops committed positions `[length, committedLength)`, for every storage form this layer can
    /// hold (raw, pool, compressed incl. the per-channel kivi4 Key layout). Pending K/V and the
    /// per-call FlashBlock state are not touched — callers only truncate a store that is between
    /// generation calls. Lazy: slices are views until the next commit concatenates/updates them.
    public func truncateCommitted(to length: Int) {
        let current = committedLength
        precondition(length >= 0 && length <= current,
            "LayerKVCache.truncateCommitted: length \(length) outside [0, \(current)]")
        if length == current { return }
        if let ck = compressedKeys, let cv = compressedValues {
            if length == 0 {
                compressedKeys = nil
                compressedValues = nil
            } else {
                compressedKeys = ck.truncated(toLength: length)
                compressedValues = cv.truncated(toLength: length)
            }
        } else if let pool {
            pool.truncate(to: length)
        } else if let rk = rawKeys, let rv = rawValues {
            if length == 0 {
                rawKeys = nil
                rawValues = nil
            } else {
                rawKeys = rk[0..., 0..., 0 ..< length, 0...]
                rawValues = rv[0..., 0..., 0 ..< length, 0...]
            }
        }
    }

    /// Moves `other`'s committed storage (raw / pool / compressed, plus its quantization mode) into
    /// this layer and leaves `other` empty. Ownership is *moved*, not shared, so the pool's
    /// in-place `mlx_slice_update` never sees a second reference to its buffer. This layer must be
    /// empty (fresh) — it is how a session seeds a new call's per-call cache.
    public func adoptCommittedStorage(from other: LayerKVCache) {
        precondition(committedLength == 0 && pendingKeys == nil,
            "adoptCommittedStorage: target layer must be empty")
        rawKeys = other.rawKeys
        rawValues = other.rawValues
        pool = other.pool
        compressedKeys = other.compressedKeys
        compressedValues = other.compressedValues
        quantizationMode = other.quantizationMode
        other.rawKeys = nil
        other.rawValues = nil
        other.pool = nil
        other.compressedKeys = nil
        other.compressedValues = nil
    }

    /// Forces evaluation of the underlying stored tensors (compressed, pooled, or raw) without
    /// triggering unnecessary dequantization.
    public func evalStorage() {
        if let ck = compressedKeys, let cv = compressedValues {
            if let bk = ck.biases, let bv = cv.biases {
                eval(ck.packed, ck.scales, bk, cv.packed, cv.scales, bv)
            } else {
                eval(ck.packed, ck.scales, cv.packed, cv.scales)
            }
        } else if let pool = pool, let kb = pool.keysBuffer, let vb = pool.valuesBuffer {
            eval(kb, vb)
        } else if let rk = rawKeys, let rv = rawValues {
            eval(rk, rv)
        }
    }
}
