import Foundation
import MLX
import Cmlx

/// High-performance pre-allocated buffer pool for committed Prefix KV caches (2026 Dual Caching Tier 1).
///
/// **Problem**: Standard `LayerKVCache.commitPending()` calls `concatenated([keys, pending], axis: 2)`
/// on every block commit, causing O(N^2) memory reallocation churn and fragmentation on Apple Silicon unified memory.
///
/// **Solution**: `KVCachePool` allocates a contiguous tensor of shape `[1, nKV, capacity, headDim]` once,
/// writing incoming active blocks at `[..., committedLength ..< committedLength + B, ...]` via `mlx_slice_update`.
/// When capacity is exceeded, capacity doubles geometrically (amortized O(1) allocation).
public final class KVCachePool {
    public private(set) var capacity: Int
    public private(set) var committedLength: Int
    
    public private(set) var keysBuffer: MLXArray?
    public private(set) var valuesBuffer: MLXArray?
    
    public init(initialCapacity: Int = 256) {
        self.capacity = initialCapacity
        self.committedLength = 0
    }
    
    /// Logical byte footprint of the active committed slice `[1, nKV, committedLength, headDim]`.
    public var logicalBytes: Int {
        guard committedLength > 0, let k = keys, let v = values else { return 0 }
        return k.nbytes + v.nbytes
    }
    
    /// Total allocated byte footprint of the pre-allocated pool buffers `[1, nKV, capacity, headDim]`.
    public var allocatedBytes: Int {
        (keysBuffer?.nbytes ?? 0) + (valuesBuffer?.nbytes ?? 0)
    }
    
    /// Returns the active view of committed keys `[1, nKV, committedLength, headDim]`, or `nil` if empty.
    public var keys: MLXArray? {
        guard committedLength > 0, let buf = keysBuffer else { return nil }
        return buf[0..., 0..., 0 ..< committedLength, 0...]
    }
    
    /// Returns the active view of committed values `[1, nKV, committedLength, headDim]`, or `nil` if empty.
    public var values: MLXArray? {
        guard committedLength > 0, let buf = valuesBuffer else { return nil }
        return buf[0..., 0..., 0 ..< committedLength, 0...]
    }
    
    /// Commits a new block of pending keys and values into the pre-allocated pool in-place.
    public func commit(pendingKeys: MLXArray, pendingValues: MLXArray) {
        let B = pendingKeys.dim(0)
        let nKV = pendingKeys.dim(1)
        let blockLen = pendingKeys.dim(2)
        let headDim = pendingKeys.dim(3)
        
        let newLength = committedLength + blockLen
        
        // Ensure pool buffer is allocated and has sufficient capacity
        if keysBuffer == nil || valuesBuffer == nil {
            let initialCap = max(capacity, newLength)
            self.capacity = initialCap
            self.keysBuffer = MLXArray.zeros([B, nKV, initialCap, headDim], dtype: pendingKeys.dtype)
            self.valuesBuffer = MLXArray.zeros([B, nKV, initialCap, headDim], dtype: pendingValues.dtype)
        } else if newLength > capacity {
            // Geometric growth
            let newCap = max(capacity * 2, newLength)
            let oldKeys = keysBuffer!
            let oldValues = valuesBuffer!
            
            var newK = MLXArray.zeros([B, nKV, newCap, headDim], dtype: pendingKeys.dtype)
            var newV = MLXArray.zeros([B, nKV, newCap, headDim], dtype: pendingValues.dtype)
            
            if committedLength > 0 {
                let start: [Int32] = [0, 0, 0, 0]
                let stop: [Int32] = [Int32(B), Int32(nKV), Int32(committedLength), Int32(headDim)]
                let strides: [Int32] = [1, 1, 1, 1]
                
                let curSliceK = oldKeys[0..., 0..., 0 ..< committedLength, 0...]
                let curSliceV = oldValues[0..., 0..., 0 ..< committedLength, 0...]
                
                var resK = mlx_array_new()
                mlx_slice_update(&resK, newK.ctx, curSliceK.ctx, start, 4, stop, 4, strides, 4, StreamOrDevice.default.ctx)
                newK = MLXArray(resK)
                
                var resV = mlx_array_new()
                mlx_slice_update(&resV, newV.ctx, curSliceV.ctx, start, 4, stop, 4, strides, 4, StreamOrDevice.default.ctx)
                newV = MLXArray(resV)
            }
            self.capacity = newCap
            self.keysBuffer = newK
            self.valuesBuffer = newV
        }
        
        // In-place slice update into pre-allocated memory
        let start: [Int32] = [0, 0, Int32(committedLength), 0]
        let stop: [Int32] = [Int32(B), Int32(nKV), Int32(newLength), Int32(headDim)]
        let strides: [Int32] = [1, 1, 1, 1]
        
        var resK = mlx_array_new()
        mlx_slice_update(&resK, keysBuffer!.ctx, pendingKeys.ctx, start, 4, stop, 4, strides, 4, StreamOrDevice.default.ctx)
        keysBuffer = MLXArray(resK)
        
        var resV = mlx_array_new()
        mlx_slice_update(&resV, valuesBuffer!.ctx, pendingValues.ctx, start, 4, stop, 4, strides, 4, StreamOrDevice.default.ctx)
        valuesBuffer = MLXArray(resV)
        
        committedLength = newLength
    }
    
    public func clear() {
        committedLength = 0
        keysBuffer = nil
        valuesBuffer = nil
    }
}
