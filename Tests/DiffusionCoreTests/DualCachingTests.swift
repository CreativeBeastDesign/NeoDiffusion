import XCTest
import MLX
import MLXRandom
import Cmlx
@testable import DiffusionCore

final class DualCachingTests: XCTestCase {

    /// Test 1: Verify KV Quantization (bits: 8 and bits: 4) on standard KV shapes [1, nKV, L, D]
    func testKVQuantizationAccuracyAndSizing() throws {
        let B = 1
        let nKV = 4
        let L = 128
        let D = 128
        
        let kFP16 = MLXRandom.normal([B, nKV, L, D], key: MLXRandom.key(42)).asType(.float16)
        
        // 1. 8-bit quantization (FP8 / Int8 equivalent with fine-grained scales)
        let (kQ8, scales8, biases8) = MLX.quantized(kFP16, groupSize: 64, bits: 8)
        let kDeq8 = dequantized(kQ8, scales: scales8, biases: biases8, groupSize: 64, bits: 8, dtype: .float16)
        if let b8 = biases8 { eval(kQ8, scales8, b8, kDeq8) } else { eval(kQ8, scales8, kDeq8) }
        
        // Size comparison:
        // kFP16: 1 * 4 * 128 * 128 * 2 bytes = 131,072 bytes
        // kQ8: 1 * 4 * 128 * 32 uint32 elements * 4 bytes = 65,536 bytes (50% reduction)
        XCTAssertEqual(kQ8.shape, [B, nKV, L, D * 8 / 32])
        XCTAssertEqual(kQ8.dtype, .uint32)
        
        let diff8 = abs(kFP16 - kDeq8).max().item(Float.self)
        print(String(format: "[KV Quantization 8-bit] max diff = %.4f (expected < 0.05)", diff8))
        XCTAssertLessThan(diff8, 0.05, "8-bit quantization error too high: \(diff8)")
        
        // 2. 4-bit quantization (DeepSeek-V4.1-Flash E2M1 FP4 style)
        let (kQ4, scales4, biases4) = MLX.quantized(kFP16, groupSize: 64, bits: 4)
        let kDeq4 = dequantized(kQ4, scales: scales4, biases: biases4, groupSize: 64, bits: 4, dtype: .float16)
        if let b4 = biases4 { eval(kQ4, scales4, b4, kDeq4) } else { eval(kQ4, scales4, kDeq4) }

        
        // kQ4: 1 * 4 * 128 * 16 uint32 elements * 4 bytes = 32,768 bytes (75% reduction!)
        XCTAssertEqual(kQ4.shape, [B, nKV, L, D * 4 / 32])
        XCTAssertEqual(kQ4.dtype, .uint32)
    }
        
    /// Test 2: Verify mlx_slice_update for preallocated KV cache pool
    func testBufferPoolSliceUpdate() throws {
        let B = 1
        let nKV = 4
        let capacity = 128
        let D = 64
        let blockLen = 32
        
        var pool = MLXArray.zeros([B, nKV, capacity, D], dtype: .float16)
        
        let block0 = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(1)).asType(.float16)
        let block1 = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(2)).asType(.float16)
        
        // Write block 0 at offset 0
        let start0: [Int32] = [0, 0, 0, 0]
        let stop0: [Int32] = [Int32(B), Int32(nKV), Int32(blockLen), Int32(D)]
        let strides: [Int32] = [1, 1, 1, 1]
        
        var res0 = mlx_array_new()
        mlx_slice_update(&res0, pool.ctx, block0.ctx, start0, 4, stop0, 4, strides, 4, StreamOrDevice.default.ctx)
        pool = MLXArray(res0)
        
        // Write block 1 at offset 32
        let start1: [Int32] = [0, 0, Int32(blockLen), 0]
        let stop1: [Int32] = [Int32(B), Int32(nKV), Int32(2 * blockLen), Int32(D)]
        
        var res1 = mlx_array_new()
        mlx_slice_update(&res1, pool.ctx, block1.ctx, start1, 4, stop1, 4, strides, 4, StreamOrDevice.default.ctx)
        pool = MLXArray(res1)
        
        eval(pool)
        
        // Check that pool[..., 0..<64, ...] matches concatenated([block0, block1], axis: 2)
        let expected = concatenated([block0, block1], axis: 2)
        let slice = pool[0..., 0..., 0 ..< (2 * blockLen), 0...]
        eval(expected, slice)
        
        let diff = abs(slice - expected).max().item(Float.self)
        print(String(format: "[Pool Slice Update] max diff = %.6f", diff))
        XCTAssertEqual(diff, 0.0, "Pool slice update does not match exact concatenation!")
        
        // Verify remaining buffer is still zero
        let remaining = pool[0..., 0..., (2 * blockLen)..., 0...]
        eval(remaining)
        XCTAssertEqual(abs(remaining).max().item(Float.self), 0.0)
    }

    /// Test 3: Verify LayerKVCache integration with KVCachePool and KVCacheQuantization
    func testLayerKVCacheWithPoolAndQuantization() throws {
        let cache = LayerKVCache(usePool: true, initialPoolCapacity: 128, quantization: .fp8)
        let B = 1
        let nKV = 4
        let blockLen = 32
        let D = 64

        let k0 = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(10)).asType(.float16)
        let v0 = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(11)).asType(.float16)

        cache.pendingKeys = k0
        cache.pendingValues = v0
        cache.commitPending(usePool: true, quantization: .fp8)


        XCTAssertEqual(cache.committedLength, blockLen)
        XCTAssertNotNil(cache.compressedKeys)
        XCTAssertNotNil(cache.compressedValues)

        // Dequantize and check against original
        let kRestored = cache.keys!
        let diff = abs(k0 - kRestored).max().item(Float.self)
        print(String(format: "[LayerKVCache FP8] max reconstruction diff = %.4f", diff))
        XCTAssertLessThan(diff, 0.05)
    }

    /// Test 4: Verify CSA2 static layer classification
    func testCSA2LayerSelection() throws {
        // Boundary mode: layers < 8 reuse, layers >= 8 recompute
        let csa2Boundary = CSA2Config.boundary(8)
        // Step 0: never reuses (full mode for all layers)
        XCTAssertFalse(csa2Boundary.shouldReuse(layerIndex: 4, stepInBlock: 0, totalLayers: 16))
        XCTAssertFalse(csa2Boundary.shouldReuse(layerIndex: 10, stepInBlock: 0, totalLayers: 16))

        // Step > 0:
        XCTAssertTrue(csa2Boundary.shouldReuse(layerIndex: 4, stepInBlock: 1, totalLayers: 16))
        XCTAssertFalse(csa2Boundary.shouldReuse(layerIndex: 10, stepInBlock: 1, totalLayers: 16))

        // Alternating mode (stride 2): stride-aligned layers (0, 2, 4...) recompute; intermediate layers (1, 3, 5...) reuse
        let csa2Alt = CSA2Config.alternating(stride: 2)
        XCTAssertFalse(csa2Alt.shouldReuse(layerIndex: 0, stepInBlock: 0, totalLayers: 16))
        XCTAssertFalse(csa2Alt.shouldReuse(layerIndex: 0, stepInBlock: 1, totalLayers: 16))
        XCTAssertTrue(csa2Alt.shouldReuse(layerIndex: 1, stepInBlock: 1, totalLayers: 16))
        XCTAssertFalse(csa2Alt.shouldReuse(layerIndex: 2, stepInBlock: 1, totalLayers: 16))
    }
}



