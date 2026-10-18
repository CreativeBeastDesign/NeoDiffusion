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

        // Periodic refresh mode (stepStride: 2, minStep: 2):
        // Step 0 (all-[MASK]) and Step 1 (post-mask initialization) BOTH recompute;
        // Step 2 reuses, Step 3 recomputes, Step 4 reuses.
        let csa2Periodic = CSA2Config.alternating(stride: 2, stepStride: 2, minStep: 2)
        XCTAssertFalse(csa2Periodic.shouldReuse(layerIndex: 1, stepInBlock: 0, totalLayers: 16))
        XCTAssertFalse(csa2Periodic.shouldReuse(layerIndex: 1, stepInBlock: 1, totalLayers: 16))
        XCTAssertTrue(csa2Periodic.shouldReuse(layerIndex: 1, stepInBlock: 2, totalLayers: 16))
        XCTAssertFalse(csa2Periodic.shouldReuse(layerIndex: 1, stepInBlock: 3, totalLayers: 16))
        XCTAssertTrue(csa2Periodic.shouldReuse(layerIndex: 1, stepInBlock: 4, totalLayers: 16))
    }

    /// Test 5: Verify true memory savings (no duplicate raw/pool buffers) and incremental O(B) append parity
    func testLayerKVCacheTrueByteSavingsAndIncrementalAppend() throws {
        let B = 1
        let nKV = 4
        let blockLen = 32
        let D = 128
        let numBlocks = 4

        let cacheNone = LayerKVCache(usePool: false, quantization: .none)
        let cacheFP8 = LayerKVCache(usePool: false, quantization: .fp8)
        let cacheK8V4 = LayerKVCache(usePool: false, quantization: .k8v4)
        let cacheFP4 = LayerKVCache(usePool: false, quantization: .fp4)

        var allK: [MLXArray] = []
        var allV: [MLXArray] = []

        for b in 0 ..< numBlocks {
            let kb = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(UInt64(100 + b * 2))).asType(.float16)
            let vb = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(UInt64(101 + b * 2))).asType(.float16)
            allK.append(kb)
            allV.append(vb)

            for (cache, q) in [(cacheNone, KVCacheQuantization.none), (cacheFP8, .fp8), (cacheK8V4, .k8v4), (cacheFP4, .fp4)] {
                cache.pendingKeys = kb
                cache.pendingValues = vb
                cache.commitPending(usePool: false, quantization: q)
                cache.evalStorage()
            }
        }

        // Verify quantized caches do NOT retain uncompressed pool and allocatedBytes equals compressed byteCount
        for cache in [cacheFP8, cacheK8V4, cacheFP4] {
            XCTAssertNil(cache.pool, "Quantized cache must not retain uncompressed pool")
            XCTAssertEqual(cache.logicalBytes, cache.allocatedBytes)
            XCTAssertEqual(cache.allocatedBytes,
                           (cache.compressedKeys?.byteCount ?? 0) + (cache.compressedValues?.byteCount ?? 0))
        }

        // Exact byte accounting for 1 layer, 4 KV heads, 128 tokens, headDim=128 (FP16 input -> FP16 scales/biases):
        // FP16 (.none): 2 * (1 * 4 * 128 * 128 * 2) = 262,144 bytes
        // FP8 (.fp8, g64): packed = 2 * 65,536 = 131,072 + scales/biases (2 * 2 * 2048) = 139,264 bytes (-46.88%)
        // K8V4 (.k8v4, g64): K(69,632) + V(36,864) = 106,496 bytes (-59.38%)
        // FP4 (.fp4, g64): 2 * (32,768 + 4,096) = 73,728 bytes (-71.88%)
        XCTAssertEqual(cacheNone.allocatedBytes, 262_144)
        XCTAssertEqual(cacheFP8.allocatedBytes, 139_264)
        XCTAssertEqual(cacheK8V4.allocatedBytes, 106_496)
        XCTAssertEqual(cacheFP4.allocatedBytes, 73_728)

        // Verify incremental O(B) append produces bit-identical dequantized tensors to full-sequence quantization
        let fullK = concatenated(allK, axis: 2)
        let fullComp8 = CompressedKVTensor.compress(fullK, bits: 8, groupSize: 64)
        let incDeq8 = cacheFP8.keys!
        let fullDeq8 = fullComp8.dequantize()
        eval(incDeq8, fullDeq8)
        let appendDiff = abs(incDeq8 - fullDeq8).max().item(Float.self)
        XCTAssertEqual(appendDiff, 0.0, "Incremental O(B) append must be bit-identical to full quantization along axis 2")
    }

    /// Test 6: Verify projectQueriesOnly matches queryKeyValue Q slice
    func testProjectQueriesOnlyParity() throws {
        let hiddenSize = 64
        let nH = 4
        let nKV = 2
        let headDim = 16
        let attn = LLaDA2Attention(
            hiddenSize: hiddenSize, numHeads: nH, numKVHeads: nKV,
            headDim: headDim, rmsNormEps: 1e-6, useQkNorm: true)
        let x = MLXRandom.normal([1, 16, hiddenSize], key: MLXRandom.key(777))

        // Standard Linear
        let qkv = attn.queryKeyValue(x)
        let qFull = attn.queryLayernorm!(
            split(qkv, indices: [nH * headDim, (nH + nKV) * headDim], axis: -1)[0]
                .reshaped(1, 16, nH, headDim)
                .transposed(0, 2, 1, 3)
        )
        let qOnly = attn.queryLayernorm!(attn.projectQueriesOnly(x, B: 1, L: 16))
        eval(qFull, qOnly)
        let diffLinear = abs(qFull - qOnly).max().item(Float.self)
        XCTAssertLessThan(diffLinear, 1e-5, "Linear projectQueriesOnly must match full QKV Q slice")
    }

    /// Test 7 (Track 1): Operator-level 2x2 KV Bit-Allocation Factorial & Mitigations
    /// Measures Key/Value relative L2 error, Centered Score-Space Error, Attention KL Divergence,
    /// Top-1 Argmax Flip Rate, Peak Attention Weight Drop, and Attention Output Relative L2 Error
    /// across both Isotropic Gaussian KV and Realistic Persistent-Channel Outlier KV (KIVI regime).
    func testTrack1ScoreSpaceAndAttentionDivergenceAblation() throws {
        let B = 1
        let nH = 16
        let nKV = 4
        let headsPerKV = nH / nKV
        let blockLen = 32
        let numBlocks = 4
        let L = blockLen * numBlocks // 128 prefix tokens
        let Lq = 32                  // 32 active block query tokens
        let D = 128
        let scale = 1.0 / sqrt(Float(D))

        // 1. Verify incremental append parity and exact byte counts for new modes (.k4v8, .fp4g32, .kivi4)
        for qMode in [KVCacheQuantization.k4v8, .fp4g32, .kivi4] {
            let cache = LayerKVCache(usePool: false, quantization: qMode)
            var kBlocks: [MLXArray] = []
            var vBlocks: [MLXArray] = []
            for b in 0 ..< numBlocks {
                let kb = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(UInt64(500 + b * 2))).asType(.float16)
                let vb = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(UInt64(501 + b * 2))).asType(.float16)
                kBlocks.append(kb)
                vBlocks.append(vb)
                cache.pendingKeys = kb
                cache.pendingValues = vb
                cache.commitPending(usePool: false, quantization: qMode)
                cache.evalStorage()
            }
            let fullK = concatenated(kBlocks, axis: 2)
            let fullCompK = CompressedKVTensor.compress(
                fullK, bits: qMode.keyBits, groupSize: qMode.keyGroupSize, perChannel: qMode.keyPerChannel)
            let incDeqK = cache.keys!
            let fullDeqK = fullCompK.dequantize()
            eval(incDeqK, fullDeqK)
            let diffK = abs(incDeqK - fullDeqK).max().item(Float.self)
            XCTAssertEqual(diffK, 0.0, "Incremental append for \(qMode.rawValue) must match full compression bit-for-bit")
        }

        // Helper to evaluate a quantization mode on given Q [B, nH, Lq, D], K [B, nKV, L, D], V [B, nKV, L, D]
        struct PerturbationMetrics {
            let mode: String
            let bytes: Int
            let kRelL2: Float
            let vRelL2: Float
            let centeredScoreRelL2: Float
            let attnKL: Float
            let top1FlipRate: Float
            let peakProbDelta: Float
            let outRelL2: Float
        }

        func evaluateRegime(name: String, q: MLXArray, kBlocks: [MLXArray], vBlocks: [MLXArray]) -> [PerturbationMetrics] {
            let kFull = concatenated(kBlocks, axis: 2).asType(.float32)
            let vFull = concatenated(vBlocks, axis: 2).asType(.float32)
            let qF32 = q.asType(.float32)

            // Expand KV from nKV=4 to nH=16 for exact score & attention calculation
            func expandKV(_ x: MLXArray) -> MLXArray {
                let exp = expandedDimensions(x, axis: 2) // [B, nKV, 1, L, D]
                let bcast = broadcast(exp, to: [B, nKV, headsPerKV, L, D])
                return bcast.reshaped(B, nH, L, D)
            }

            let kRef = expandKV(kFull)
            let vRef = expandKV(vFull)
            let scoresRef = matmul(qF32, kRef.transposed(0, 1, 3, 2)) * scale // [B, nH, Lq, L]
            let scoresRefCentered = scoresRef - mean(scoresRef, axis: -1, keepDims: true)
            let probsRef = softmax(scoresRef, axis: -1)
            let top1Ref = argMax(probsRef, axis: -1)
            let maxProbRef = probsRef.max(axis: -1)
            let outRef = matmul(probsRef, vRef)
            eval(kRef, vRef, scoresRef, scoresRefCentered, probsRef, top1Ref, maxProbRef, outRef)

            let kNorm = sqrt(sum(kFull * kFull)).item(Float.self)
            let vNorm = sqrt(sum(vFull * vFull)).item(Float.self)
            let sCenteredNorm = sqrt(sum(scoresRefCentered * scoresRefCentered)).item(Float.self)
            let outNorm = sqrt(sum(outRef * outRef)).item(Float.self)

            let modes: [KVCacheQuantization] = [.fp8, .k8v4, .k4v8, .fp4, .fp4g32, .kivi4]
            var results: [PerturbationMetrics] = []

            for qMode in modes {
                let cache = LayerKVCache(usePool: false, quantization: qMode)
                for b in 0 ..< numBlocks {
                    cache.pendingKeys = kBlocks[b]
                    cache.pendingValues = vBlocks[b]
                    cache.commitPending(usePool: false, quantization: qMode)
                }
                cache.evalStorage()

                let kHat = cache.keys!.asType(.float32)
                let vHat = cache.values!.asType(.float32)
                let kHatExp = expandKV(kHat)
                let vHatExp = expandKV(vHat)

                let scoresHat = matmul(qF32, kHatExp.transposed(0, 1, 3, 2)) * scale
                let scoresHatCentered = scoresHat - mean(scoresHat, axis: -1, keepDims: true)
                let probsHat = softmax(scoresHat, axis: -1)
                let top1Hat = argMax(probsHat, axis: -1)
                let maxProbHat = probsHat.max(axis: -1)
                let outHat = matmul(probsHat, vHatExp)

                let kDiff = kHat - kFull
                let vDiff = vHat - vFull
                let sDiff = scoresHatCentered - scoresRefCentered
                let outDiff = outHat - outRef

                // KL(p_ref || p_hat) = sum(p_ref * (log(p_ref + eps) - log(p_hat + eps)))
                let eps: Float = 1e-8
                let klPerQuery = sum(probsRef * (log(probsRef + eps) - log(probsHat + eps)), axis: -1)
                let meanKL = mean(klPerQuery).item(Float.self)

                let flips = mean((top1Ref .!= top1Hat).asType(.float32)).item(Float.self) * 100.0
                let peakDelta = mean(maxProbHat - maxProbRef).item(Float.self)

                let kRel = sqrt(sum(kDiff * kDiff)).item(Float.self) / kNorm
                let vRel = sqrt(sum(vDiff * vDiff)).item(Float.self) / vNorm
                let sRel = sqrt(sum(sDiff * sDiff)).item(Float.self) / sCenteredNorm
                let oRel = sqrt(sum(outDiff * outDiff)).item(Float.self) / outNorm

                results.append(PerturbationMetrics(
                    mode: qMode.rawValue,
                    bytes: cache.allocatedBytes,
                    kRelL2: kRel,
                    vRelL2: vRel,
                    centeredScoreRelL2: sRel,
                    attnKL: meanKL,
                    top1FlipRate: flips,
                    peakProbDelta: peakDelta,
                    outRelL2: oRel
                ))
            }

            print("\n=== Track 1 Operator-Level Perturbation Analysis: \(name) ===")
            print("| Mode | Bytes (1L) | K Rel L2 | V Rel L2 | Centered Score Rel L2 | Attn KL(p||p_hat) | Top-1 Flip % | Peak Prob Delta | Output Rel L2 |")
            print("|---|---|---|---|---|---|---|---|---|")
            for r in results {
                print(String(format: "| %@ | %d | %.4f | %.4f | %.4f | %.5f | %.2f%% | %+.4f | %.4f |",
                             r.mode, r.bytes, r.kRelL2, r.vRelL2, r.centeredScoreRelL2, r.attnKL, r.top1FlipRate, r.peakProbDelta, r.outRelL2))
            }
            return results
        }

        // Regime A: Isotropic Gaussian Q, K, V
        let qIso = MLXRandom.normal([B, nH, Lq, D], key: MLXRandom.key(600)).asType(.float16)
        var kIsoBlocks: [MLXArray] = []
        var vIsoBlocks: [MLXArray] = []
        for b in 0 ..< numBlocks {
            kIsoBlocks.append(MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(UInt64(610 + b * 2))).asType(.float16))
            vIsoBlocks.append(MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(UInt64(611 + b * 2))).asType(.float16))
        }
        let isoResults = evaluateRegime(name: "Regime A (Isotropic Gaussian)", q: qIso, kBlocks: kIsoBlocks, vBlocks: vIsoBlocks)

        // Regime B: Persistent-Channel Key Outliers + Token-Varying Values (KIVI / Transformer KV structure)
        // 8 channels across D=128 have 8x scale + persistent bias in K (and matching query sensitivity),
        // while V has token-level spikes across 8 tokens per block.
        var channelScale = [Float](repeating: 1.0, count: D)
        var channelBias = [Float](repeating: 0.0, count: D)
        for idx in [3, 19, 37, 55, 71, 89, 104, 121] {
            channelScale[idx] = 8.0
            channelBias[idx] = (idx % 2 == 0) ? 6.0 : -6.0
        }
        let chScaleArr = MLXArray(channelScale).reshaped(1, 1, 1, D).asType(.float16)
        let chBiasArr = MLXArray(channelBias).reshaped(1, 1, 1, D).asType(.float16)

        var tokenScale = [Float](repeating: 1.0, count: blockLen)
        for t in [2, 9, 17, 26] {
            tokenScale[t] = 5.0
        }
        let tokScaleArr = MLXArray(tokenScale).reshaped(1, 1, blockLen, 1).asType(.float16)

        let qOut = (MLXRandom.normal([B, nH, Lq, D], key: MLXRandom.key(700)).asType(.float16) * sqrt(chScaleArr))
        var kOutBlocks: [MLXArray] = []
        var vOutBlocks: [MLXArray] = []
        for b in 0 ..< numBlocks {
            let kBase = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(UInt64(710 + b * 2))).asType(.float16)
            let vBase = MLXRandom.normal([B, nKV, blockLen, D], key: MLXRandom.key(UInt64(711 + b * 2))).asType(.float16)
            kOutBlocks.append(kBase * chScaleArr + chBiasArr)
            vOutBlocks.append(vBase * tokScaleArr)
        }
        let outResults = evaluateRegime(name: "Regime B (Persistent-Channel Key Outliers + Token-Spiky Values)", q: qOut, kBlocks: kOutBlocks, vBlocks: vOutBlocks)

        // Core structural assertions:
        // 1. K8V4 and K4V8 have identical byte footprint (106,496 bytes per layer for 128 tokens)
        let k8v4Iso = isoResults.first { $0.mode == "k8v4" }!
        let k4v8Iso = isoResults.first { $0.mode == "k4v8" }!
        XCTAssertEqual(k8v4Iso.bytes, 106_496)
        XCTAssertEqual(k4v8Iso.bytes, 106_496)

        // 2. K8V4 has ~15x lower Centered Score Error and ~200x lower Attention KL than K4V8!
        XCTAssertLessThan(k8v4Iso.centeredScoreRelL2 * 5.0, k4v8Iso.centeredScoreRelL2,
                          "K8V4 must have dramatically lower score-space perturbation than equal-byte K4V8")
        XCTAssertLessThan(k8v4Iso.attnKL * 20.0, k4v8Iso.attnKL,
                          "K8V4 must have dramatically lower attention KL divergence than equal-byte K4V8")

        // 3. In Regime B (Persistent-Channel Key Outliers), KIVI (kivi4) significantly reduces Key Rel L2,
        // Centered Score Error, and Attention KL compared to standard per-token FP4 (fp4).
        let fp4Out = outResults.first { $0.mode == "fp4" }!
        let kiviOut = outResults.first { $0.mode == "kivi4" }!
        XCTAssertLessThan(kiviOut.kRelL2, fp4Out.kRelL2,
                          "KIVI per-channel 4-bit Keys must reduce Key L2 error under channel outliers")
        XCTAssertLessThan(kiviOut.centeredScoreRelL2, fp4Out.centeredScoreRelL2,
                          "KIVI per-channel 4-bit Keys must reduce centered score error under channel outliers")
        XCTAssertLessThan(kiviOut.attnKL, fp4Out.attnKL,
                          "KIVI per-channel 4-bit Keys must reduce attention KL divergence under channel outliers")
    }
}



