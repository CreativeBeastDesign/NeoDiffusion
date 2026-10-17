import XCTest
import MLX
import MLXRandom
@testable import DiffusionCore

final class FlashBlockRunnerTests: XCTestCase {

    func testNativeLayoutDirectParityFP16() throws {
        try runDirectParity(dtype: .float16, headDim: 128, numHeads: 16, numKVHeads: 4, blockLen: 32, committedLen: 128)
    }

    func testNativeLayoutDirectParityFP32() throws {
        try runDirectParity(dtype: .float32, headDim: 32, numHeads: 4, numKVHeads: 1, blockLen: 16, committedLen: 32)
    }

    func testNativeLayoutDirectParityHeads64() throws {
        try runDirectParity(dtype: .float16, headDim: 64, numHeads: 8, numKVHeads: 2, blockLen: 32, committedLen: 64)
    }

    private func runDirectParity(
        dtype: DType,
        headDim: Int,
        numHeads: Int,
        numKVHeads: Int,
        blockLen: Int,
        committedLen: Int
    ) throws {
        let B = 1
        let L = blockLen

        let config = FlashBlockConfig(
            numSeqs: B,
            blockLen: L,
            numQHeads: numHeads,
            numKVHeads: numKVHeads,
            headDim: headDim,
            pageSize: 256,
            maxPagesPerSeq: 64,
            blockM: 32,
            blockN: 32,
            tau: 4,
            dtype: dtype
        )
        let runner = try FlashBlockRunner(config: config)

        let Q = MLXRandom.normal([B, numHeads, L, headDim], key: MLXRandom.key(10)).asType(dtype)
        let Kcur = MLXRandom.normal([B, numKVHeads, L, headDim], key: MLXRandom.key(20)).asType(dtype)
        let Vcur = MLXRandom.normal([B, numKVHeads, L, headDim], key: MLXRandom.key(30)).asType(dtype)
        let kCache = MLXRandom.normal([B, numKVHeads, committedLen, headDim], key: MLXRandom.key(40)).asType(dtype)
        let vCache = MLXRandom.normal([B, numKVHeads, committedLen, headDim], key: MLXRandom.key(50)).asType(dtype)
        let ctxLens = MLXArray([Int32(committedLen)])

        let attnOutPast = MLXArray.zeros([B, numHeads, L, headDim], dtype: .float32)
        let logsumexp = MLXArray.zeros([B, numHeads, L], dtype: .float32)

        // 1. Refresh cache pass
        let refreshRes = runner.forward(
            kind: .refreshCache,
            Q: Q,
            Kcur: Kcur,
            Vcur: Vcur,
            kCache: kCache,
            vCache: vCache,
            ctxLens: ctxLens,
            attnOutPast: attnOutPast,
            logsumexp: logsumexp
        )
        eval(refreshRes.O, refreshRes.attnOutPast, refreshRes.logsumexp)

        // 2. Reuse cache pass
        let reuseRes = runner.forward(
            kind: .reuseCache,
            Q: Q,
            Kcur: Kcur,
            Vcur: Vcur,
            kCache: nil,
            vCache: nil,
            ctxLens: ctxLens,
            attnOutPast: refreshRes.attnOutPast,
            logsumexp: refreshRes.logsumexp
        )
        eval(reuseRes.O)

        // Reference: attend against [kCache ++ Kcur]
        let kAll = concatenated([kCache, Kcur], axis: 2)
        let vAll = concatenated([vCache, Vcur], axis: 2)
        let scale = pow(Float(headDim), -0.5)
        let expectedAttended = LLaDA2Attention.attend(
            queries: Q, keys: kAll, values: vAll, scale: scale, mask: nil
        ) // shape: [B, numHeads, L, headDim]

        // Output of FlashBlock is [B, L, numHeads * headDim]
        let expectedFlat = expectedAttended.transposed(0, 2, 1, 3).reshaped(B, L, numHeads * headDim)

        let diffRefresh = abs(refreshRes.O - expectedFlat).max().item(Float.self)
        let diffReuse = abs(reuseRes.O - expectedFlat).max().item(Float.self)

        let tol: Float = (dtype == .float32) ? 1e-4 : 5e-3
        print(String(format: "[FlashBlock direct] H=%d, HKV=%d, D=%d, dtype=%@ | refreshDiff=%.2e, reuseDiff=%.2e (tol=%.2e)",
                     numHeads, numKVHeads, headDim, "\(dtype)", diffRefresh, diffReuse, tol))

        XCTAssertLessThanOrEqual(diffRefresh, tol, "refreshCache vs SDPA max diff \(diffRefresh) > \(tol)")
        XCTAssertLessThanOrEqual(diffReuse, tol, "reuseCache vs SDPA max diff \(diffReuse) > \(tol)")
    }

    /// Isolated attention microbenchmark comparing raw latency (ms) of:
    /// 1. Vanilla MLX SDPA + transpose
    /// 2. FlashBlock refreshCache (external prefix pass + active block)
    /// 3. FlashBlock reuseCache (internal active pass + LSE blend)
    /// 4. Prior FlashBlock with transpose overhead
    func testAttentionMicrobench() throws {
        let (B, H, HKV, D, L, committedLen) = (1, 16, 4, 128, 32, 512)
        let reps = 50

        let config = FlashBlockConfig(
            numSeqs: B, blockLen: L,
            numQHeads: H, numKVHeads: HKV, headDim: D,
            pageSize: 256, maxPagesPerSeq: 64,
            blockM: 32, blockN: 32,
            tau: 4, dtype: .float16
        )
        let runner = try FlashBlockRunner(config: config)

        let Q = MLXRandom.normal([B, H, L, D], key: MLXRandom.key(1)).asType(.float16)
        let Kcur = MLXRandom.normal([B, HKV, L, D], key: MLXRandom.key(2)).asType(.float16)
        let Vcur = MLXRandom.normal([B, HKV, L, D], key: MLXRandom.key(3)).asType(.float16)
        let kCache = MLXRandom.normal([B, HKV, committedLen, D], key: MLXRandom.key(4)).asType(.float16)
        let vCache = MLXRandom.normal([B, HKV, committedLen, D], key: MLXRandom.key(5)).asType(.float16)
        let ctxLens = MLXArray([Int32(committedLen)])
        let attnOutPast = MLXArray.zeros([B, H, L, D], dtype: .float32)
        let logsumexp = MLXArray.zeros([B, H, L], dtype: .float32)

        let kAll = concatenated([kCache, Kcur], axis: 2)
        let vAll = concatenated([vCache, Vcur], axis: 2)
        let scale = pow(Float(D), -0.5)

        func timeMs(_ body: () -> [MLXArray]) -> Double {
            for arr in body() { eval(arr) } // warmup
            let start = Date()
            for _ in 0..<reps {
                for arr in body() { eval(arr) }
            }
            return (Date().timeIntervalSince(start) * 1000.0) / Double(reps)
        }

        let sdpaMs = timeMs {
            let att = MLXFast.scaledDotProductAttention(queries: Q, keys: kAll, values: vAll, scale: scale, mask: nil)
            let out = att.transposed(0, 2, 1, 3).reshaped(B, L, H * D)
            return [out]
        }

        let refreshMs = timeMs {
            let res = runner.forward(
                kind: .refreshCache,
                Q: Q, Kcur: Kcur, Vcur: Vcur,
                kCache: kCache, vCache: vCache, ctxLens: ctxLens,
                attnOutPast: attnOutPast, logsumexp: logsumexp
            )
            return [res.O, res.attnOutPast, res.logsumexp]
        }

        // Run one refresh to get actual past cache values for reuse
        let initialRefresh = runner.forward(
            kind: .refreshCache,
            Q: Q, Kcur: Kcur, Vcur: Vcur,
            kCache: kCache, vCache: vCache, ctxLens: ctxLens,
            attnOutPast: attnOutPast, logsumexp: logsumexp
        )
        eval(initialRefresh.attnOutPast, initialRefresh.logsumexp)

        let reuseMs = timeMs {
            let res = runner.forward(
                kind: .reuseCache,
                Q: Q, Kcur: Kcur, Vcur: Vcur,
                kCache: nil, vCache: nil, ctxLens: ctxLens,
                attnOutPast: initialRefresh.attnOutPast, logsumexp: initialRefresh.logsumexp
            )
            return [res.O]
        }

        // Simulate prior overhead: transposing cacheKeys and cacheValues on reuse
        let priorReuseWithTransposesMs = timeMs {
            let kc = kCache.transposed(0, 2, 1, 3).reshaped(committedLen, HKV, D).contiguous()
            let vc = vCache.transposed(0, 2, 1, 3).reshaped(committedLen, HKV, D).contiguous()
            let qc = Q.transposed(0, 2, 1, 3).reshaped(B * L, H, D).contiguous()
            let res = runner.forward(
                kind: .reuseCache,
                Q: Q, Kcur: Kcur, Vcur: Vcur,
                kCache: nil, vCache: nil, ctxLens: ctxLens,
                attnOutPast: initialRefresh.attnOutPast, logsumexp: initialRefresh.logsumexp
            )
            return [res.O, kc, vc, qc]
        }

        print(String(format: """
        === Attention Pipeline Latency Microbench (H=%d, HKV=%d, D=%d, L=%d, Ctx=%d) ===
          • Vanilla MLX SDPA + transpose:               %6.3f ms
          • Optimized FlashBlock refreshCache:          %6.3f ms
          • Optimized FlashBlock reuseCache (zero-copy):%6.3f ms
          • Prior FlashBlock reuse (with transposes):   %6.3f ms
          • Reuse Speedup over Vanilla SDPA:            %6.1fx faster
          • Reuse Speedup over Prior Transposing Path:  %6.1fx faster
        ================================================================================
        """, H, HKV, D, L, committedLen, sdpaMs, refreshMs, reuseMs, priorReuseWithTransposesMs,
        sdpaMs / reuseMs, priorReuseWithTransposesMs / reuseMs))
    }
}

