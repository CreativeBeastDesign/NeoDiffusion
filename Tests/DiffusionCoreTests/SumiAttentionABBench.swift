import XCTest
import MLX
import MLXRandom
@testable import DiffusionCore

/// A/B micro-bench for the off-by-one attention variants at real Sumi shapes
/// (B=1, 32 Q / 8 KV heads, head_dim 128), requested by André 2026-07-09. Random FP16
/// tensors — no model load. Measures wall-clock and peak GPU memory per variant:
///
///   eager     — materialise `[32, S, S]` FP32 scores, softmax_one, weights @ v
///   fast      — fused SDPA × sigmoid(chunked row-LSE)  (QKᵀ computed twice, once fused)
///   sdpaOnly  — fused SDPA with standard softmax: the floor (what a perfect
///               single-pass off-by-one Metal kernel could approach)
///   lseOnly   — the chunked LSE alone (= the extra cost `fast` pays over the floor)
///
/// Context for reading the numbers: at canvas 1024–1536 attention is <1% of the step's
/// FLOPs (the forward is GEMM-bound in the 7B weights), so these deltas bound what any
/// attention work can ever buy at Sumi scale. Opt-in:
///   NEODIFFUSION_SUMI_ABBENCH=1 swift test --filter SumiAttentionABBench
final class SumiAttentionABBench: XCTestCase {

    func testAttentionVariants() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_ABBENCH"] == "1",
            "opt-in: set NEODIFFUSION_SUMI_ABBENCH=1")

        let (H, HKV, D) = (32, 8, 128)
        let reps = 5

        func time(_ label: String, _ body: () -> MLXArray) -> (ms: Double, peakGB: Double) {
            eval(body())  // warmup (kernel compile, allocation)
            GPU.resetPeakMemory()
            let start = Date()
            for _ in 0 ..< reps { eval(body()) }
            let ms = Date().timeIntervalSince(start) * 1000 / Double(reps)
            return (ms, Double(Memory.peakMemory) / 1_073_741_824)
        }

        for S in [512, 1024, 1536] {
            let key = MLXRandom.key(UInt64(S))
            let keys3 = MLXRandom.split(key: key, into: 3)
            let q = MLXRandom.normal([1, H, S, D], key: keys3[0]).asType(.float16)
            let k = MLXRandom.normal([1, HKV, S, D], key: keys3[1]).asType(.float16)
            let v = MLXRandom.normal([1, HKV, S, D], key: keys3[2]).asType(.float16)
            let scale = pow(Float(D), -0.5)

            let eager = time("eager") {
                OffByOneAttention.attendEager(queries: q, keys: k, values: v, scale: scale, mask: nil)
            }
            let fast = time("fast(sinks)") {
                OffByOneAttention.attendFast(queries: q, keys: k, values: v, scale: scale)
            }
            let fastLSE = time("fastLSE") {
                OffByOneAttention.attendFastLSE(queries: q, keys: k, values: v, scale: scale)
            }
            let sdpa = time("sdpaOnly") {
                MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: scale, mask: nil)
            }
            let lse = time("lseOnly") {
                OffByOneAttention.rowLogSumExp(queries: q, keys: k, scale: scale)
            }
            let lse512 = time("lse-c512") {
                OffByOneAttention.rowLogSumExp(queries: q, keys: k, scale: scale, chunkSize: 512)
            }
            let lseFull = time("lse-c\(S)") {
                OffByOneAttention.rowLogSumExp(queries: q, keys: k, scale: scale, chunkSize: S)
            }

            print(String(
                format: "[sumi-AB] S=%4d  eager %7.1f ms / %.2f GB | fast(sinks) %6.1f ms / %.2f GB | "
                    + "fastLSE %6.1f ms / %.2f GB | sdpaOnly %6.1f ms / %.2f GB | "
                    + "lseOnly %6.1f ms / %.2f GB | lse-c512 %6.1f ms | lse-full %6.1f ms",
                S, eager.ms, eager.peakGB, fast.ms, fast.peakGB,
                fastLSE.ms, fastLSE.peakGB, sdpa.ms, sdpa.peakGB,
                lse.ms, lse.peakGB, lse512.ms, lseFull.ms))

            // Sanity while we're here: fast ≡ eager numerically at these shapes.
            let a = OffByOneAttention.attendFast(queries: q, keys: k, values: v, scale: scale)
            let b = OffByOneAttention.attendEager(
                queries: q, keys: k, values: v, scale: scale, mask: nil)
            let d = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
            XCTAssertLessThanOrEqual(d, 2e-3, "fast vs eager at S=\(S): max abs diff \(d)")
        }
    }
}
