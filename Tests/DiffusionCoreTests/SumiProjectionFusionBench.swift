import XCTest
import MLX
import MLXRandom

/// S3.2(c) decision micro-bench (requested by André 2026-07-09): does row-concatenating
/// projection weights into one quantized GEMM pay? Two candidates at real Sumi shapes
/// (hidden 4096, 4-bit affine g64, FP16 activations, S = 1024):
///
///   QKV:      q [4096] + k [1024] + v [1024]  vs  fused [6144]
///   gate+up:  [12288] + [12288]               vs  fused [24576]
///
/// Row-wise concat is quantization-valid: each output row carries its own scales/biases,
/// so packed weights, scales and biases all stack along the output axis. Numerical
/// equivalence is asserted, not assumed. Projected step savings = per-layer delta × 36.
/// Opt-in: NEODIFFUSION_SUMI_ABBENCH=1 swift test --filter SumiProjectionFusionBench
final class SumiProjectionFusionBench: XCTestCase {

    func testFusedVsSeparateProjections() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_ABBENCH"] == "1",
            "opt-in: set NEODIFFUSION_SUMI_ABBENCH=1")

        let H = 4096
        let S = 1024
        let reps = 10

        func makeQuantized(_ rows: Int, seed: UInt64) -> (MLXArray, MLXArray, MLXArray) {
            let w = MLXRandom.normal([rows, H], key: MLXRandom.key(seed)).asType(.float16) * 0.05
            let (wq, scales, biases) = quantized(w, groupSize: 64, bits: 4)
            eval(wq, scales, biases!)  // affine mode always produces biases
            return (wq, scales, biases!)
        }

        func concat3(_ parts: [(MLXArray, MLXArray, MLXArray)]) -> (MLXArray, MLXArray, MLXArray) {
            let fused = (
                concatenated(parts.map(\.0), axis: 0),
                concatenated(parts.map(\.1), axis: 0),
                concatenated(parts.map(\.2), axis: 0)
            )
            eval(fused.0, fused.1, fused.2)
            return fused
        }

        func qmm(_ x: MLXArray, _ w: (MLXArray, MLXArray, MLXArray)) -> MLXArray {
            quantizedMM(x, w.0, scales: w.1, biases: w.2, transpose: true, groupSize: 64, bits: 4)
        }

        func time(_ body: () -> [MLXArray]) -> Double {
            eval(body())  // warmup
            let start = Date()
            for _ in 0 ..< reps { eval(body()) }
            return Date().timeIntervalSince(start) * 1000 / Double(reps)
        }

        let x = MLXRandom.normal([1, S, H], key: MLXRandom.key(0)).asType(.float16)

        for (label, rowSets) in [("qkv", [4096, 1024, 1024]), ("gate+up", [12288, 12288])] {
            let parts = rowSets.enumerated().map { makeQuantized($0.element, seed: UInt64($0.offset + 1)) }
            let fused = concat3(parts)

            // Numerical equivalence: fused output slices ≡ separate outputs.
            let separateOut = parts.map { qmm(x, $0) }
            let fusedOut = qmm(x, fused)
            var offset = 0
            for (i, rows) in rowSets.enumerated() {
                let slice = fusedOut[.ellipsis, offset ..< (offset + rows)]
                let d = abs(slice - separateOut[i]).max().item(Float.self)
                XCTAssertLessThanOrEqual(d, 1e-3, "\(label) part \(i): fused ≠ separate (\(d))")
                offset += rows
            }

            let tSeparate = time { parts.map { qmm(x, $0) } }
            let tFused = time { [qmm(x, fused)] }
            let perStep36 = (tSeparate - tFused) * 36 / 1000
            print(String(
                format: "[sumi-FUSE] %@ S=%d: separate %.2f ms | fused %.2f ms | "
                    + "delta ×36 layers ≈ %+.2f s/step",
                label, S, tSeparate, tFused, perStep36))
        }
    }
}
