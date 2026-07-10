import XCTest
import MLX
import MLXNN
import MLXRandom
@testable import DiffusionCore

/// M6 Phase-A micro-benches (m6-logbook H2/H4, Finding 3): dispatch-variant ranking and
/// module-level cost attribution at the real LLaDA2.1-mini shapes. Random weights — perf
/// attribution only, values irrelevant; numerical equivalence between dispatch variants
/// IS asserted so a future fallback swap starts from proven-identical math.
///
/// **This is a negative regression test with scoped assumptions.** The 2026-07-10 result
/// (production `gatherQuantizedMM` fastest: 4.5 ms vs 11.7 ms dense-all / 26.4 ms
/// dequant+gather ⇒ the segmented-qmm fallback would be a regression, do not build) holds
/// for: M1-generation GPU, mlx-swift 0.31.6, [E=256 × I=512 × H=2048], T=32 × K=8,
/// 4-bit g64 affine. Dispatch rankings can flip with MLX version, GPU generation, expert
/// count, or quantization layout — RE-RUN this bench before trusting the ranking after
/// any of those change (Studio M2 Ultra run planned).
///
/// Opt-in (GPU-heavy, allocates ~1–2 GB): NEODIFFUSION_M6_BENCH=1 swift test --filter LLaDAMoEDispatchBench
final class LLaDAMoEDispatchBench: XCTestCase {

    // Real model shapes (CLAUDE.md quick facts).
    static let E = 256        // experts
    static let H = 2048       // hidden
    static let I = 512        // moe intermediate
    static let T = 32         // active-block tokens
    static let K = 8          // experts per token

    private func skipUnlessOptedIn() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_M6_BENCH"] == "1",
            "opt-in: set NEODIFFUSION_M6_BENCH=1 (GPU-heavy micro-bench)")
    }

    /// Wall-clock a closure: `warmup` un-timed reps, then `reps` timed (each fully eval'd).
    private func time(_ label: String, warmup: Int = 2, reps: Int = 10,
                      _ body: () -> MLXArray) -> Double {
        for _ in 0 ..< warmup { eval(body()) }
        let start = Date()
        for _ in 0 ..< reps { eval(body()) }
        let mean = Date().timeIntervalSince(start) / Double(reps)
        print(String(format: "[m6-bench] %@: %.4f s/op", label, mean))
        return mean
    }

    /// H2 — the three dispatch variants for one routed projection (gate/up shape
    /// [E, I, H]), identical inputs, asserted-equivalent outputs where comparable:
    /// (i) `gatherQuantizedMM` exactly as `QuantizedSwitchLinear` issues it,
    /// (ii) dequantize once + `gatherMM` (the F16 gather path),
    /// (iii) one dense `quantizedMatmul` over ALL experts (compute-everything upper
    ///      bound: [T, H] × [E·I, H]ᵀ — 32× the useful FLOPs for K=8).
    func testGatherQMMVariants() throws {
        try skipUnlessOptedIn()
        MLXRandom.seed(11)
        let w = MLXRandom.normal([Self.E, Self.I, Self.H]) * 0.05
        let (wq, scales, biases) = MLX.quantized(w, groupSize: 64, bits: 4)
        let wDeq = dequantized(
            wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
        let x = MLXRandom.normal([Self.T, 1, 1, Self.H]).asType(.float16)
        let indices = MLXRandom.randInt(0 ..< Int32(Self.E), [Self.T, Self.K])
        eval(wq, scales, biases, wDeq, x, indices)

        let tGather = time("gatherQuantizedMM (production path)") {
            gatherQuantizedMM(
                x, wq, scales: scales, biases: biases,
                rhsIndices: indices, transpose: true, groupSize: 64, bits: 4)
        }
        let tDeqGather = time("dequantized + gatherMM") {
            gatherMM(x.asType(wDeq.dtype), wDeq.swappedAxes(-1, -2), rhsIndices: indices)
        }
        let tFullQMM = time("dense qmm over ALL experts (upper bound)") {
            quantizedMatmul(
                x.reshaped(Self.T, Self.H),
                wq.reshaped(Self.E * Self.I, Self.H / 8),
                scales: scales.reshaped(Self.E * Self.I, Self.H / 64),
                biases: biases!.reshaped(Self.E * Self.I, Self.H / 64),
                transpose: true, groupSize: 64, bits: 4)
        }

        // Equivalence gate (i) vs (ii): same math up to fp16 accumulation.
        let a = gatherQuantizedMM(
            x, wq, scales: scales, biases: biases,
            rhsIndices: indices, transpose: true, groupSize: 64, bits: 4)
        let b = gatherMM(x.asType(wDeq.dtype), wDeq.swappedAxes(-1, -2), rhsIndices: indices)
        let maxDelta = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        print("[m6-bench] gather vs dequantized-gather max |Δ| = \(maxDelta)")
        XCTAssertLessThan(maxDelta, 0.05)

        print(String(format:
            "[m6-bench] H2 ratios: gather/fullQMM = %.1f, gather/deqGather = %.1f",
            tGather / tFullQMM, tGather / tDeqGather))
    }

    /// H4 — module-level attribution at real shapes (warm): one quantized MoE block
    /// (router + gather_qmm experts + F16 shared expert), one quantized attention layer
    /// (cached path, 128 committed KV), the F16 lm_head, all on a 32-token active block.
    func testModuleAttribution() throws {
        try skipUnlessOptedIn()
        MLXRandom.seed(13)
        let x = MLXRandom.normal([1, Self.T, Self.H]).asType(.float16)
        eval(x)

        // MoE block, production quantization (SwitchLinear 4-bit, shared expert F16).
        let moe = LLaDA2SparseMoEBlock(
            hiddenSize: Self.H, moeIntermediateSize: Self.I, numExperts: Self.E,
            numSharedExperts: 1, numExpertsPerTok: Self.K, nGroup: 8, topkGroup: 4,
            routedScalingFactor: 2.5)
        MLXNN.quantize(model: moe, groupSize: 64, bits: 4, mode: .affine,
                       filter: { _, module in module is SwitchLinear })
        eval(moe)
        _ = time("MoE block [1, 32, 2048] (quantized, warm)") { moe(x) }

        // Router alone (FP32 matmul + sigmoid + group top-k on [32, 2048]).
        let gate = LLaDA2MoEGate(
            hiddenSize: Self.H, numExperts: Self.E, numExpertsPerTok: Self.K,
            nGroup: 8, topkGroup: 4, routedScalingFactor: 2.5)
        eval(gate)
        _ = time("router alone [32, 2048]") {
            let (i, w, _) = gate(x.reshaped(Self.T, Self.H))
            return concatenated([i.asType(.float32), w], axis: -1)
        }

        // lm_head F16 [157184, 2048] on the active block.
        let lmHead = Linear(Self.H, 157_184, bias: false)
        eval(lmHead)
        _ = time("lm_head [1, 32, 2048] → [1, 32, 157184] (F16)") {
            lmHead(x).asType(.float32)
        }

        // Dense FFN (layer 0 shape, intermediate 5120, quantized).
        let dense = LLaDA2MLP(hiddenSize: Self.H, intermediateSize: 5120)
        MLXNN.quantize(model: dense, groupSize: 64, bits: 4, mode: .affine,
                       filter: { _, module in module is Linear })
        eval(dense)
        _ = time("dense FFN (layer-0 shape, quantized)") { dense(x) }
    }
}
