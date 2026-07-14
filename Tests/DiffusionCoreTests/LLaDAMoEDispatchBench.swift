import XCTest
import MLX
import MLXNN
import MLXRandom
import Darwin
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
    func testGatherQMMVariantsMini() throws {
        try skipUnlessOptedIn()
        print("==== BENCHMARKING MINI SHAPES ====")
        try runGatherQMMVariants(E: 256, H: 2048, I: 512, T: 32, K: 8, modelLabel: "Mini")
    }

    func testGatherQMMVariantsFlash() throws {
        try skipUnlessOptedIn()
        print("==== BENCHMARKING FLASH SHAPES ====")
        try runGatherQMMVariants(E: 256, H: 4096, I: 1024, T: 32, K: 8, modelLabel: "Flash")
    }

    private func runGatherQMMVariants(E: Int, H: Int, I: Int, T: Int, K: Int, modelLabel: String) throws {
        let freeMemBefore = freeMemoryMB()
        let swapBefore = swapUsedMB()
        let thermalBefore = thermalStateName()

        MLXRandom.seed(11)
        let w = MLXRandom.normal([E, I, H]) * 0.05
        let (wq, scales, biases) = MLX.quantized(w, groupSize: 64, bits: 4)
        let wDeq = dequantized(
            wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
        let x = MLXRandom.normal([T, 1, 1, H]).asType(.float16)
        let indices = MLXRandom.randInt(0 ..< Int32(E), [T, K])
        if let biases {
            eval(wq, scales, biases, wDeq, x, indices)
        } else {
            eval(wq, scales, wDeq, x, indices)
        }

        let tGather = time("\(modelLabel) - gatherQuantizedMM (production path)") {
            gatherQuantizedMM(
                x, wq, scales: scales, biases: biases,
                rhsIndices: indices, transpose: true, groupSize: 64, bits: 4)
        }
        let tDeqGather = time("\(modelLabel) - dequantized + gatherMM") {
            gatherMM(x.asType(wDeq.dtype), wDeq.swappedAxes(-1, -2), rhsIndices: indices)
        }
        let tFullQMM = time("\(modelLabel) - dense qmm over ALL experts (upper bound)") {
            quantizedMM(
                x.reshaped(T, H),
                wq.reshaped(E * I, H / 8),
                scales: scales.reshaped(E * I, H / 64),
                biases: biases!.reshaped(E * I, H / 64),
                transpose: true, groupSize: 64, bits: 4)
        }

        // Equivalence gate (i) vs (ii): same math up to fp16 accumulation.
        let a = gatherQuantizedMM(
            x, wq, scales: scales, biases: biases,
            rhsIndices: indices, transpose: true, groupSize: 64, bits: 4)
        let b = gatherMM(x.asType(wDeq.dtype), wDeq.swappedAxes(-1, -2), rhsIndices: indices)
        let maxDelta = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        print("[\(modelLabel)-bench] gather vs dequantized-gather max |Δ| = \(maxDelta)")
        XCTAssertLessThan(maxDelta, 0.05)

        print(String(format:
            "[\(modelLabel)-bench] ratios: gather/fullQMM = %.1f, gather/deqGather = %.1f",
            tGather / tFullQMM, tGather / tDeqGather))

        let freeMemAfter = freeMemoryMB()
        let swapAfter = swapUsedMB()
        let thermalAfter = thermalStateName()
        let swapGrowth = swapAfter - swapBefore
        let isValid = swapGrowth <= 256.0 && freeMemBefore >= 1024.0 && (thermalBefore == "nominal" || thermalBefore == "fair")
        print(String(format: "Telemetry - swapBefore: %.1fMB, swapAfter: %.1fMB, swapGrowth: %.1fMB, freeMemBefore: %.1fMB, thermalBefore: %@, thermalAfter: %@, envValid: %@",
            swapBefore, swapAfter, swapGrowth, freeMemBefore, thermalBefore, thermalAfter, isValid ? "true" : "false"))
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

    func testOccupancySweep() throws {
        try skipUnlessOptedIn()
        print("==== OCCUPANCY SWEEP: BATCH SIZE T ====")
        let E = 256
        let H = 4096
        let I = 1024
        let K = 8
        
        for T in [32, 128, 512, 2048] {
            print("--- Running T = \(T) ---")
            MLXRandom.seed(11)
            let w = MLXRandom.normal([E, I, H]) * 0.05
            let (wq, scales, biases) = MLX.quantized(w, groupSize: 64, bits: 4)
            let x = MLXRandom.normal([T, 1, 1, H]).asType(.float16)
            let indices = MLXRandom.randInt(0 ..< Int32(E), [T, K])
            eval(wq, scales, biases, x, indices)
            
            let tGather = time("T=\(T) - gatherQuantizedMM") {
                gatherQuantizedMM(
                    x, wq, scales: scales, biases: biases,
                    rhsIndices: indices, transpose: true, groupSize: 64, bits: 4)
            }
            let tFullQMM = time("T=\(T) - dense quantizedMM") {
                quantizedMM(
                    x.reshaped(T, H),
                    wq.reshaped(E * I, H / 8),
                    scales: scales.reshaped(E * I, H / 64),
                    biases: biases!.reshaped(E * I, H / 64),
                    transpose: true, groupSize: 64, bits: 4)
            }
            
            let uniqueCount = Set(indices.asArray(Int32.self)).count
            let weightsReadBytes = Double(uniqueCount * I * H) * 0.5
            let scalesBiasesReadBytes = Double(uniqueCount * I * H) / 64.0 * 2.0 * 2.0
            let totalReadMB = (weightsReadBytes + scalesBiasesReadBytes) / 1_048_576.0
            let gatherBW = totalReadMB / (tGather * 1000.0)
            
            let denseReadBytes = Double(E * I * H) * 0.5
            let denseScalesBiasesBytes = Double(E * I * H) / 64.0 * 2.0 * 2.0
            let denseReadMB = (denseReadBytes + denseScalesBiasesBytes) / 1_048_576.0
            let denseBW = denseReadMB / (tFullQMM * 1000.0)
            
            print(String(format: "[occupancy-bench] T=%d: UniqueExperts=%d, Gather=%.2f ms (%.2f GB/s), Dense=%.2f ms (%.2f GB/s)",
                T, uniqueCount, tGather * 1000.0, gatherBW, tFullQMM * 1000.0, denseBW))
        }
    }

    func testDivergenceAB() throws {
        try skipUnlessOptedIn()
        print("==== DIRECT CAUSAL TEST FOR DIVERGENCE (A/B) ====")
        let E = 256
        let H = 4096
        let I = 1024
        let T = 512
        let K = 8
        
        MLXRandom.seed(11)
        let w = MLXRandom.normal([E, I, H]) * 0.05
        let (wq, scales, biases) = MLX.quantized(w, groupSize: 64, bits: 4)
        let x = MLXRandom.normal([T, 1, 1, H]).asType(.float16)
        
        let indicesUnsorted = MLXRandom.randInt(0 ..< Int32(E), [T, K])
        let indicesSorted = sorted(indicesUnsorted, axis: 0)
        eval(wq, scales, biases, x, indicesUnsorted, indicesSorted)
        
        let tUnsorted = time("Unsorted routing") {
            gatherQuantizedMM(
                x, wq, scales: scales, biases: biases,
                rhsIndices: indicesUnsorted, transpose: true, groupSize: 64, bits: 4, sortedIndices: false)
        }
        
        let tSorted = time("Sorted routing") {
            gatherQuantizedMM(
                x, wq, scales: scales, biases: biases,
                rhsIndices: indicesSorted, transpose: true, groupSize: 64, bits: 4, sortedIndices: true)
        }
        
        let uniqueCount = Set(indicesUnsorted.asArray(Int32.self)).count
        let weightsReadBytes = Double(uniqueCount * I * H) * 0.5
        let scalesBiasesReadBytes = Double(uniqueCount * I * H) / 64.0 * 2.0 * 2.0
        let totalReadMB = (weightsReadBytes + scalesBiasesReadBytes) / 1_048_576.0
        
        let bwUnsorted = totalReadMB / (tUnsorted * 1000.0)
        let bwSorted = totalReadMB / (tSorted * 1000.0)
        
        print(String(format: "[A/B-bench] Unsorted: %.2f ms (%.2f GB/s), Sorted: %.2f ms (%.2f GB/s)",
            tUnsorted * 1000.0, bwUnsorted, tSorted * 1000.0, bwSorted))
        print(String(format: "[A/B-bench] Speedup from sorting: %.2fx", tUnsorted / tSorted))
    }

    /// Telemetry helper: Free memory in MB
    private func freeMemoryMB() -> Double {
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let kerr = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        guard kerr == KERN_SUCCESS else { return 0.0 }
        var pageSize: Int = 0
        var sizeSize = MemoryLayout<Int>.size
        sysctlbyname("hw.pagesize", &pageSize, &sizeSize, nil, 0)
        return Double(stats.free_count) * Double(pageSize) / 1_048_576
    }

    /// Telemetry helper: Swap used in MB
    private func swapUsedMB() -> Double {
        var usage = xsw_usage()
        var size = MemoryLayout<xsw_usage>.size
        sysctlbyname("vm.swapusage", &usage, &size, nil, 0)
        return Double(usage.xsu_used) / 1_048_576
    }

    /// Telemetry helper: Thermal state name
    private func thermalStateName() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    func testSegmentedMMSandbox() throws {
        print("--- testSegmentedMMSandbox start ---")
        let a = MLXArray(1...12, [3, 4]).asType(.float32)
        let b = MLXArray(1...8, [4, 2]).asType(.float32)
        print("a:\n\(a)")
        print("b:\n\(b)")
        let segments = MLXArray([0, 2, 2, 4] as [Int32], [2, 2])
        print("segments:\n\(segments)")
        let result = segmentedMM(a, b, segments: segments)
        print("result:\n\(result)")
        print("result shape: \(result.shape)")
        print("--- testSegmentedMMSandbox end ---")
    }
}
