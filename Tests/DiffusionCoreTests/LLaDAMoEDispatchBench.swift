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

    /// Step 2 of the kernel plan — the fused `MoEGatherQMVRunner` walking skeleton vs the stock
    /// three-`gatherQuantizedMM` SwitchGLU at real Mini shapes, same quantized weights, SwitchGLU-
    /// level A/B (both arms include the MLX-side SwiGLU combine; the k-weighted sum happens
    /// outside SwitchGLU in both). Weights fp16, x float32 — the production dtype pairing (the
    /// Step 5a trace shows the production dispatch is `..._float_...`).
    func testFusedQMVRunnerMini() throws {
        try skipUnlessOptedIn()
        try runFusedQMVComparison(E: Self.E, H: Self.H, I: Self.I, T: Self.T, K: Self.K, label: "Mini")
    }

    /// Same A/B at Flash shapes (H=4096, I=1024 — both still multiples of 512, so
    /// `MoEGatherQMVRunner.isEligible` holds and the fused path is taken, not a silent
    /// fallback). Flash's weights are ~4x Mini's (still fine on this host). Step 8 insurance:
    /// confirms the fused kernel's ranking isn't Mini-shape-specific before trusting it
    /// end-to-end on a model that uses these dims.
    func testFusedQMVRunnerFlash() throws {
        try skipUnlessOptedIn()
        try runFusedQMVComparison(E: Self.E, H: 4096, I: 1024, T: Self.T, K: Self.K, label: "Flash")
    }

    /// Step 2 of the kernel plan — the fused `MoEGatherQMVRunner` walking skeleton vs the stock
    /// three-`gatherQuantizedMM` SwitchGLU, same quantized weights, SwitchGLU-level A/B (both
    /// arms include the MLX-side SwiGLU combine; the k-weighted sum happens outside SwitchGLU
    /// in both). Weights fp16, x float32 — the production dtype pairing (the Step 5a trace
    /// shows the production dispatch is `..._float_...`).
    private func runFusedQMVComparison(E: Int, H: Int, I: Int, T: Int, K: Int, label: String) throws {
        let freeMemBefore = freeMemoryMB()
        let swapBefore = swapUsedMB()
        let thermalBefore = thermalStateName()

        MLXRandom.seed(23)
        let glu = SwitchGLU(hiddenSize: H, intermediateSize: I, numExperts: E)
        let params: [String: MLXArray] = [
            "gate_proj.weight": (MLXRandom.normal([E, I, H]) * 0.05)
                .asType(.float16),
            "up_proj.weight": (MLXRandom.normal([E, I, H]) * 0.05)
                .asType(.float16),
            "down_proj.weight": (MLXRandom.normal([E, H, I]) * 0.05)
                .asType(.float16),
        ]
        try glu.update(parameters: ModuleParameters.unflattened(params), verify: .all)
        MLXNN.quantize(
            model: glu, groupSize: 64, bits: 4, mode: .affine,
            filter: { _, module in module is SwitchLinear })
        let x = MLXRandom.normal([T, H])  // float32, per production
        let indices = MLXRandom.randInt(0 ..< Int32(E), [T, K])
        eval(glu, x, indices)

        // reps 50: at ~1.3 ms/op the default 10 reps is inside cross-process drift; kernel
        // iteration needs the within-process ratio sharp.
        let tStock = time("\(label) - SwitchGLU stock (3x gatherQuantizedMM)", warmup: 3, reps: 50) {
            MoEFusedQMVConfig.enabled = false
            return glu(x, indices: indices)
        }
        let dispatchesBefore = MoEFusedQMVConfig.dispatchCount
        let tFused = time("\(label) - SwitchGLU fused MoEGatherQMVRunner", warmup: 3, reps: 50) {
            MoEFusedQMVConfig.enabled = true
            return glu(x, indices: indices)
        }
        MoEFusedQMVConfig.enabled = false
        XCTAssertGreaterThan(
            MoEFusedQMVConfig.dispatchCount, dispatchesBefore,
            "fused arm silently fell back to stock — timings are meaningless")

        MoEFusedQMVConfig.enabled = false
        let stock = glu(x, indices: indices)
        MoEFusedQMVConfig.enabled = true
        let fused = glu(x, indices: indices)
        MoEFusedQMVConfig.enabled = false
        let maxDelta = abs(fused.asType(.float32) - stock.asType(.float32)).max().item(Float.self)
        print("[m6-bench] \(label) fused vs stock max |Δ| = \(maxDelta)")
        XCTAssertLessThan(maxDelta, 1e-3)
        print(String(format: "[m6-bench] \(label) fused/stock time ratio = %.3f (stock %.4f s, fused %.4f s)",
            tFused / tStock, tStock, tFused))

        let swapAfter = swapUsedMB()
        let swapGrowth = swapAfter - swapBefore
        let isValid = swapGrowth <= 256.0 && freeMemBefore >= 1024.0
            && (thermalBefore == "nominal" || thermalBefore == "fair")
        print(String(format:
            "Telemetry - swapGrowth: %.1fMB, freeMemBefore: %.1fMB, thermalBefore: %@, thermalAfter: %@, envValid: %@",
            swapGrowth, freeMemBefore, thermalBefore, thermalStateName(),
            isValid ? "true" : "false"))
    }

    /// Step 5a — produce Xcode-openable `.gputrace` captures of the production
    /// `gatherQuantizedMM` and its FP16 `gatherMM` sibling at Mini shapes, T ∈ {32, 64}.
    /// This gathers the last pre-kernel diagnostic (`gather_qmm_handoff.md` §3/§10): what
    /// caps the 4-bit gather's occupancy at T=32.
    ///
    /// **Each arm gets its OWN trace file, built from inputs never evaluated before the
    /// capture window.** Two earlier attempts failed the same way: sharing one window across
    /// both `eval()` calls, and then even isolating each arm in its own window, both left the
    /// 4-bit trace essentially empty (40 KB vs the FP16 arm's 2.7 GB) — calling `gatherQuantizedMM`
    /// again on inputs already evaluated during warmup produced zero new GPU dispatches.
    /// The fix: warm up the pipelines on THROWAWAY inputs, then build a completely FRESH set
    /// of inputs and call each op EXACTLY ONCE, ever, inside its capture window — there is
    /// nothing for MLX to have already computed.
    ///
    /// Requires BOTH the opt-in gate and Apple's capture env var:
    ///   MTL_CAPTURE_ENABLED=1 NEODIFFUSION_GPU_CAPTURE=1 \
    ///     swift test --filter LLaDAMoEDispatchBench/testCaptureGatherQMMTrace
    /// Output: `$NEODIFFUSION_CAPTURE_DIR` (default `<cwd>/scratch/captures`)/gather_t{32,64}_{4bit,fp16}.gputrace.
    func testCaptureGatherQMMTrace() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_GPU_CAPTURE"] == "1",
            "opt-in: set NEODIFFUSION_GPU_CAPTURE=1 (writes .gputrace files)")
        // GPU.startCapture silently no-ops / errors without this Apple env gate — fail loud.
        XCTAssertEqual(
            ProcessInfo.processInfo.environment["MTL_CAPTURE_ENABLED"], "1",
            "GPU.startCapture requires MTL_CAPTURE_ENABLED=1 in the environment")

        let dir = ProcessInfo.processInfo.environment["NEODIFFUSION_CAPTURE_DIR"]
            ?? (FileManager.default.currentDirectoryPath + "/scratch/captures")
        try FileManager.default.createDirectory(
            atPath: dir, withIntermediateDirectories: true)

        for T in [32, 64] {
            try captureGatherQMM(E: Self.E, H: Self.H, I: Self.I, T: T, K: Self.K, dir: dir)
        }
    }

    private typealias GatherInputs = (
        wq: MLXArray, scales: MLXArray, biases: MLXArray?, wDeq: MLXArray,
        x: MLXArray, indices: MLXArray
    )

    /// Builds one fresh, immediately-evaluated set of Mini-shape inputs (same construction
    /// as `runGatherQMMVariants`). Each call produces genuinely new arrays — never reuse a
    /// previous call's arrays across a warmup/capture boundary (see `captureGatherQMM`).
    private func buildGatherInputs(E: Int, H: Int, I: Int, T: Int, K: Int, seed: UInt64) -> GatherInputs {
        MLXRandom.seed(seed)
        let w = MLXRandom.normal([E, I, H]) * 0.05
        let (wq, scales, biases) = MLX.quantized(w, groupSize: 64, bits: 4)
        // Explicit fp16 cast: `dequantized(...)` inherits w's dtype (float32 here), but the
        // FP16 arm must genuinely run in fp16 to match the production Case-B comparison
        // (`gather_qmm_handoff.md` §12 anchors its 433 GB/s figure to FP16, not FP32).
        let wDeq = dequantized(wq, scales: scales, biases: biases, groupSize: 64, bits: 4)
            .asType(.float16)
        let x = MLXRandom.normal([T, 1, 1, H]).asType(.float16)
        let indices = MLXRandom.randInt(0 ..< Int32(E), [T, K])
        if let biases {
            eval(wq, scales, biases, wDeq, x, indices)
        } else {
            eval(wq, scales, wDeq, x, indices)
        }
        return (wq, scales, biases, wDeq, x, indices)
    }

    private func quantizedGather(_ w: GatherInputs) -> MLXArray {
        gatherQuantizedMM(
            w.x, w.wq, scales: w.scales, biases: w.biases,
            rhsIndices: w.indices, transpose: true, groupSize: 64, bits: 4)
    }
    private func fp16Gather(_ w: GatherInputs) -> MLXArray {
        gatherMM(w.x.asType(w.wDeq.dtype), w.wDeq.swappedAxes(-1, -2), rhsIndices: w.indices)
    }

    /// Captures the 4-bit `gatherQuantizedMM` and the FP16 `gatherMM` sibling into two
    /// SEPARATE `.gputrace` files, each built from inputs that have NEVER been evaluated
    /// before entering the capture window (see the type doc comment for why).
    private func captureGatherQMM(E: Int, H: Int, I: Int, T: Int, K: Int, dir: String) throws {
        // 1. Warm up BOTH pipelines (JIT-compile) on throwaway inputs, run twice, entirely
        //    outside any capture window and never touched again.
        let warm = buildGatherInputs(E: E, H: H, I: I, T: T, K: K, seed: 11)
        for _ in 0 ..< 2 { eval(quantizedGather(warm)); eval(fp16Gather(warm)) }

        // Expected buffer sizes, printed so a trace can be sanity-checked from the outside:
        // 4-bit packed weight ≈ E·I·H·bits/8 bytes; FP16 weight ≈ E·I·H·2 bytes (4× larger).
        let packedMB = Double(E * I * H * 4 / 8) / 1_048_576
        let fp16MB = Double(E * I * H * 2) / 1_048_576
        print("[capture] T=\(T) expected weight-buffer sizes: 4bit≈\(Int(packedMB)) MiB (+ scales/biases), fp16≈\(Int(fp16MB)) MiB (no scales/biases)")

        // 2. Capture: a completely fresh set of inputs per arm, called exactly once, ever.
        // `.sum().item(...)` (not just `eval`) forces an actual CPU-blocking readback, so the
        // capture window cannot close before the GPU work is fully submitted and complete —
        // `eval()` alone left the 4-bit trace empty even with fresh inputs, suggesting eval()
        // does not guarantee a host sync boundary for this op under an open capture.
        let cap4bit = buildGatherInputs(E: E, H: H, I: I, T: T, K: K, seed: 101)
        try captureOneArm(label: "4bit", dir: dir, T: T) {
            _ = quantizedGather(cap4bit).sum().item(Float.self)
        }

        let capFp16 = buildGatherInputs(E: E, H: H, I: I, T: T, K: K, seed: 102)
        try captureOneArm(label: "fp16", dir: dir, T: T) {
            _ = fp16Gather(capFp16).sum().item(Float.self)
        }

        print("[capture] T=\(T) -> \(dir)/gather_t\(T)_{4bit,fp16}.gputrace  (Mini E=\(E) H=\(H) I=\(I) K=\(K))")
    }

    /// Captures exactly one call of `body` into its own trace file — no warmup here, `body`
    /// must already be operating on freshly-built, never-evaluated inputs (see caller).
    private func captureOneArm(label: String, dir: String, T: Int, body: () -> Void) throws {
        // startCapture requires the destination not already exist (it is a bundle directory).
        let path = "\(dir)/gather_t\(T)_\(label).gputrace"
        if FileManager.default.fileExists(atPath: path) {
            try FileManager.default.removeItem(atPath: path)
        }
        let url = URL(fileURLWithPath: path)

        MLX.GPU.startCapture(url: url)
        body()
        MLX.GPU.stopCapture(url: url)
    }

    /// Step 5a fallback — the isolated `gatherQuantizedMM` capture above produced only a
    /// 40 KB (empty) trace for the 4-bit arm across three independent fixes (separate trace
    /// files, fresh never-evaluated inputs, forced host-sync `.item()` readback) — ruling out
    /// timing/caching/sync as the cause and pointing at something structural in how an
    /// ISOLATED `gatherQuantizedMM` call captures. This captures a REAL MoE block forward
    /// instead — the same construction `testModuleAttribution` uses, extensively validated by
    /// wall-clock timing (Step 4 / `gather_qmm_handoff.md`) to dispatch `gatherQuantizedMM` for
    /// several ms of real, measured GPU time. Same env gate as `testCaptureGatherQMMTrace`.
    func testCaptureMoEBlockTrace() throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_GPU_CAPTURE"] == "1",
            "opt-in: set NEODIFFUSION_GPU_CAPTURE=1 (writes .gputrace files)")
        XCTAssertEqual(
            ProcessInfo.processInfo.environment["MTL_CAPTURE_ENABLED"], "1",
            "GPU.startCapture requires MTL_CAPTURE_ENABLED=1 in the environment")

        let dir = ProcessInfo.processInfo.environment["NEODIFFUSION_CAPTURE_DIR"]
            ?? (FileManager.default.currentDirectoryPath + "/scratch/captures")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)

        let moe = LLaDA2SparseMoEBlock(
            hiddenSize: Self.H, moeIntermediateSize: Self.I, numExperts: Self.E,
            numSharedExperts: 1, numExpertsPerTok: Self.K, nGroup: 8, topkGroup: 4,
            routedScalingFactor: 2.5)
        MLXNN.quantize(model: moe, groupSize: 64, bits: 4, mode: .affine,
                       filter: { _, module in module is SwitchLinear })
        eval(moe)

        // Warm up (JIT-compile) on throwaway input, outside the capture window.
        MLXRandom.seed(21)
        let warmX = MLXRandom.normal([1, Self.T, Self.H]).asType(.float16)
        eval(warmX)
        for _ in 0 ..< 2 { _ = moe(warmX).sum().item(Float.self) }

        // Capture: a fresh, never-before-evaluated input, forced sync readback inside window.
        MLXRandom.seed(201)
        let capX = MLXRandom.normal([1, Self.T, Self.H]).asType(.float16)
        eval(capX)

        let path = "\(dir)/moe_block_t32.gputrace"
        if FileManager.default.fileExists(atPath: path) {
            try FileManager.default.removeItem(atPath: path)
        }
        let url = URL(fileURLWithPath: path)
        MLX.GPU.startCapture(url: url)
        _ = moe(capX).sum().item(Float.self)
        MLX.GPU.stopCapture(url: url)

        print("[capture] MoE block forward (production quantized, T=32) -> \(path)")
        // Which MoE dispatch path did the captured forward take? (NEODIFFUSION_FUSED_QMV=1
        // seeds the flag, but eligibility can silently fall back to stock — echo the proof.)
        print("[capture] fused gather-QMV dispatches this process: \(MoEFusedQMVConfig.dispatchCount)")
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

    private func runSortingOverheadSweepHelper(E: Int, H: Int, I: Int, K: Int, modelLabel: String) {
        print("\n--- Evaluating \(modelLabel) Shapes (E=\(E), H=\(H), I=\(I), K=\(K)) ---")
        MLXRandom.seed(11)
        let w = MLXRandom.normal([E, I, H]) * 0.05
        let (wq, scales, biases) = MLX.quantized(w, groupSize: 64, bits: 4)
        if let biases {
            eval(wq, scales, biases)
        } else {
            eval(wq, scales)
        }

        let TSweep = [32, 48, 64, 80, 96, 128, 256, 512, 1024, 2048]
        for T in TSweep {
            let x = MLXRandom.normal([T, 1, 1, H]).asType(.float16)
            let indicesUnsorted = MLXRandom.randInt(0 ..< Int32(E), [T, K])
            let indicesSorted = sorted(indicesUnsorted.flattened()).reshaped([T, K])
            
            if let biases {
                eval(x, indicesUnsorted, indicesSorted, wq, scales, biases)
            } else {
                eval(x, indicesUnsorted, indicesSorted, wq, scales)
            }

            // Warmup
            let _ = gatherQuantizedMM(x, wq, scales: scales, biases: biases, rhsIndices: indicesUnsorted, transpose: true, groupSize: 64, bits: 4, sortedIndices: false)
            let _ = gatherQuantizedMM(x, wq, scales: scales, biases: biases, rhsIndices: indicesSorted, transpose: true, groupSize: 64, bits: 4, sortedIndices: true)
            
            let indicesFlat = indicesUnsorted.flattened()
            let _ = argSort(indicesFlat, axis: -1)
            eval(x)

            // 1. Measure unsorted execution
            let tUnsorted = time("  gatherQuantizedMM (unsorted)") {
                gatherQuantizedMM(x, wq, scales: scales, biases: biases, rhsIndices: indicesUnsorted, transpose: true, groupSize: 64, bits: 4, sortedIndices: false)
            }

            // 2. Measure sorted execution (pure GPU matmul)
            let tSorted = time("  gatherQuantizedMM (sorted)") {
                gatherQuantizedMM(x, wq, scales: scales, biases: biases, rhsIndices: indicesSorted, transpose: true, groupSize: 64, bits: 4, sortedIndices: true)
            }

            // 3. Measure GPU sorting overhead
            let tSort = time("  GPU sorting overhead (argSort)") {
                let sortOrder = argSort(indicesFlat, axis: -1)
                let sortedIndicesArray = indicesFlat[sortOrder]
                let tokenIndices = sortOrder / Int32(K)
                return concatenated([sortedIndicesArray, tokenIndices])
            }

            let netTime = tSorted + tSort
            let netSpeedup = tUnsorted / netTime

            print(String(format: "  T=%4d | Unsorted: %7.4f ms | Sorted Matmul: %7.4f ms | Sort Overhead: %7.4f ms | Net Time: %7.4f ms (Net Speedup: %.3fx)",
                         T, tUnsorted * 1000.0, tSorted * 1000.0, tSort * 1000.0, netTime * 1000.0, netSpeedup))
        }
    }

    func testSortingOverheadSweep() throws {
        guard ProcessInfo.processInfo.environment["NEODIFFUSION_M6_BENCH"] != nil else {
            print("Skipping testSortingOverheadSweep (set NEODIFFUSION_M6_BENCH=1 to run)")
            return
        }

        print("\n=== EXPERIMENT: Quantized MoE GPU Sorting Overhead & Break-Even Sweep ===")
        let freeMemBefore = freeMemoryMB()
        let swapBefore = swapUsedMB()
        let thermalBefore = thermalStateName()

        print(String(format: "envValid pre-conditions: Free Memory = %.1f MB (target >= 1024), Swap Used = %.1f MB, Thermal = %@", freeMemBefore, swapBefore, thermalBefore))

        // Sweep Mini Shapes (H=2048, I=512)
        runSortingOverheadSweepHelper(E: 256, H: 2048, I: 512, K: 8, modelLabel: "Mini")

        // Sweep Flash Shapes (H=4096, I=1024)
        runSortingOverheadSweepHelper(E: 256, H: 4096, I: 1024, K: 8, modelLabel: "Flash")

        let freeMemAfter = freeMemoryMB()
        let swapAfter = swapUsedMB()
        let thermalAfter = thermalStateName()
        let envValid = (swapAfter - swapBefore <= 256.0) && (freeMemBefore >= 1024.0) && (thermalAfter == "nominal" || thermalAfter == "fair")

        print(String(format: "\nenvValid post-conditions: Free Memory = %.1f MB, Swap Growth = %.1f MB, Thermal = %@, envValid = %@", freeMemAfter, swapAfter - swapBefore, thermalAfter, envValid ? "TRUE" : "FALSE"))
    }

    func testSegmentedMMComparison() throws {
        guard ProcessInfo.processInfo.environment["NEODIFFUSION_M6_BENCH"] != nil else {
            print("Skipping testSegmentedMMComparison (set NEODIFFUSION_M6_BENCH=1 to run)")
            return
        }

        print("\n=== EXPERIMENT: segmentedMM vs gatherMM Comparison (Unquantized) ===")
        let E = 256
        let H = 4096
        let I = 1024
        let K = 8

        let freeMemBefore = freeMemoryMB()
        let swapBefore = swapUsedMB()
        let thermalBefore = thermalStateName()

        print(String(format: "envValid pre-conditions: Free Memory = %.1f MB (target >= 1024), Swap Used = %.1f MB, Thermal = %@", freeMemBefore, swapBefore, thermalBefore))

        let TSweep = [32, 48, 64, 80, 96, 128, 256, 512, 1024, 2048]
        for T in TSweep {
            print("\nEvaluating T = \(T)")
            MLXRandom.seed(42)
            
            let w = MLXRandom.normal([E, H, I]).asType(.float16)
            let x = MLXRandom.normal([T, 1, 1, H]).asType(.float16)
            
            let indicesUnsorted = MLXRandom.randInt(0 ..< Int32(E), [T, K])
            let indicesSorted = sorted(indicesUnsorted.flattened()).reshaped([T, K])
            eval(w, x, indicesUnsorted, indicesSorted)

            // Warmup
            let _ = gatherMM(x, w, rhsIndices: indicesUnsorted)
            let _ = gatherMM(x, w, rhsIndices: indicesSorted, sortedIndices: true)
            eval(w, x)

            // 1. Benchmark gatherMM (unsorted)
            let tUnsorted = time("  gatherMM (unsorted)") {
                gatherMM(x, w, rhsIndices: indicesUnsorted)
            }

            // 2. Benchmark gatherMM (sorted)
            let tSorted = time("  gatherMM (sorted)") {
                gatherMM(x, w, rhsIndices: indicesSorted, sortedIndices: true)
            }

            // 3. Benchmark segmentedMM
            let sortedIndicesCPU = indicesSorted.asArray(Int32.self)
            var segmentRanges: [Int32] = []
            var currentExpert = Int32(-1)
            var currentStart = Int32(0)
            for i in 0..<sortedIndicesCPU.count {
                let exp = sortedIndicesCPU[i]
                if exp != currentExpert {
                    if currentExpert != -1 {
                        segmentRanges.append(currentStart)
                        segmentRanges.append(Int32(i))
                    }
                    currentExpert = exp
                    currentStart = Int32(i)
                }
            }
            if currentExpert != -1 {
                segmentRanges.append(currentStart)
                segmentRanges.append(Int32(sortedIndicesCPU.count))
            }
            
            let numSegments = segmentRanges.count / 2
            let segments = MLXArray(segmentRanges, [numSegments, 2])
            
            let T_total = T * K
            let segA = MLXRandom.normal([I, T_total]).asType(.float16)
            let segB = MLXRandom.normal([T_total, H]).asType(.float16)
            eval(segA, segB, segments)
            
            // Warmup segmentedMM
            let _ = segmentedMM(segA, segB, segments: segments)
            eval(segA, segB)

            let tSegmented = time("  segmentedMM primitive") {
                segmentedMM(segA, segB, segments: segments)
            }

            print(String(format: "  Unsorted: %.4f ms", tUnsorted * 1000.0))
            print(String(format: "  Sorted:   %.4f ms (speedup: %.2fx vs unsorted)", tSorted * 1000.0, tUnsorted / tSorted))
            print(String(format: "  segmentedMM primitive: %.4f ms", tSegmented * 1000.0))
        }

        let freeMemAfter = freeMemoryMB()
        let swapAfter = swapUsedMB()
        let thermalAfter = thermalStateName()
        let envValid = (swapAfter - swapBefore <= 256.0) && (freeMemBefore >= 1024.0) && (thermalAfter == "nominal" || thermalAfter == "fair")

        print(String(format: "\nenvValid post-conditions: Free Memory = %.1f MB, Swap Growth = %.1f MB, Thermal = %@, envValid = %@", freeMemAfter, swapAfter - swapBefore, thermalAfter, envValid ? "TRUE" : "FALSE"))
    }

    private func runTiersBench(tiers: [(String, [Int])], H: Int, I: Int, modelLabel: String) {
        print("\n--- Running Production Tiers Bench for \(modelLabel) Shapes (H=\(H), I=\(I)) ---")
        
        let moeBlock = LLaDA2SparseMoEBlock(
            hiddenSize: H,
            moeIntermediateSize: I,
            numExperts: 256,
            numSharedExperts: 0,
            numExpertsPerTok: 8,
            nGroup: 1,
            topkGroup: 1,
            routedScalingFactor: 1.0
        )
        
        MLXNN.quantize(
            model: moeBlock,
            filter: { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                if module is SwitchLinear {
                    return (64, 4, .affine)
                }
                return nil
            }
        )
        
        for (tierLabel, Ts) in tiers {
            print("\n  [\(tierLabel)]")
            for T in Ts {
                let x = MLXRandom.normal([T, H]).asType(.float16)
                eval(x)
                
                // Warmup
                let _ = moeBlock(x)
                eval(x)
                
                // Measure execution
                let tBlock = time("    T=\(T)") {
                    moeBlock(x)
                }
                print(String(format: "    T=%4d: %.4f ms", T, tBlock * 1000.0))
            }
        }
    }

    func testProductionTiersDistribution() throws {
        guard ProcessInfo.processInfo.environment["NEODIFFUSION_M6_BENCH"] != nil else {
            print("Skipping testProductionTiersDistribution (set NEODIFFUSION_M6_BENCH=1 to run)")
            return
        }
        
        print("\n=== EXPERIMENT: Production Tiers Distribution Bench ===")
        // Tier 1: Short chat turns: T=32, T=64, T=128
        // Tier 2: Multi-turn chat: T=512, T=1024
        // Tier 3: Coding assistant: T=2048, T=4096
        
        let tiers = [
            ("Tier 1: Short chat turns", [32, 64, 128]),
            ("Tier 2: Multi-turn chat with history", [512, 1024]),
            ("Tier 3: Coding assistant sessions", [2048, 4096])
        ]
        
        runTiersBench(tiers: tiers, H: 2048, I: 512, modelLabel: "Mini")
        runTiersBench(tiers: tiers, H: 4096, I: 1024, modelLabel: "Flash")
    }
}

