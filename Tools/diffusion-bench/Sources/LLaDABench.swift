import Foundation
import MLX
import DiffusionCore
import DiffusionModel
import DiffusionGeneration

// LLaDA mode — the phase-2 M6 metric set (phase-2 §4 M6, handoff §3):
// TPS, TPF (honest = tokens/forwards-evaluated incl. speculative overshoot + capture
// forwards; logical = tokens/denoising-steps), steps/block, post-steps/block, sync-point
// count, peak memory, per-phase wall-clock (prefill/denoise/commit). JSONL per
// (arm, suite, prompt, run); cross-run variance gate per arm (<5%, the M6 acceptance).
//
// Notes carried from M5 (handoff §4): `stepsPerBlock` counts the budget-break iteration
// the reference does not (+1 on budget-exit blocks); the 4-bit dev artefact is task-level
// only, never parity (gotcha 5) — the numbers frozen here are the Phase-3 *dev* baseline,
// re-freeze on the Studio.

// MARK: - Benchmark-environment telemetry (m6-logbook Findings 1/7)
//
// The ~50 s/forward failure mode is machine memory pressure (paging), and cross-process
// totals drift +6–10% with thermal state — so every JSONL row carries the environment it
// was measured under, and an `envValid` label applying the frozen validity rule (label,
// never reject: contaminated rows stay in the record, marked).

/// vm.swapusage "used" in MB.
func swapUsedMB() -> Double {
    var usage = xsw_usage()
    var size = MemoryLayout<xsw_usage>.size
    guard sysctlbyname("vm.swapusage", &usage, &size, nil, 0) == 0 else { return -1 }
    return Double(usage.xsu_used) / 1_048_576
}

/// Host free-page memory in MB (the "Pages free" figure — deliberately conservative:
/// macOS keeps this low under healthy load too, hence the 1 GB validity threshold).
func freeMemoryMB() -> Double {
    var stats = vm_statistics64()
    var count = mach_msg_type_number_t(
        MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
    let result = withUnsafeMutablePointer(to: &stats) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
        }
    }
    guard result == KERN_SUCCESS else { return -1 }
    var pageSize: Int = 0
    var sizeSize = MemoryLayout<Int>.size
    sysctlbyname("hw.pagesize", &pageSize, &sizeSize, nil, 0)
    return Double(stats.free_count) * Double(pageSize) / 1_048_576
}

/// ProcessInfo thermal state as a stable string.
func thermalStateName() -> String {
    switch ProcessInfo.processInfo.thermalState {
    case .nominal: return "nominal"
    case .fair: return "fair"
    case .serious: return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func sysctlString(_ name: String) -> String {
    var size = 0
    sysctlbyname(name, nil, &size, nil, 0)
    guard size > 0 else { return "unknown" }
    var buf = [CChar](repeating: 0, count: size)
    sysctlbyname(name, &buf, &size, nil, 0)
    return String(cString: buf)
}

/// Snapshot of the environment around one generation.
struct EnvSnapshot: Codable {
    let swapUsedMBBefore: Double
    let swapUsedMBAfter: Double
    let freeMemoryMBBefore: Double
    let thermalBefore: String
    let thermalAfter: String

    /// Frozen validity rule (m6-logbook §5 rule 5): a timed row is environment-valid iff
    /// swap growth ≤ 256 MB, ≥ 1 GB free pages at start, and thermal ≤ fair at start.
    var isValid: Bool {
        swapUsedMBAfter - swapUsedMBBefore <= 256
            && freeMemoryMBBefore >= 1024
            && (thermalBefore == "nominal" || thermalBefore == "fair")
    }

    var note: String {
        var parts: [String] = []
        if swapUsedMBAfter - swapUsedMBBefore > 256 {
            parts.append(String(format: "swap +%.0f MB", swapUsedMBAfter - swapUsedMBBefore))
        }
        if freeMemoryMBBefore < 1024 {
            parts.append(String(format: "free %.0f MB", freeMemoryMBBefore))
        }
        if thermalBefore != "nominal" && thermalBefore != "fair" {
            parts.append("thermal \(thermalBefore)")
        }
        return parts.joined(separator: ", ")
    }
}

/// True until the process's first generation completes — that generation carries the
/// ~19 s per-process warmup (m6-logbook Finding 2) and must be excluded from
/// steady-state comparisons. The bench is single-threaded top-level code
/// (generations run strictly sequentially), hence `nonisolated(unsafe)`.
nonisolated(unsafe) var llaDAProcessFirstGeneration = true

struct LLaDAPromptCase: Codable {
    let id: String
    let user: String
}

struct LLaDAPromptSuite: Codable {
    let name: String
    let prompts: [LLaDAPromptCase]
}

struct LLaDAArm {
    let name: String
    let mode: GenerationParams.Mode
    let cached: Bool
    let mask: BlockDiffusionMask.Semantics
}

struct LLaDARunResult: Codable {
    let arm: String
    let run: Int
    let suite: String
    let promptId: String
    let mode: String
    // Explicit thresholds, not just the mode label (modes may be retuned; rows must
    // stay self-describing).
    let thresholdMask: Float
    let thresholdEdit: Float
    let cached: Bool
    let mask: String
    let blockLength: Int
    let genLength: Int
    let promptTokens: Int
    let tokensGenerated: Int
    let blocks: Int
    let logicalSteps: Int
    let postSteps: Int
    // Full per-block distributions, not just means (steps/block spans 2–21 by content;
    // a mean hides the shape).
    let stepsPerBlock: [Int]
    let postStepsPerBlock: [Int]
    let forwardsEvaluated: Int
    let syncPoints: Int
    // Trajectory summary (M8 E1): total Γ transfers, total Δ edits, overall mean
    // transferred-token confidence, and the block eos first appeared in (nil = never).
    let transfersTotal: Int
    let editsTotal: Int
    let meanTransferConfidence: Double
    let eosBlockIndex: Int?
    let totalSeconds: Double
    let prefillSeconds: Double
    let denoiseSeconds: Double
    let commitSeconds: Double
    let tps: Double
    let tpfHonest: Double
    let tpfLogical: Double
    let stepsPerBlockMean: Double
    let postStepsPerBlockMean: Double
    let peakMemoryGB: Double
    let speculationK: Int
    let eosEarlyStop: Bool
    // WP-1b MultiBD — ENGINE effective echoes (never CLI inputs; provenance rule F7),
    // plus the dual-phase diagnostics that decide the WP under roadmap §0.1.
    let nBuf: Int
    let tauAdd: Float
    let tauSemi: Float
    let dualActiveSteps: Int
    let activationSteps: [Int]
    let trailingStarvedStepsPerBlock: [Int]
    let singleActiveDenoiseSeconds: Double
    let dualActiveDenoiseSeconds: Double
    // WP-2a speculation — engine effective echoes + width-aware accounting.
    let speculation: String
    let tauSpan: Int
    let acceptedTotal: Int
    let acceptedPerVerifiedStepMean: Double
    let tokensProcessedInForwards: Int
    /// Tokens ÷ (tokensProcessedInForwards / blockLength): the cross-policy decider — plain
    /// TPF-honest misleads when forwards differ in width (32 target vs 64 verifier).
    let tpfWidthCorrected: Double
    let elasticCacheEnabled: Bool
    let elasticGamma: Float
    let elasticBeta: Int
    let elasticStaticBoundary: Int?
    // Warmup / process / environment classification (m6-logbook Findings 1/2/7):
    // the process's first generation carries ~19 s one-off cost; cross-process
    // comparisons carry thermal drift; env fields + validity label per the frozen rule.
    let warmupIncluded: Bool
    let processId: Int
    let host: String
    let env: EnvSnapshot
    let envValid: Bool
    let textPrefix: String
    let date: String
}

func runLLaDABench() async throws {
    let repoRoot = FileManager.default.currentDirectoryPath
    let modelDir = URL(fileURLWithPath: argValue("--model")
        ?? "\(repoRoot)/models/llada2-1-mini-4bit")
    let tokenizerDir = URL(fileURLWithPath: argValue("--tokenizer")
        ?? "\(repoRoot)/models/llada2-1-mini")
    let runs = Int(argValue("--runs") ?? "3") ?? 3
    let cooldown = Int(argValue("--cooldown") ?? "30") ?? 30
    let jsonPath = argValue("--json") ?? "\(repoRoot)/scratch/llada_bench.jsonl"
    let genLength = Int(argValue("--gen-length") ?? "128") ?? 128
    let blockLength = Int(argValue("--block-length") ?? "32") ?? 32
    let speculationK = Int(argValue("--speculation-k") ?? "4") ?? 4
    let eosEarlyStop = !hasFlag("--no-early-stop")
    let instrument = !hasFlag("--no-instrument")
    let maskDiagnostic = hasFlag("--mask-diagnostic")
    
    let elasticCache = hasFlag("--elastic-cache")
    let elasticGamma = Float(argValue("--elastic-gamma") ?? "0.9") ?? 0.9
    let elasticBeta = Int(argValue("--elastic-beta") ?? "16") ?? 16
    let elasticStaticBoundary = argValue("--elastic-static-boundary").flatMap { Int($0) }

    // WP-1b MultiBD (arXiv:2606.29215 Alg. 5). JSONL rows record the ENGINE's effective
    // echoes, not these CLI values (provenance rule, elastic-cache logbook F7).
    let nBuf = Int(argValue("--n-buf") ?? "1") ?? 1
    let tauAdd = Float(argValue("--tau-add") ?? "2.0") ?? 2.0
    let tauSemi = Float(argValue("--tau-semi") ?? "0.9") ?? 0.9
    // Full-text dump for blind quality scoring (JSONL keeps only 160-char prefixes):
    // one JSON line per generation {arm, suite, promptId, run, text, tokens, eosBlock}.
    let dumpTextPath = argValue("--dump-text")

    // WP-2a speculation. JSONL rows record ENGINE effective echoes (rule F7).
    let speculation: GenerationParams.SpeculationKind =
        (argValue("--speculation") == "s2d2") ? .s2d2 : .none
    let tauSpan = Int(argValue("--tau-span") ?? "1") ?? 1
    // Optional Γ/Δ threshold overrides (WP-2a conservative-baseline probe: the S2D2 papers
    // benchmark against τ_M2T=0.95-style decoding; the Q-mode default is 0.7).
    let thresholdMaskOverride = argValue("--threshold-mask").flatMap(Float.init)
    let thresholdEditOverride = argValue("--threshold-edit").flatMap(Float.init)

    // Prompt suites: fixed cases checked into Tools/diffusion-bench/PromptSuites (M6).
    // --prompt TEXT replaces them with a single ad-hoc case.
    var suites: [LLaDAPromptSuite]
    if let text = argValue("--prompt") {
        suites = [LLaDAPromptSuite(
            name: "custom", prompts: [LLaDAPromptCase(id: "custom-0", user: text)])]
    } else {
        let wanted = (argValue("--suites") ?? "chat,reasoning,code")
            .split(separator: ",").map(String.init)
        suites = try wanted.map { name in
            let url = URL(fileURLWithPath:
                "\(repoRoot)/Tools/diffusion-bench/PromptSuites/\(name).json")
            return try JSONDecoder().decode(LLaDAPromptSuite.self, from: Data(contentsOf: url))
        }
    }

    // Arms. Default: the two served modes on the shipped (cached) engine. A custom arm
    // via --arm composes --arm-mode q|s, --uncached, --mask strict|referenceBias.
    var arms: [LLaDAArm]
    if let armName = argValue("--arm") {
        let mode = GenerationParams.Mode(rawValue: argValue("--arm-mode") ?? "q") ?? .q
        let cached = !hasFlag("--uncached")
        let mask: BlockDiffusionMask.Semantics =
            (argValue("--mask") == "referenceBias") ? .referenceBias : .strict
        arms = [LLaDAArm(name: armName, mode: mode, cached: cached, mask: mask)]
    } else {
        arms = [
            LLaDAArm(name: "q-cached", mode: .q, cached: true, mask: .strict),
            LLaDAArm(name: "s-cached", mode: .s, cached: true, mask: .strict),
        ]
        if let filter = argValue("--arms") {
            let wanted = Set(filter.split(separator: ",").map(String.init))
            arms = arms.filter { wanted.contains($0.name) }
        }
    }
    for arm in arms {
        precondition(!(arm.cached && arm.mask == .referenceBias),
            "`.referenceBias` is a cache-off diagnostic only — ExactPrefixCache exactness "
            + "holds only under `.strict` (phase-2 §6)")
    }

    // H3 diagnostic (m6-logbook): cap MLX's buffer cache so freed intermediates return to
    // the OS instead of pinning towards the 16 GB ceiling alongside 9.5 GB of weights.
    if let mb = argValue("--cache-limit-mb").flatMap(Int.init) {
        Memory.cacheLimit = mb * 1_048_576
        print("MLX cache limit set to \(mb) MB")
    }

    print("diffusion-bench llada — model \(modelDir.path)")
    let loadStart = Date()
    let container = try DiffusionModel.load(from: modelDir)
    if hasFlag("--quantize-lm-head") {
        container.quantizeLMHead()
        print("lm_head quantized to 4-bit (g64) in memory (M8 E5 axis)")
    }
    let tokenizer = try await DiffusionTokenizer.from(modelFolder: tokenizerDir)
    let loadSeconds = Date().timeIntervalSince(loadStart)
    print(String(format: "loaded in %.1fs | arms: %@ | runs/arm: %d | gen-length: %d "
                 + "| suites: %@ | instrument: %@",
                 loadSeconds, arms.map(\.name).joined(separator: ", "), runs, genLength,
                 suites.map(\.name).joined(separator: ", "), instrument ? "on" : "off"))

    let engine = DiffusionEngine(
        model: container.model, speculationK: speculationK, instrument: instrument)
    let isoFormatter = ISO8601DateFormatter()
    var jsonLines: [String] = []

    // WP-2a calibration / JOT pre-experiment: per-step per-position trace dump (offline runs
    // only — adds a per-step readback; use with --runs 1). One JSONL line per logical step.
    var tracePromptId = ""
    if let dumpTracesPath = argValue("--dump-traces") {
        FileManager.default.createFile(atPath: dumpTracesPath, contents: nil)
        guard let traceHandle = FileHandle(forWritingAtPath: dumpTracesPath) else {
            fatalError("cannot open --dump-traces path \(dumpTracesPath)")
        }
        print("trace dump enabled -> \(dumpTracesPath) (per-step readbacks; offline use only)")
        engine.onTrace = { t in
            let rec: [String: Any] = [
                "promptId": tracePromptId,
                "block": t.blockIndex,
                "step": t.stepInBlock,
                "conf": t.confidence.map { Double($0) },
                "gamma": t.transferred.map { $0 ? 1 : 0 },
                "delta": t.edited.map { $0 ? 1 : 0 },
                "x0": t.argmaxToken,
                "masked": t.masked.map { $0 ? 1 : 0 },
            ]
            if let data = try? JSONSerialization.data(withJSONObject: rec),
               let line = String(data: data, encoding: .utf8) {
                traceHandle.write(Data((line + "\n").utf8))
            }
        }
    }

    func params(for mode: GenerationParams.Mode) -> GenerationParams {
        var p = GenerationParams.mode(
            mode, blockLength: blockLength, genLength: genLength,
            maskId: tokenizer.maskId, eosId: tokenizer.eosId, eosEarlyStop: eosEarlyStop,
            nBuf: nBuf, tauAdd: tauAdd, tauSemi: tauSemi,
            speculation: speculation, tauSpan: tauSpan,
            elasticCacheEnabled: elasticCache, elasticGamma: elasticGamma, elasticBeta: elasticBeta,
            elasticStaticBoundary: elasticStaticBoundary)
        if let t = thresholdMaskOverride { p.threshold = t }
        if let t = thresholdEditOverride { p.editingThreshold = t }
        return p
    }

    func generate(_ arm: LLaDAArm, promptIds: [Int])
        -> (output: DiffusionEngine.Output, seconds: Double, peakGB: Double,
            env: EnvSnapshot, warmup: Bool) {
        let warmup = llaDAProcessFirstGeneration
        llaDAProcessFirstGeneration = false
        let swapBefore = swapUsedMB()
        let freeBefore = freeMemoryMB()
        let thermalBefore = thermalStateName()
        GPU.resetPeakMemory()
        let start = Date()
        // Per-block wall-clock (H1 diagnostic: first-block cost includes prefill + kernel
        // compile; steady-state is the later blocks). streamBlock fires at each commit.
        var lastCommit = start
        var blockIndex = 0
        let logBlock: ([Int]) -> Void = { _ in
            let now = Date()
            print(String(format: "    block %d committed at +%.1fs (Δ %.1fs)",
                         blockIndex, now.timeIntervalSince(start),
                         now.timeIntervalSince(lastCommit)))
            lastCommit = now
            blockIndex += 1
        }
        let output = arm.cached
            ? engine.generateCached(prompt: promptIds, params: params(for: arm.mode),
                                    streamBlock: logBlock)
            : engine.generate(prompt: promptIds, params: params(for: arm.mode),
                              maskSemantics: arm.mask, streamBlock: logBlock)
        let seconds = Date().timeIntervalSince(start)
        let env = EnvSnapshot(
            swapUsedMBBefore: swapBefore,
            swapUsedMBAfter: swapUsedMB(),
            freeMemoryMBBefore: freeBefore,
            thermalBefore: thermalBefore,
            thermalAfter: thermalStateName())
        return (output, seconds, Double(GPU.peakMemory) / 1_073_741_824, env, warmup)
    }

    func encodePrompt(_ user: String) throws -> [Int] {
        try tokenizer.applyChatTemplate(messages: [["role": "user", "content": user]])
    }

    func appendResult(
        _ arm: LLaDAArm, run: Int, suite: String, prompt: LLaDAPromptCase,
        promptIds: [Int], output: DiffusionEngine.Output, seconds: Double,
        peakGB: Double, env: EnvSnapshot, warmup: Bool, text: String
    ) {
        // Global logical steps from the engine (the TPF denominator). At nBuf=2 a step can
        // advance two blocks, so sum(stepsPerBlock) would double-count dual-phase steps.
        let steps = output.metrics.logicalStepsTotal
        let stepsBlockSum = output.stepsPerBlock.reduce(0, +)
        let posts = output.metrics.postStepsPerBlock.reduce(0, +)
        let blocks = output.stepsPerBlock.count
        // Effective thresholds (overrides included), not the mode label's — rule F7.
        let effectiveParams = params(for: arm.mode)
        let thresholds = (mask: effectiveParams.threshold, edit: effectiveParams.editingThreshold)
        let result = LLaDARunResult(
            arm: arm.name, run: run, suite: suite, promptId: prompt.id,
            mode: arm.mode.rawValue,
            thresholdMask: thresholds.mask, thresholdEdit: thresholds.edit,
            cached: arm.cached,
            mask: arm.mask == .strict ? "strict" : "referenceBias",
            blockLength: blockLength, genLength: genLength,
            promptTokens: promptIds.count, tokensGenerated: output.tokens.count,
            blocks: blocks, logicalSteps: steps, postSteps: posts,
            stepsPerBlock: output.stepsPerBlock,
            postStepsPerBlock: output.metrics.postStepsPerBlock,
            forwardsEvaluated: output.metrics.forwardsEvaluated,
            syncPoints: output.syncPoints,
            transfersTotal: output.metrics.transfersPerStep.flatMap { $0 }.reduce(0, +),
            editsTotal: output.metrics.editsPerStep.flatMap { $0 }.reduce(0, +),
            meanTransferConfidence: {
                let confs = zip(output.metrics.meanTransferConfidencePerStep,
                                output.metrics.transfersPerStep)
                    .flatMap { zip($0.0, $0.1) }
                    .filter { $0.1 > 0 }
                guard !confs.isEmpty else { return 0 }
                return confs.map { Double($0.0) }.reduce(0, +) / Double(confs.count)
            }(),
            eosBlockIndex: output.metrics.eosBlockIndex,
            totalSeconds: seconds,
            prefillSeconds: output.metrics.prefillSeconds,
            denoiseSeconds: output.metrics.denoiseSeconds,
            commitSeconds: output.metrics.commitSeconds,
            tps: Double(output.tokens.count) / seconds,
            tpfHonest: output.metrics.forwardsEvaluated > 0
                ? Double(output.tokens.count) / Double(output.metrics.forwardsEvaluated) : 0,
            tpfLogical: steps > 0 ? Double(output.tokens.count) / Double(steps) : 0,
            stepsPerBlockMean: blocks > 0 ? Double(stepsBlockSum) / Double(blocks) : 0,
            postStepsPerBlockMean: blocks > 0 ? Double(posts) / Double(blocks) : 0,
            peakMemoryGB: peakGB,
            speculationK: output.metrics.effectiveSpeculationK,
            eosEarlyStop: eosEarlyStop,
            nBuf: output.metrics.effectiveNBuf,
            tauAdd: output.metrics.effectiveTauAdd,
            tauSemi: output.metrics.effectiveTauSemi,
            dualActiveSteps: output.metrics.dualActiveSteps,
            activationSteps: output.metrics.activationSteps,
            trailingStarvedStepsPerBlock: output.metrics.trailingStarvedStepsPerBlock,
            singleActiveDenoiseSeconds: output.metrics.singleActiveDenoiseSeconds,
            dualActiveDenoiseSeconds: output.metrics.dualActiveDenoiseSeconds,
            speculation: output.metrics.effectiveSpeculation,
            tauSpan: output.metrics.effectiveTauSpan,
            acceptedTotal: output.metrics.acceptedPerStep.flatMap { $0 }.reduce(0, +),
            acceptedPerVerifiedStepMean: {
                let verified = output.metrics.acceptedPerStep.flatMap { $0 }.filter { $0 > 0 }
                guard !verified.isEmpty else { return 0 }
                return Double(verified.reduce(0, +)) / Double(verified.count)
            }(),
            tokensProcessedInForwards: output.metrics.tokensProcessedInForwards,
            tpfWidthCorrected: output.metrics.tokensProcessedInForwards > 0
                ? Double(output.tokens.count)
                    / (Double(output.metrics.tokensProcessedInForwards) / Double(blockLength))
                : 0,
            elasticCacheEnabled: arm.cached && elasticCache,
            elasticGamma: elasticGamma,
            elasticBeta: elasticBeta,
            elasticStaticBoundary: elasticStaticBoundary,
            warmupIncluded: warmup,
            processId: Int(ProcessInfo.processInfo.processIdentifier),
            host: sysctlString("hw.model"),
            env: env,
            envValid: env.isValid,
            textPrefix: String(text.prefix(160)),
            date: isoFormatter.string(from: Date()))
        if let data = try? JSONEncoder().encode(result),
           let line = String(data: data, encoding: .utf8) {
            jsonLines.append(line)
            // Append immediately — a killed/crashed run must not lose completed rows
            // (m6-logbook lesson: the first baseline grid was stopped mid-run and its
            // rows existed only in stdout).
            if let handle = FileHandle(forWritingAtPath: jsonPath) {
                handle.seekToEndOfFile()
                handle.write(Data((line + "\n").utf8))
                try? handle.close()
            } else {
                try? (line + "\n").write(toFile: jsonPath, atomically: true, encoding: .utf8)
            }
        }
    }

    func flushJSON() {
        guard !jsonLines.isEmpty else { return }
        print("JSONL appended to \(jsonPath) (\(jsonLines.count) lines)")
    }

    // §6-item-4 quality diagnostic: same prompt, cache off, `.strict` vs `.referenceBias`;
    // record both outputs for side-by-side comparison. Diagnostic only — never a gate, and
    // no variance suite (single run per mask).
    if maskDiagnostic {
        print("---- mask diagnostic: .strict vs .referenceBias (uncached, Q mode) ----")
        // M8 E8: --blind-out PATH additionally writes FULL paired outputs (JSONL rows keep
        // only 160-char prefixes) for the blind-scoring sheet (`Tools/m8_blind_sheet.py`).
        var blindPairs: [[String: String]] = []
        for suite in suites {
            for prompt in suite.prompts {
                let promptIds = try encodePrompt(prompt.user)
                for mask in [BlockDiffusionMask.Semantics.strict, .referenceBias] {
                    let arm = LLaDAArm(
                        name: "mask-diag-\(mask == .strict ? "strict" : "referenceBias")",
                        mode: .q, cached: false, mask: mask)
                    let (output, seconds, peakGB, env, warmup) = generate(arm, promptIds: promptIds)
                    let text = tokenizer.decode(tokens: output.tokens)
                    appendResult(arm, run: 0, suite: suite.name, prompt: prompt,
                                 promptIds: promptIds, output: output, seconds: seconds,
                                 peakGB: peakGB, env: env, warmup: warmup, text: text)
                    blindPairs.append([
                        "suite": suite.name, "prompt": prompt.id, "user": prompt.user,
                        "mask": arm.mask == .strict ? "strict" : "referenceBias",
                        "text": text,
                        "tokens": String(output.tokens.count),
                        "eosBlock": output.metrics.eosBlockIndex.map(String.init) ?? "none",
                    ])
                    print("[\(prompt.id) | \(arm.name)] \(String(text.prefix(200)))\n")
                }
            }
        }
        if let blindPath = argValue("--blind-out") {
            let data = try JSONSerialization.data(
                withJSONObject: ["pairs": blindPairs],
                options: [.prettyPrinted, .sortedKeys])
            try data.write(to: URL(fileURLWithPath: blindPath))
            print("full paired outputs written to \(blindPath)")
        }
        flushJSON()
        return
    }

    var armTotals: [String: [Double]] = [:]
    var firstRun = true
    for arm in arms {
        for run in 0 ..< runs {
            if !firstRun && cooldown > 0 {
                try await Task.sleep(nanoseconds: UInt64(cooldown) * 1_000_000_000)
            }
            firstRun = false
            var runTotal = 0.0

            for suite in suites {
                for prompt in suite.prompts {
                    let promptIds = try encodePrompt(prompt.user)
                    tracePromptId = prompt.id
                    let (output, seconds, peakGB, env, warmup) = generate(arm, promptIds: promptIds)
                    runTotal += seconds
                    let text = tokenizer.decode(tokens: output.tokens)
                    appendResult(arm, run: run, suite: suite.name, prompt: prompt,
                                 promptIds: promptIds, output: output, seconds: seconds,
                                 peakGB: peakGB, env: env, warmup: warmup, text: text)
                    if let dumpTextPath {
                        let record: [String: String] = [
                            "arm": arm.name, "suite": suite.name, "promptId": prompt.id,
                            "run": String(run), "user": prompt.user, "text": text,
                            "tokens": String(output.tokens.count),
                            "eosBlock": output.metrics.eosBlockIndex.map(String.init) ?? "none",
                        ]
                        if let data = try? JSONEncoder().encode(record),
                           let line = String(data: data, encoding: .utf8) {
                            if let handle = FileHandle(forWritingAtPath: dumpTextPath) {
                                handle.seekToEndOfFile()
                                handle.write(Data((line + "\n").utf8))
                                try? handle.close()
                            } else {
                                try? (line + "\n").write(
                                    toFile: dumpTextPath, atomically: true, encoding: .utf8)
                            }
                        }
                    }

                    // Console TPF-logical uses the engine's global step counter — at nBuf=2
                    // sum(stepsPerBlock) double-counts dual-phase steps (JSONL was already right).
                    let steps = output.metrics.logicalStepsTotal
                    let stepsBlockSum = output.stepsPerBlock.reduce(0, +)
                    var flags = ""
                    if warmup { flags += " [warmup]" }
                    if !env.isValid { flags += " [ENV-INVALID: \(env.note)]" }
                    print(String(
                        format: "[%@ run %d | %@] %d tok in %.1fs | %.2f tok/s | "
                            + "TPF %.2f (honest %.2f) | steps/blk %.1f | post/blk %.1f | "
                            + "sync %d | peak %.2f GB%@",
                        arm.name, run, prompt.id, output.tokens.count, seconds,
                        Double(output.tokens.count) / seconds,
                        steps > 0 ? Double(output.tokens.count) / Double(steps) : 0,
                        output.metrics.forwardsEvaluated > 0
                            ? Double(output.tokens.count)
                                / Double(output.metrics.forwardsEvaluated) : 0,
                        output.stepsPerBlock.isEmpty ? 0
                            : Double(stepsBlockSum) / Double(output.stepsPerBlock.count),
                        output.metrics.postStepsPerBlock.isEmpty ? 0
                            : Double(output.metrics.postStepsPerBlock.reduce(0, +))
                                / Double(output.metrics.postStepsPerBlock.count),
                        output.syncPoints, peakGB, flags))
                }
            }
            armTotals[arm.name, default: []].append(runTotal)
            print(String(format: "[%@ run %d] suite total %.1fs", arm.name, run, runTotal))
        }
    }

    // M6 acceptance: <5% cross-run variance per arm. Laptop caveat (Sumi campaign quirk 4):
    // >10-minute arms on the M1 can fail this on thermals alone — compare within-run steady
    // medians there and re-check on the Studio.
    print("---- variance across runs (M6 acceptance: <5%) ----")
    for arm in arms {
        guard let totals = armTotals[arm.name], totals.count > 1 else { continue }
        let mean = totals.reduce(0, +) / Double(totals.count)
        let maxDev = totals.map { abs($0 - mean) / mean }.max() ?? 0
        let verdict = maxDev < 0.05 ? "PASS" : "FAIL"
        print(String(format: "%@: totals %@ | max dev %.1f%% %@",
                     arm.name, totals.map { String(format: "%.0fs", $0) }
                        .joined(separator: " "),
                     maxDev * 100, verdict))
    }
    flushJSON()
}
