import Foundation
import MLX
import DiffusionModel
import DiffusionGeneration

// diffusion-bench — NeoDiffusion benchmark CLI (phase-2 M6 metric set).
//
// Sumi mode (sumi-plan.md §5 S3.1):
//   swift run -c release diffusion-bench sumi [--model DIR] [--tokenizer DIR]
//       [--runs N] [--budget N] [--prompt TEXT] [--json PATH]
//       [--arm NAME --sampler adaptive|greedy|ancestral --canvas N --k N --steps N
//        [--freeze] [--temperature F]]
//
// Without --arm, runs the frozen baseline suite (3 arms × --runs, default 3):
//   recipe-adaptive-k4   canvas 1024, k=4, 16 steps   (the measured M1 operating point)
//   quality-adaptive-k1  canvas 1024, k=1, 64 steps   (coherence reference)
//   reference-ancestral  canvas 1024, 64 steps, t=0.7 (reference-default sampler)
//
// Metric mapping vs the LLaDA M6 set (see README): TPF and steps/block do not apply
// (no blocks); per-phase split is load / canvas+first-step / steady-state steps.
// Sync points are analytic: 1 blocking eval per step + 1 final trim readback.

// Line-buffer stdout: long runs are usually redirected to a log, and block buffering
// otherwise holds all results back until process exit.
setvbuf(stdout, nil, _IOLBF, 0)

struct ArmSpec {
    let name: String
    let sampler: SumiGenerationRequest.Sampler
    let canvas: Int
    let k: Int
    let steps: Int
    let freeze: Bool
    let temperature: Float
    let useDenoiseEnd: Bool
    var pipelined: Bool = false
    var earlyExitStable: Int? = nil
    var earlyExitMinSteps: Int = 0
    var earlyExitFraction: Float = 1.0
}

struct RunResult: Codable {
    let arm: String
    let run: Int
    let sampler: String
    let canvas: Int
    let tokensPerStep: Int
    let steps: Int
    let budget: Int
    let stepMeanSeconds: Double
    let stepStdSeconds: Double
    let stepMinSeconds: Double
    let stepMaxSeconds: Double
    let stepP95Seconds: Double
    let firstStepSeconds: Double
    let totalSeconds: Double
    let effectiveTPS: Double
    let peakMemoryGB: Double
    let syncPoints: Int
    let pipelined: Bool
    let stepsExecuted: Int
    let textPrefix: String
    let date: String
}

func argValue(_ name: String) -> String? {
    guard let i = CommandLine.arguments.firstIndex(of: name),
        i + 1 < CommandLine.arguments.count
    else { return nil }
    return CommandLine.arguments[i + 1]
}

func hasFlag(_ name: String) -> Bool {
    CommandLine.arguments.contains(name)
}

func percentile(_ sorted: [Double], _ p: Double) -> Double {
    guard !sorted.isEmpty else { return 0 }
    let idx = min(sorted.count - 1, Int((Double(sorted.count) * p).rounded(.down)))
    return sorted[idx]
}

if CommandLine.arguments.count > 1, CommandLine.arguments[1] == "llada" {
    try await runLLaDABench()
    exit(0)
}

guard CommandLine.arguments.count > 1, CommandLine.arguments[1] == "sumi" else {
    print("""
        diffusion-bench — NeoDiffusion benchmark CLI

        Usage: diffusion-bench llada [options]   (phase-2 M6 metric set)
        Options:
          --model DIR        converted model dir (default models/llada2-1-mini-4bit)
          --tokenizer DIR    tokenizer dir (default models/llada2-1-mini)
          --runs N           runs per arm (default 3)
          --cooldown N       seconds between runs (default 30)
          --gen-length N     tokens to generate per prompt (default 128)
          --suites LIST      prompt suites, comma-sep (default chat,reasoning,code)
          --prompt TEXT      single ad-hoc prompt instead of the suites
          --json PATH        append JSONL results (default scratch/llada_bench.jsonl)
          --speculation-k N  loop-control speculation depth (default 4)
          --no-early-stop    disable eos_early_stop
          --no-instrument    skip phase-boundary evals (phase clocks become unreliable)
          --mask-diagnostic  .strict vs .referenceBias output comparison (uncached; §6 item 4)
          --arm NAME         single custom arm: --arm-mode q|s, --uncached,
                             --mask strict|referenceBias
          --arms LIST        filter the default arms (q-cached,s-cached)

        Usage: diffusion-bench sumi [options]    (sumi-plan.md §5 S3.1)
        Options:
          --model DIR       converted model dir (default models/sumi-7b-4bit)
          --tokenizer DIR   tokenizer dir (default Tools/reference/sumi)
          --runs N          runs per arm (default 3)
          --budget N        content budget in tokens (default 64)
          --prompt TEXT     prompt (default: capital-of-Japan QA)
          --json PATH       append JSONL results (default scratch/sumi_bench.jsonl)
          --arm NAME        run a single custom arm instead of the baseline suite,
                            with --sampler/--canvas/--k/--steps/--freeze/--temperature
        """)
    exit(CommandLine.arguments.count > 1 ? 1 : 0)
}

let repoRoot = FileManager.default.currentDirectoryPath
let modelDir = URL(fileURLWithPath: argValue("--model") ?? "\(repoRoot)/models/sumi-7b-4bit")
let tokenizerDir = URL(
    fileURLWithPath: argValue("--tokenizer") ?? "\(repoRoot)/Tools/reference/sumi")
let runs = Int(argValue("--runs") ?? "3") ?? 3
let budget = Int(argValue("--budget") ?? "64") ?? 64
// Cooldown between runs: the M1 under sustained GPU load throttles and background
// activity injects slow-step outliers; a pause improves cross-run reproducibility.
let cooldown = Int(argValue("--cooldown") ?? "30") ?? 30
let prompt = argValue("--prompt") ?? "Question: What is the capital of Japan?\nAnswer:"
let jsonPath = argValue("--json") ?? "\(repoRoot)/scratch/sumi_bench.jsonl"

let arms: [ArmSpec]
if let armName = argValue("--arm") {
    let sampler = SumiGenerationRequest.Sampler(
        rawValue: argValue("--sampler") ?? "adaptive") ?? .adaptive
    let k = Int(argValue("--k") ?? "4") ?? 4
    arms = [
        ArmSpec(
            name: armName,
            sampler: sampler,
            canvas: Int(argValue("--canvas") ?? "1024") ?? 1024,
            k: k,
            steps: Int(argValue("--steps") ?? String((budget + k - 1) / k))
                ?? (budget + k - 1) / k,
            freeze: hasFlag("--freeze"),
            temperature: Float(argValue("--temperature") ?? (sampler == .ancestral ? "0.7" : "0"))
                ?? 0,
            useDenoiseEnd: sampler == .adaptive || hasFlag("--denoise-end"),
            pipelined: hasFlag("--pipelined"),
            earlyExitStable: argValue("--early-exit").flatMap(Int.init),
            earlyExitMinSteps: Int(argValue("--min-steps") ?? "0") ?? 0,
            earlyExitFraction: Float(argValue("--stable-fraction") ?? "1.0") ?? 1.0)
    ]
} else {
    let baseline = [
        ArmSpec(name: "recipe-adaptive-k4", sampler: .adaptive, canvas: 1024, k: 4,
                steps: 16, freeze: false, temperature: 0, useDenoiseEnd: true),
        ArmSpec(name: "quality-adaptive-k1", sampler: .adaptive, canvas: 1024, k: 1,
                steps: 64, freeze: false, temperature: 0, useDenoiseEnd: true),
        ArmSpec(name: "reference-ancestral", sampler: .ancestral, canvas: 1024, k: 1,
                steps: 64, freeze: false, temperature: 0.7, useDenoiseEnd: false),
    ]
    if let filter = argValue("--arms") {
        let wanted = Set(filter.split(separator: ",").map(String.init))
        arms = baseline.filter { wanted.contains($0.name) }
    } else {
        arms = baseline
    }
}

print("diffusion-bench sumi — model \(modelDir.path)")
let loadStart = Date()
let container = try SumiDiffusionModel.load(from: modelDir)
if hasFlag("--quantize-lm-head") {
    container.quantizeLMHead()
    print("lm_head quantized to 4-bit (g64) in memory")
}
let tokenizer = try await SumiTokenizer.from(modelFolder: tokenizerDir)
let loadSeconds = Date().timeIntervalSince(loadStart)
print(String(format: "loaded in %.1fs | arms: %@ | runs/arm: %d | budget: %d",
             loadSeconds, arms.map(\.name).joined(separator: ", "), runs, budget))

let engine = SumiEngine(model: container.model)
let promptIds = tokenizer.encode(text: prompt).map(Int32.init)
let isoFormatter = ISO8601DateFormatter()
var jsonLines: [String] = []
var armTotals: [String: [Double]] = [:]

var firstRun = true
for arm in arms {
    for run in 0 ..< runs {
        if !firstRun && cooldown > 0 {
            try await Task.sleep(nanoseconds: UInt64(cooldown) * 1_000_000_000)
        }
        firstRun = false
        GPU.resetPeakMemory()
        var stepTimes: [Double] = []
        var last = Date()
        let start = Date()

        let request = SumiGenerationRequest(
            promptIds: promptIds,
            maxNewTokens: budget,
            canvasLength: arm.canvas,
            numDenoisingSteps: arm.steps,
            sampler: arm.sampler,
            temperature: arm.temperature,
            tokensPerStep: arm.k,
            denoiseEnd: arm.useDenoiseEnd ? promptIds.count + budget + 2 : nil,
            trimAtEOS: true,
            seed: UInt64(17 + run),
            freezeCommitted: arm.freeze,
            earlyExit: arm.earlyExitStable.map {
                SumiGenerationRequest.EarlyExit(
                    stableSteps: $0, minSteps: arm.earlyExitMinSteps,
                    stableFraction: arm.earlyExitFraction)
            })

        let onStep: ((Int, MLX.MLXArray) -> Void)? = arm.pipelined ? nil : { _, _ in
            let now = Date()
            stepTimes.append(now.timeIntervalSince(last))
            last = now
        }
        let output = engine.generate(request, pipelined: arm.pipelined, onStep: onStep)

        let total = Date().timeIntervalSince(start)
        let steady = Array(stepTimes.dropFirst()).sorted()
        let mean = steady.reduce(0, +) / Double(max(steady.count, 1))
        let variance = steady.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
            / Double(max(steady.count, 1))
        let text = tokenizer.decode(tokens: output.sequences.map(Int.init))

        let result = RunResult(
            arm: arm.name, run: run, sampler: arm.sampler.rawValue,
            canvas: arm.canvas, tokensPerStep: arm.k, steps: arm.steps, budget: budget,
            stepMeanSeconds: mean,
            stepStdSeconds: variance.squareRoot(),
            stepMinSeconds: steady.first ?? 0,
            stepMaxSeconds: steady.last ?? 0,
            stepP95Seconds: percentile(steady, 0.95),
            firstStepSeconds: stepTimes.first ?? 0,
            totalSeconds: total,
            effectiveTPS: Double(budget) / total,
            peakMemoryGB: Double(Memory.peakMemory) / 1_073_741_824,
            syncPoints: arm.pipelined ? 1 : output.stepsExecuted + 1,
            pipelined: arm.pipelined,
            stepsExecuted: output.stepsExecuted,
            textPrefix: String(text.prefix(160)),
            date: isoFormatter.string(from: Date()))

        armTotals[arm.name, default: []].append(total)
        if let data = try? JSONEncoder().encode(result),
            let line = String(data: data, encoding: .utf8) {
            jsonLines.append(line)
        }
        print(String(
            format: "[%@ run %d] step %.2f±%.2fs (p95 %.2f, first %.2f) | total %.0fs | "
                + "%.3f tok/s | peak %.2f GB",
            arm.name, run, mean, variance.squareRoot(),
            percentile(steady, 0.95), stepTimes.first ?? 0, total,
            Double(budget) / total, Double(Memory.peakMemory) / 1_073_741_824))
    }
}

// Acceptance check (sumi-plan.md §5 S3.1): <5% total-time variance across runs per arm.
print("---- variance across runs (acceptance: <5%) ----")
for arm in arms {
    guard let totals = armTotals[arm.name], totals.count > 1 else { continue }
    let mean = totals.reduce(0, +) / Double(totals.count)
    let maxDev = totals.map { abs($0 - mean) / mean }.max() ?? 0
    let verdict = maxDev < 0.05 ? "PASS" : "FAIL"
    print(String(format: "%@: totals %@ | max dev %.1f%% %@",
                 arm.name, totals.map { String(format: "%.0fs", $0) }.joined(separator: " "),
                 maxDev * 100, verdict))
}

if !jsonLines.isEmpty {
    let existing = (try? String(contentsOfFile: jsonPath, encoding: .utf8)) ?? ""
    try? (existing + jsonLines.joined(separator: "\n") + "\n")
        .write(toFile: jsonPath, atomically: true, encoding: .utf8)
    print("JSONL appended to \(jsonPath) (\(jsonLines.count) lines)")
}
