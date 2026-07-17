import XCTest
import MLX
import DiffusionModel
@testable import DiffusionGeneration

/// S2.2 smoke gate (sumi-plan.md §4): the converted 4-bit artefact loads on the M1 and
/// produces text from a real prompt; peak memory and per-step wall-clock are recorded.
/// This is a *smoke* gate, not parity — the 4-bit path is held to task-level quality only
/// (house rule, gotcha 5); the BF16 Levenshtein gate is Studio-deferred (§7).
///
/// **Canvas-length note (revised 2026-07-09)**: generation works at canvas 1024 (correct,
/// clean output) and marginally at 512 — see `SumiLeverExperiments.testCanvasAxis` and
/// sumi-plan.md §4 Finding 1 (revised). The clean-text *reconstruction* probe below is
/// reliable only at canvas ≥ 1536 (a mostly-clean canvas is itself off-distribution); it is
/// kept solely as a weight-unpacking sanity check, not as evidence about usable lengths.
/// Generation smokes run at canvas 1024, the measured operating point.
///
/// Opt-in: minutes of wall-clock on the M1, so it runs only with
/// `NEODIFFUSION_SUMI_REAL=1 swift test --filter SumiRealWeightSmokeTests`.
final class SumiRealWeightSmokeTests: XCTestCase {

    static let modelDir = URL(fileURLWithPath:
        "./models/sumi-7b-4bit")
    static let tokenizerDir = URL(fileURLWithPath:
        "./Tools/reference/sumi")
    static let canvas = 1024        // generation operating point (Finding 1, revised)
    static let probeCanvas = 1536   // reconstruction probe needs ≥1536 (see header)

    private func loadContainerOrSkip() throws -> SumiDiffusionModel {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_REAL"] == "1",
            "opt-in: set NEODIFFUSION_SUMI_REAL=1 (loads 5.5 GB, minutes of wall-clock)")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.modelDir.appendingPathComponent("model.safetensors").path),
            "4-bit Sumi artefact missing at models/sumi-7b-4bit")
        return try SumiDiffusionModel.load(from: Self.modelDir)
    }

    /// Weight-unpacking sanity probe ONLY (see header — do not read canvas semantics into
    /// it): clean text at the front of a ≥1536 canvas (random tail) must be largely
    /// reconstructed. Near-chance agreement means broken weight unpacking.
    func testCleanReconstructionInCanvas() async throws {
        let container = try loadContainerOrSkip()
        let tokenizer = try await SumiTokenizer.from(modelFolder: Self.tokenizerDir)

        let text = "The quick brown fox jumps over the lazy dog. The capital of Japan is Tokyo, "
            + "and the capital of France is Paris. Water boils at 100 degrees Celsius."
        let clean = tokenizer.encode(text: text).map(Int32.init)
        let P = clean.count

        var ids = clean
        var state: UInt64 = 0x9E3779B97F4A7C15
        for _ in P ..< Self.probeCanvas {  // cheap deterministic LCG tail; distribution irrelevant
            state = state &* 6364136223846793005 &+ 1442695040888963407
            ids.append(Int32(state % UInt64(container.config.vocabSize)))
        }
        let input = MLXArray(ids).reshaped(1, ids.count)

        let logits = container.model.logits(forTokens: input)
        let predicted = logits.argMax(axis: -1).asType(.int32)
        let agreement = (predicted[0..., ..<P] .== MLXArray(clean).reshaped(1, P))
            .sum().item(Int.self)
        let ratio = Float(agreement) / Float(P)

        print("[sumi-S2.2] clean reconstruction @canvas \(Self.probeCanvas): \(agreement)/\(P) (\(ratio))")
        XCTAssertGreaterThan(
            ratio, 0.5,
            "clean-prefix self-agreement \(ratio) at working canvas — check weight unpacking")
    }

    func testRealWeightGeneration() async throws {
        GPU.resetPeakMemory()
        let loadStart = Date()
        let container = try loadContainerOrSkip()
        let loadSeconds = Date().timeIntervalSince(loadStart)
        XCTAssertEqual(container.config.quantization?.bits, 4)
        XCTAssertEqual(container.config.quantization?.groupSize, 64)

        let tokenizer = try await SumiTokenizer.from(modelFolder: Self.tokenizerDir)
        let engine = SumiEngine(model: container.model)
        let config = container.config

        let prompt = "The capital city of Japan is"
        let promptIds = tokenizer.encode(text: prompt).map(Int32.init)
        let P = promptIds.count
        let budget = 48
        let steps = 8

        var stepTimes: [TimeInterval] = []
        var last = Date()
        let out = engine.generate(
            SumiGenerationRequest(
                promptIds: promptIds, maxNewTokens: budget, canvasLength: Self.canvas,
                numDenoisingSteps: steps, sampler: .greedy, temperature: 0,
                trimAtEOS: true, seed: 7),
            onStep: { _, _ in
                let now = Date()
                stepTimes.append(now.timeIntervalSince(last))
                last = now
            })
        let text = tokenizer.decode(tokens: out.sequences.map(Int.self.init))

        let canvasArr = out.canvas.reshaped(-1).asArray(Int32.self)
        XCTAssertEqual(Array(canvasArr[0 ..< P]), promptIds, "prompt must stay frozen")
        XCTAssertEqual(canvasArr[P + budget], Int32(config.eosTokenId), "EOS anchor")
        XCTAssertTrue(canvasArr.allSatisfy { $0 >= 0 && $0 < Int32(config.vocabSize) })

        let peakGB = Double(Memory.peakMemory) / 1_073_741_824
        let avgStep = stepTimes.dropFirst().reduce(0, +) / Double(max(stepTimes.count - 1, 1))
        let tps = Double(budget) / stepTimes.reduce(0, +)
        print("""
            [sumi-S2.2] 4-bit real-weight smoke (canvas \(Self.canvas), greedy \(steps) steps):
              load: \(String(format: "%.1f", loadSeconds))s, peak GPU memory: \(String(format: "%.2f", peakGB)) GB
              avg step: \(String(format: "%.1f", avgStep))s, effective \(String(format: "%.2f", tps)) tok/s over the \(budget)-token budget
              greedy: \(text)
            """)
    }

    /// Quality read with the reference-default sampler: ancestral, README-style step count,
    /// temperature 0.7 (the README's ancestral example). No hard quality assert — 4-bit is
    /// task-level only; this prints the output for eyeballing and records the timing.
    func testAncestralQuality() async throws {
        let container = try loadContainerOrSkip()
        let tokenizer = try await SumiTokenizer.from(modelFolder: Self.tokenizerDir)
        let engine = SumiEngine(model: container.model)

        let prompt = "Question: What is the capital of Japan?\nAnswer:"
        let promptIds = tokenizer.encode(text: prompt).map(Int32.init)
        let steps = ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_STEPS"]
            .flatMap(Int.init) ?? 64

        var stepTimes: [TimeInterval] = []
        var last = Date()
        let out = engine.generate(
            SumiGenerationRequest(
                promptIds: promptIds, maxNewTokens: 64, canvasLength: Self.canvas,
                numDenoisingSteps: steps, sampler: .ancestral, temperature: 0.7,
                trimAtEOS: true, seed: 3),
            onStep: { _, _ in
                let now = Date()
                stepTimes.append(now.timeIntervalSince(last))
                last = now
            })
        let text = tokenizer.decode(tokens: out.sequences.map(Int.self.init))
        let avgStep = stepTimes.dropFirst().reduce(0, +) / Double(max(stepTimes.count - 1, 1))
        print("""
            [sumi-S2.2] ancestral quality (canvas \(Self.canvas), \(steps) steps, temp 0.7):
              avg step: \(String(format: "%.1f", avgStep))s
              output: \(text)
            """)
        XCTAssertGreaterThan(out.sequences.count, promptIds.count)
    }
}
