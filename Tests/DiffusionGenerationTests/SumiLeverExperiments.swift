import XCTest
import MLX
import DiffusionModel
@testable import DiffusionGeneration

/// Speed-lever experiments on the real 4-bit weights (sumi-plan.md §5, requested by André
/// 2026-07-09). Three axes, adaptive sampler at temperature 0 throughout:
///
/// 1. **Canvas**: 512 vs 1024 (vs 1536 anchor) — de-confounds the canvas-length-lock
///    finding (the earlier probe evidence mixed sampler/step changes with canvas changes;
///    training reportedly used canvases 1024–4096, so 1024 should work if the probe misled).
/// 2. **Tokens per step (k)**: 1 vs 4 — paper Figure 4 shows k=4 holds accuracy on
///    HumanEval/MBPP (GSM8K degrades; collapse starts at k≥8). A direct 4× step reduction.
/// 3. **Revision budget**: the paper finds passes beyond first commit change ≤1% of tokens
///    (mostly A→B→A round trips) and never improve accuracy → compare freeze-after-commit
///    (steps = window/k exactly) vs revisions allowed at 1× and 2× steps, measuring
///    post-coverage token changes and round trips.
///
/// Diagnostic suite, not a gate: outputs are printed for eyeballing (one QA prompt — task
/// accuracy claims need the S3 bench, not this). Per-step canvas readbacks are deliberate
/// (sync budget does not apply here). Opt-in:
///   NEODIFFUSION_SUMI_EXPERIMENTS=1 swift test --filter SumiLeverExperiments
/// Expect ~45–50 min total on the M1; each test method is independently filterable.
final class SumiLeverExperiments: XCTestCase {

    static let modelDir = URL(fileURLWithPath:
        "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/models/sumi-7b-4bit")
    static let tokenizerDir = URL(fileURLWithPath:
        "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/Tools/reference/sumi")

    static let prompt = "Question: What is the capital of Japan?\nAnswer:"
    static let budget = 64

    // Shared across test methods: load the 5.5 GB artefact once per process.
    // XCTest runs these methods serially in one process; no concurrent access exists.
    nonisolated(unsafe) static var shared:
        (container: SumiDiffusionModel, tokenizer: SumiTokenizer)?

    private func harness() async throws -> (SumiDiffusionModel, SumiTokenizer) {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_EXPERIMENTS"] == "1",
            "opt-in: set NEODIFFUSION_SUMI_EXPERIMENTS=1 (~45-50 min on the M1)")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.modelDir.appendingPathComponent("model.safetensors").path),
            "4-bit Sumi artefact missing at models/sumi-7b-4bit")
        if let shared = Self.shared { return shared }
        let container = try SumiDiffusionModel.load(from: Self.modelDir)
        let tokenizer = try await SumiTokenizer.from(modelFolder: Self.tokenizerDir)
        Self.shared = (container, tokenizer)
        return (container, tokenizer)
    }

    struct ArmResult {
        let name: String
        let avgStep: Double
        let total: Double
        let text: String
        let stepCanvases: [[Int32]]
        let contentRange: Range<Int>
    }

    @discardableResult
    private func runArm(
        _ name: String,
        canvas: Int,
        k: Int,
        steps: Int,
        freeze: Bool,
        container: SumiDiffusionModel,
        tokenizer: SumiTokenizer
    ) -> ArmResult {
        let promptIds = tokenizer.encode(text: Self.prompt).map(Int32.init)
        let P = promptIds.count
        let engine = SumiEngine(model: container.model)

        var stepCanvases: [[Int32]] = []
        var stepTimes: [TimeInterval] = []
        var last = Date()
        let start = Date()
        let out = engine.generate(
            SumiGenerationRequest(
                promptIds: promptIds, maxNewTokens: Self.budget, canvasLength: canvas,
                numDenoisingSteps: steps, sampler: .adaptive, temperature: 0,
                tokensPerStep: k, denoiseEnd: P + Self.budget + 2,
                trimAtEOS: true, seed: 11, freezeCommitted: freeze),
            onStep: { _, z in
                let now = Date()
                stepTimes.append(now.timeIntervalSince(last))
                last = now
                stepCanvases.append(z.reshaped(-1).asArray(Int32.self))
            })
        let total = Date().timeIntervalSince(start)
        let avgStep = stepTimes.dropFirst().reduce(0, +) / Double(max(stepTimes.count - 1, 1))
        let text = tokenizer.decode(tokens: out.sequences.map(Int.self.init))
        let tps = Double(Self.budget) / total

        print("[sumi-EXP] \(name): canvas \(canvas), k=\(k), \(steps) steps, freeze=\(freeze) | "
            + String(format: "avg step %.1fs, total %.0fs, %.3f tok/s", avgStep, total, tps))
        print("[sumi-EXP]   text: \(text.replacingOccurrences(of: "\n", with: "⏎"))")
        return ArmResult(
            name: name, avgStep: avgStep, total: total, text: text,
            stepCanvases: stepCanvases, contentRange: P ..< (P + Self.budget))
    }

    /// Post-coverage revision statistics: per-step changed-token counts in the content
    /// window after the full-coverage step, and A→B(→…)→A round trips (position revised at
    /// least once after its first change yet ending at its first-change value).
    private func revisionStats(_ arm: ArmResult, coverageStep: Int) -> String {
        let range = arm.contentRange
        var perStepChanges: [Int] = []
        for s in 1 ..< arm.stepCanvases.count {
            let prev = arm.stepCanvases[s - 1]
            let cur = arm.stepCanvases[s]
            perStepChanges.append(range.reduce(0) { $0 + (prev[$1] != cur[$1] ? 1 : 0) })
        }
        let postCoverage = perStepChanges.suffix(from: min(coverageStep, perStepChanges.count))
        let totalPost = postCoverage.reduce(0, +)

        var roundTrips = 0
        let final = arm.stepCanvases.last!
        for pos in range {
            var firstChangeValue: Int32? = nil
            var changesAfterFirst = 0
            var prev = arm.stepCanvases[0][pos]
            for s in 1 ..< arm.stepCanvases.count {
                let cur = arm.stepCanvases[s][pos]
                if cur != prev {
                    if firstChangeValue == nil { firstChangeValue = cur } else { changesAfterFirst += 1 }
                }
                prev = cur
            }
            if let f = firstChangeValue, changesAfterFirst > 0, final[pos] == f {
                roundTrips += 1
            }
        }
        return "post-coverage changes \(totalPost) (per step \(Array(postCoverage))), "
            + "round trips \(roundTrips)/\(range.count)"
    }

    // MARK: - Axis 1: canvas length (k=4, 16 steps)

    func testCanvasAxis() async throws {
        let (container, tokenizer) = try await harness()
        for canvas in [512, 1024, 1536] {
            runArm("canvas-\(canvas)", canvas: canvas, k: 4, steps: 16, freeze: false,
                   container: container, tokenizer: tokenizer)
        }
    }

    // MARK: - Axis 2: tokens per step at canvas 1024 (k=1 vs k=4, full coverage each)

    func testTokensPerStepAxis() async throws {
        let (container, tokenizer) = try await harness()
        runArm("k1", canvas: 1024, k: 1, steps: 64, freeze: false,
               container: container, tokenizer: tokenizer)
        runArm("k4", canvas: 1024, k: 4, steps: 16, freeze: false,
               container: container, tokenizer: tokenizer)
    }

    // MARK: - Axis 3: revision budget at canvas 1024, k=4

    func testRevisionBudget() async throws {
        let (container, tokenizer) = try await harness()
        // (a) commit-once: exactly window/k steps, no revisions possible.
        let frozen = runArm("freeze-16", canvas: 1024, k: 4, steps: 16, freeze: true,
                            container: container, tokenizer: tokenizer)
        // (b) revisions allowed, same step count (reference behaviour).
        let ref16 = runArm("revise-16", canvas: 1024, k: 4, steps: 16, freeze: false,
                           container: container, tokenizer: tokenizer)
        // (c) revisions allowed, 2× steps — measures what the extra budget actually does.
        let ref32 = runArm("revise-32", canvas: 1024, k: 4, steps: 32, freeze: false,
                           container: container, tokenizer: tokenizer)

        print("[sumi-EXP] revise-16 vs freeze-16 identical text: \(ref16.text == frozen.text)")
        print("[sumi-EXP] revise-32 stats: \(revisionStats(ref32, coverageStep: 16))")
        print("[sumi-EXP] revise-32 vs revise-16 identical text: \(ref32.text == ref16.text)")
    }
}
