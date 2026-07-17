import XCTest
import MLX
import DiffusionCore
import DiffusionModel
@testable import DiffusionGeneration

/// Real-weight smoke gate for the converted LLaDA2.1-mini 4-bit artefact — the first time
/// the full engine (loader → tokenizer → chat template → denoising loop → ExactPrefixCache
/// → eos trim) runs against real weights. This is a *smoke* gate, not parity: the 4-bit
/// path is held to task-level quality only (gotcha 5); BF16 token parity is Studio-deferred.
///
/// Also the first place eos-trim is exercised for real — toy fixtures never emit `eos_id`
/// (handoff §4), so a trimmed (< genLength) output here is itself signal.
///
/// Opt-in: loads ~9.5 GB and takes minutes on the dev M1, so it runs only with
/// `NEODIFFUSION_LLADA_REAL=1 swift test --filter LLaDARealWeightSmokeTests`.
final class LLaDARealWeightSmokeTests: XCTestCase {

    static let modelDir = URL(fileURLWithPath:
        "./models/llada2-1-mini-4bit")
    static let tokenizerDir = URL(fileURLWithPath:
        "./models/llada2-1-mini")

    private func loadContainerOrSkip() throws -> DiffusionModel {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_LLADA_REAL"] == "1",
            "opt-in: set NEODIFFUSION_LLADA_REAL=1 (loads ~9.5 GB, minutes of wall-clock)")
        try XCTSkipUnless(
            FileManager.default.fileExists(
                atPath: Self.modelDir.appendingPathComponent("model.safetensors").path),
            "4-bit LLaDA artefact missing at models/llada2-1-mini-4bit")
        return try DiffusionModel.load(from: Self.modelDir)
    }

    func testRealWeightGeneration() async throws {
        let loadStart = Date()
        let container = try loadContainerOrSkip()
        let tokenizer = try await DiffusionTokenizer.from(modelFolder: Self.tokenizerDir)
        print(String(format: "[llada-smoke] loaded in %.1fs",
                     Date().timeIntervalSince(loadStart)))

        // Sanity anchors from the config (CLAUDE.md quick facts).
        XCTAssertEqual(container.config.vocabSize, 157184)
        XCTAssertEqual(container.config.numHiddenLayers, 20)
        XCTAssertEqual(container.config.numExperts, 256)
        XCTAssertTrue(container.config.useQkNorm)

        let promptIds = try tokenizer.applyChatTemplate(
            messages: [["role": "user", "content": "What is the capital of Japan?"]])
        print("[llada-smoke] prompt tokens: \(promptIds.count)")

        let params = GenerationParams.mode(
            .q, blockLength: 32, genLength: 128,
            maskId: tokenizer.maskId, eosId: tokenizer.eosId, eosEarlyStop: true)

        let engine = DiffusionEngine(model: container.model, instrument: true)
        GPU.resetPeakMemory()
        let start = Date()
        let output = engine.generateCached(prompt: promptIds, params: params)
        let seconds = Date().timeIntervalSince(start)

        XCTAssertFalse(output.tokens.isEmpty)
        // The engine must never leak mask_id into committed output.
        XCTAssertFalse(output.finalSequence.contains(tokenizer.maskId))

        let text = tokenizer.decode(tokens: output.tokens)
        let steps = output.stepsPerBlock.reduce(0, +)
        print("[llada-smoke] output: \(text)")
        print(String(
            format: "[llada-smoke] %d tok in %.1fs (%.2f tok/s) | blocks %d | steps %d | "
                + "forwards %d | sync %d | prefill %.2fs denoise %.2fs commit %.2fs | "
                + "peak %.2f GB | eos-trimmed: %@",
            output.tokens.count, seconds, Double(output.tokens.count) / seconds,
            output.blockCommits.count, steps,
            output.metrics.forwardsEvaluated, output.syncPoints,
            output.metrics.prefillSeconds, output.metrics.denoiseSeconds,
            output.metrics.commitSeconds,
            Double(Memory.peakMemory) / 1_073_741_824,
            output.tokens.count < params.genLength ? "yes" : "no"))

        // Task-level sanity, labelled as such (one-prompt eyeball tier — the Sumi campaign's
        // day-2 pattern): the answer to a fact question this basic should mention Tokyo.
        XCTAssertTrue(text.localizedCaseInsensitiveContains("tokyo"),
            "expected 'Tokyo' in: \(text.prefix(300))")
    }
}
