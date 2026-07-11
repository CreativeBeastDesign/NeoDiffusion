import XCTest
import Foundation
import MLX
import DiffusionCore
import DiffusionModel
@testable import DiffusionGeneration

/// M8 E2 — drift-corpus dumper (m8-logbook): runs the 4-bit engine cache-disabled on a
/// small fixed prompt set and records **real mid-generation windows** (the exact token
/// sequences the model forwards, masks included) together with the engine's FP32 logits
/// over the active block. The streamed-BF16 reference (`Tools/m8_reference_logits.py`)
/// recomputes reference logits for the same windows; drift metrics compare the two.
///
/// Real canvases, not synthetic probes — the Sumi §3.3 lesson: quantization metrics on
/// off-distribution inputs measure the probe, not the model.
///
/// Opt-in: NEODIFFUSION_M8_CORPUS=1 swift test --filter LLaDADriftCorpusDumper
/// Output: scratch/m8_drift/corpus.safetensors + manifest.json
final class LLaDADriftCorpusDumper: XCTestCase {

    static let modelDir = URL(fileURLWithPath:
        "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/models/llada2-1-mini-4bit")
    static let tokenizerDir = URL(fileURLWithPath:
        "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/models/llada2-1-mini")
    static let outDir = URL(fileURLWithPath:
        "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/m8_drift")

    // One prompt per content regime (steps/block spans 2–21 by content — m6 Finding 5).
    static let prompts: [(id: String, user: String, genLength: Int)] = [
        ("chat-capital", "What is the capital of Japan, and what is the city best known "
            + "for? Answer in two or three sentences.", 128),
        ("reason-trains", "Two trains are 240 km apart and drive toward each other, one "
            + "at 70 km/h and one at 50 km/h. After how many hours do they meet? Show "
            + "your reasoning briefly.", 128),
        ("code-fizzbuzz", "Write a Python function fizzbuzz(n) that returns a list of "
            + "the classic FizzBuzz strings for 1..n.", 128),
        ("essay-bicycle", "Write a short essay about the history of the bicycle.", 256),
    ]

    func testDumpDriftCorpus() async throws {
        try XCTSkipUnless(
            ProcessInfo.processInfo.environment["NEODIFFUSION_M8_CORPUS"] == "1",
            "opt-in: set NEODIFFUSION_M8_CORPUS=1 (loads 9.5 GB, ~10 min)")
        let container = try DiffusionModel.load(from: Self.modelDir)
        let tokenizer = try await DiffusionTokenizer.from(modelFolder: Self.tokenizerDir)
        let engine = DiffusionEngine(model: container.model)
        try FileManager.default.createDirectory(
            at: Self.outDir, withIntermediateDirectories: true)

        var arrays: [String: MLXArray] = [:]
        var manifest: [[String: Any]] = []
        var windowIndex = 0

        for (pid, user, genLength) in Self.prompts {
            let promptIds = try tokenizer.applyChatTemplate(
                messages: [["role": "user", "content": user]])
            let params = GenerationParams.mode(
                .q, blockLength: 32, genLength: genLength,
                maskId: tokenizer.maskId, eosId: tokenizer.eosId, eosEarlyStop: true)

            // Recording forward: identical math to `generate`'s cache-disabled closure,
            // plus window/logits capture on a sampling schedule (block-relative steps
            // 0 and 4, then every 16th — ≤ ~8 windows per prompt).
            var stepInBlock: [Int: Int] = [:]  // block ordinal → forwards seen
            let forward: DiffusionEngine.Forward = { [model = container.model] windowIds, activeLen, frozen in
                let W = windowIds.dim(windowIds.ndim - 1)
                let logits = model.logits(forTokens: windowIds, blockLength: params.blockLength)
                let active = logits[0..., (W - activeLen)..., 0...]

                let block = W / params.blockLength - 1
                let step = stepInBlock[block, default: 0]
                stepInBlock[block] = step + 1
                if step == 0 || step == 4 || step % 16 == 0 {
                    let key = String(format: "w%03d", windowIndex)
                    arrays["\(key).ids"] = windowIds.asType(.int32)
                    arrays["\(key).logits4bit"] = active.asType(.float16)  // storage only
                    eval(arrays["\(key).ids"]!, arrays["\(key).logits4bit"]!)
                    manifest.append([
                        "key": key, "prompt": pid, "block": block, "step": step,
                        "windowLength": W, "activeLen": activeLen,
                    ])
                    windowIndex += 1
                }
                return active
            }

            let output = engine.run(
                prompt: promptIds, params: params, forward: forward, streamBlock: nil)
            print("[m8-corpus] \(pid): \(output.tokens.count) tok, "
                + "\(output.stepsPerBlock.reduce(0, +)) steps, windows so far \(windowIndex)")
        }

        try MLX.save(
            arrays: arrays, url: Self.outDir.appendingPathComponent("corpus.safetensors"))
        let meta: [String: Any] = [
            "windows": manifest,
            "artefact": Self.modelDir.path,
            "blockLength": 32,
            "mode": "q",
            "maskSemantics": "strict",
            "note": "logits4bit are FP32 engine outputs stored as F16; recompute metrics "
                + "against float32 reference logits",
        ]
        let data = try JSONSerialization.data(
            withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: Self.outDir.appendingPathComponent("manifest.json"))
        print("[m8-corpus] wrote \(windowIndex) windows to \(Self.outDir.path)")
        XCTAssertGreaterThan(windowIndex, 12, "corpus unexpectedly small")
    }
}
