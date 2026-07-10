import XCTest
import MLX
import MLXNN
import DiffusionCore
import DiffusionModel
@testable import DiffusionGeneration

/// sumi-M4'/S1.5 acceptance (sumi-plan.md §3): end-to-end deterministic generation parity —
/// the engine reproduces the reference `generate()` per-step canvases, final canvas, and
/// EOS-trimmed output **token-for-token** on 8 cases (greedy + adaptive at temperature 0,
/// anchors on/off, `denoise_end` set/unset, budget clamp, bos-only prompt).
///
/// The initial canvas comes from the fixture (`torch.randint` is not reproducible across
/// RNGs — decision S0.1-1); everything after it is deterministic.
///
/// Regenerate fixtures with:
///   cd Tools && ../scratch/sumi-venv/bin/python generate_sumi_loop_fixtures.py \
///       --out ../scratch/sumi_loop_fixtures
final class SumiLoopParityTests: XCTestCase {

    struct Manifest: Decodable {
        struct Case: Decodable {
            let name: String
            let promptIds: [Int32]
            let sampler: String
            let tokensPerStep: Int
            let anchorEosbos: Bool
            let denoiseEnd: Int?
            let maxNewTokens: Int

            enum CodingKeys: String, CodingKey {
                case name
                case promptIds = "prompt_ids"
                case sampler
                case tokensPerStep = "tokens_per_step"
                case anchorEosbos = "anchor_eosbos"
                case denoiseEnd = "denoise_end"
                case maxNewTokens = "max_new_tokens"
            }
        }
        let canvasLength: Int
        let numDenoisingSteps: Int
        let cases: [Case]

        enum CodingKeys: String, CodingKey {
            case canvasLength = "canvas_length"
            case numDenoisingSteps = "num_denoising_steps"
            case cases
        }
    }

    static var fixtureDir: URL {
        if let override = ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_LOOP_FIXTURE_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/sumi_loop_fixtures")
    }

    func testDeterministicLoopParity() throws {
        let dir = Self.fixtureDir
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: manifestURL.path),
            "Sumi loop fixtures missing — run Tools/generate_sumi_loop_fixtures.py")

        let manifestData = try Data(contentsOf: manifestURL)
        let manifest = try JSONDecoder().decode(Manifest.self, from: manifestData)
        let manifestJSON = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
        let configJSON = try JSONSerialization.data(
            withJSONObject: manifestJSON["config"] as! [String: Any])
        let config = try JSONDecoder().decode(SumiConfig.self, from: configJSON)

        let weights = try MLX.loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        let tensors = try MLX.loadArrays(url: dir.appendingPathComponent("tensors.safetensors"))

        let model = SumiModel(config: config)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        let engine = SumiEngine(model: model)

        var passed = 0
        for (idx, testCase) in manifest.cases.enumerated() {
            let request = SumiGenerationRequest(
                promptIds: testCase.promptIds,
                maxNewTokens: testCase.maxNewTokens,
                canvasLength: manifest.canvasLength,
                numDenoisingSteps: manifest.numDenoisingSteps,
                sampler: SumiGenerationRequest.Sampler(rawValue: testCase.sampler)!,
                temperature: 0.0,
                tokensPerStep: testCase.tokensPerStep,
                denoiseEnd: testCase.denoiseEnd,
                anchorEOSBOS: testCase.anchorEosbos,
                trimAtEOS: true)

            var stepCanvases: [MLXArray] = []
            let output = engine.generate(
                request,
                initialCanvas: tensors["case\(idx).canvas_init"]!,
                onStep: { _, z in stepCanvases.append(z) })

            // Per-step canvases token-for-token.
            XCTAssertEqual(stepCanvases.count, manifest.numDenoisingSteps, "\(testCase.name): step count")
            for (s, z) in stepCanvases.enumerated() {
                let expected = tensors["case\(idx).step\(s)"]!
                XCTAssertTrue(
                    (z .== expected).all().item(Bool.self),
                    "\(testCase.name): canvas differs at step \(s)")
            }

            // Final canvas + trimmed sequences.
            XCTAssertTrue(
                (output.canvas .== tensors["case\(idx).canvas_final"]!).all().item(Bool.self),
                "\(testCase.name): final canvas differs")
            let expectedSeq = tensors["case\(idx).sequences"]!.asArray(Int32.self)
            XCTAssertEqual(output.sequences, expectedSeq, "\(testCase.name): trimmed output differs")
            passed += 1
        }
        print("[sumi-S1.5] deterministic loop parity: \(passed)/\(manifest.cases.count) cases token-for-token")
    }

    /// Smoke gate for the stochastic ancestral path: runs end-to-end, respects the frozen
    /// prompt/anchors, produces in-vocab ids. Bit-parity is impossible across RNGs (S0.1-1);
    /// the deterministic parts are gated in SumiSamplerTests.
    func testAncestralSmoke() throws {
        let dir = Self.fixtureDir
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try XCTSkipUnless(FileManager.default.fileExists(atPath: manifestURL.path))

        let manifestJSON = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        let configJSON = try JSONSerialization.data(
            withJSONObject: manifestJSON["config"] as! [String: Any])
        let config = try JSONDecoder().decode(SumiConfig.self, from: configJSON)

        let weights = try MLX.loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        let model = SumiModel(config: config)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)

        let promptIds: [Int32] = [5, 6, 7]
        let request = SumiGenerationRequest(
            promptIds: promptIds, maxNewTokens: 16, canvasLength: 32,
            numDenoisingSteps: 4, sampler: .ancestral, temperature: 1.0,
            trimAtEOS: false, seed: 42)
        let output = SumiEngine(model: model).generate(request)

        let canvas = output.canvas.reshaped(-1).asArray(Int32.self)
        XCTAssertEqual(canvas.count, 32)
        XCTAssertEqual(Array(canvas[0 ..< 3]), promptIds, "prompt must stay frozen")
        // Default anchor: EOS,BOS at prompt_len + budget = 3 + 16.
        XCTAssertEqual(canvas[19], Int32(config.eosTokenId), "EOS anchor")
        XCTAssertEqual(canvas[20], Int32(config.bosTokenId), "BOS anchor")
        XCTAssertTrue(canvas.allSatisfy { $0 >= 0 && $0 < Int32(config.vocabSize) }, "ids in vocab")
    }
}
