import XCTest
import MLX
import MLXNN
import DiffusionCore
import DiffusionModel
@testable import DiffusionGeneration

/// S4.2 correctness gates for the SchED-style early exit (toy model, fast).
///
/// Soundness property used for greedy: the greedy sampler is a deterministic map
/// `z_{t+1} = f(z_t)`; once predictions are unchanged for `stableSteps` consecutive steps
/// the canvas is at a fixed point and every further step is a no-op — so an early-exited
/// run must produce the **identical final canvas** to the capped full run.
final class SumiEarlyExitTests: XCTestCase {

    static var fixtureDir: URL {
        if let override = ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_LOOP_FIXTURE_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath:
            "./scratch/sumi_loop_fixtures")
    }

    private func makeEngine() throws -> (SumiEngine, SumiConfig) {
        let dir = Self.fixtureDir
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: manifestURL.path),
            "Sumi loop fixtures missing — run Tools/generate_sumi_loop_fixtures.py")
        let manifestJSON = try JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestURL)) as! [String: Any]
        let configJSON = try JSONSerialization.data(
            withJSONObject: manifestJSON["config"] as! [String: Any])
        let config = try JSONDecoder().decode(SumiConfig.self, from: configJSON)
        let weights = try MLX.loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        let model = SumiModel(config: config)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)
        return (SumiEngine(model: model), config)
    }

    func testGreedyEarlyExitMatchesFullRun() throws {
        let (engine, _) = try makeEngine()
        let cap = 20

        func request(earlyExit: SumiGenerationRequest.EarlyExit?) -> SumiGenerationRequest {
            SumiGenerationRequest(
                promptIds: [5, 6, 7], maxNewTokens: 16, canvasLength: 32,
                numDenoisingSteps: cap, sampler: .greedy, temperature: 0,
                trimAtEOS: false, seed: 3, earlyExit: earlyExit)
        }

        let full = engine.generate(request(earlyExit: nil))
        let early = engine.generate(
            request(earlyExit: SumiGenerationRequest.EarlyExit(stableSteps: 2)))

        XCTAssertEqual(full.stepsExecuted, cap)
        XCTAssertLessThanOrEqual(early.stepsExecuted, cap)
        XCTAssertTrue(
            (early.canvas .== full.canvas).all().item(Bool.self),
            "early-exit canvas must equal the full run's (greedy fixed point); "
                + "exited after \(early.stepsExecuted)/\(cap)")
        if early.stepsExecuted < cap {
            print("[sumi-S4.2] greedy early exit: \(early.stepsExecuted)/\(cap) steps, canvas identical")
        } else {
            print("[sumi-S4.2] greedy did not stabilise within \(cap) steps on the toy model (no exit)")
        }
    }

    func testAncestralEarlyExitSmoke() throws {
        let (engine, config) = try makeEngine()
        let out = engine.generate(
            SumiGenerationRequest(
                promptIds: [5, 6, 7], maxNewTokens: 16, canvasLength: 32,
                numDenoisingSteps: 24, sampler: .ancestral, temperature: 1.0,
                trimAtEOS: false, seed: 9,
                earlyExit: SumiGenerationRequest.EarlyExit(stableSteps: 3, minSteps: 4)))
        let canvas = out.canvas.reshaped(-1).asArray(Int32.self)
        XCTAssertEqual(Array(canvas[0 ..< 3]), [5, 6, 7], "prompt frozen")
        XCTAssertTrue(canvas.allSatisfy { $0 >= 0 && $0 < Int32(config.vocabSize) })
        XCTAssertGreaterThanOrEqual(out.stepsExecuted, 4, "minSteps floor respected")
        print("[sumi-S4.2] ancestral early-exit smoke: \(out.stepsExecuted)/24 steps")
    }
}
