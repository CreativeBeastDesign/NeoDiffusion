import XCTest
import MLX
import MLXNN
import DiffusionCore
@testable import DiffusionModel

/// sumi-M3'/S1.3 acceptance (sumi-plan.md §3): full-stack forward parity against the
/// reference `_compute_logits` dump (`forward.*` in the Sumi fixtures) — top-1 agreement
/// 100% plus a max softmax-prob deviation bound.
///
/// Config-driven from manifest.json with a `NEODIFFUSION_SUMI_FIXTURE_DIR` override, so the
/// same test covers the toy FP32 run (dev) and the deferred Studio BF16 real-weight run.
final class SumiForwardParityTests: XCTestCase {

    static var fixtureDir: URL {
        if let override = ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_FIXTURE_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/sumi_core_fixtures")
    }

    func testFullForwardParity() throws {
        let dir = Self.fixtureDir
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: manifestURL.path),
            "Sumi fixtures missing — run: scratch/sumi-venv/bin/python Tools/generate_sumi_fixtures.py")

        let manifestData = try Data(contentsOf: manifestURL)
        let manifestJSON = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
        let configJSON = try JSONSerialization.data(
            withJSONObject: manifestJSON["config"] as! [String: Any])
        let config = try JSONDecoder().decode(SumiConfig.self, from: configJSON)
        let dtype = manifestJSON["dtype"] as? String ?? "float32"

        let weights = try MLX.loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        let tensors = try MLX.loadArrays(url: dir.appendingPathComponent("tensors.safetensors"))

        let model = SumiModel(config: config)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(model)

        let inputIds = tensors["forward.input_ids"]!
        let expected = tensors["forward.logits"]!  // FP32, vocab-truncated

        let logits = model.logits(forTokens: inputIds)
        XCTAssertEqual(logits.shape, expected.shape, "logits shape mismatch")

        // Top-1 agreement must be exact over every position.
        let actualTop1 = logits.argMax(axis: -1)
        let expectedTop1 = expected.argMax(axis: -1)
        let agreement = (actualTop1 .== expectedTop1).sum().item(Int.self)
        let total = expectedTop1.size

        // Softmax-prob deviation (the confidence-relevant quantity for every sampler).
        let maxProbDiff = abs(softmax(logits, axis: -1) - softmax(expected, axis: -1))
            .max().item(Float.self)
        let maxLogitDiff = abs(logits - expected).max().item(Float.self)

        print("[sumi-S1.3] full-forward parity (\(dtype)): top-1 \(agreement)/\(total), "
            + "max |Δprob| \(maxProbDiff), max |Δlogit| \(maxLogitDiff)")

        XCTAssertEqual(agreement, total, "top-1 agreement must be 100%")
        // FP32 toy: measured ~1e-8; bound kept at 1e-2 for BF16 real-weight headroom (M4 style).
        XCTAssertLessThanOrEqual(maxProbDiff, 1e-2, "max softmax-prob deviation")
    }
}
