import XCTest
import MLX
import MLXNN
import DiffusionCore
@testable import DiffusionModel

/// M4 acceptance (phase-2 §4): full-forward parity. The assembled stack (embeddings ->
/// decoder layers -> final norm -> lm_head) is diffed against a **corrected** reference
/// baseline — the reference driven under a strict 0/-inf block mask, never stock `generate()`
/// (§6 mask decision, deviation 8). Fixtures come from `Tools/generate_core_fixtures.py`
/// (`forward.*` tensors).
///
/// Config-driven, so the SAME test covers both M4 acceptance paths:
///   • dev toy config, seeded random weights, FP32 (default `scratch/core_fixtures`);
///   • Studio real weights, BF16 — regenerate on the Studio with
///       python3 Tools/generate_core_fixtures.py --config <hf_dir>/config.json \
///           --weights <hf_dir> --dtype bfloat16 --out <dir>
///     then point the test at it:  NEODIFFUSION_FIXTURE_DIR=<dir> swift test
/// The model config + block length are read from the fixture's `manifest.json`, so nothing
/// is hardcoded to the toy shape.
///
/// Gate: top-1 token agreement 100% over the fixture set; max softmax-prob deviation within
/// tolerance. (The 4-bit path is measured-and-recorded only, per M4 — not covered here.)
final class ForwardParityTests: XCTestCase {

    /// Subset of `manifest.json` the parity test needs. `config` decodes through
    /// ``LLaDA2MoeConfig``'s HF-key `Decodable`, so absent keys take the same defaults HF uses.
    struct FixtureManifest: Decodable {
        let blockLength: Int
        let dtype: String
        let config: LLaDA2MoeConfig

        enum CodingKeys: String, CodingKey {
            case blockLength = "block_length"
            case dtype
            case config
        }
    }

    /// Fixture directory: `NEODIFFUSION_FIXTURE_DIR` if set (Studio BF16 run), else the dev toy set.
    static var fixtureDir: URL {
        if let override = ProcessInfo.processInfo.environment["NEODIFFUSION_FIXTURE_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/core_fixtures")
    }

    var manifest: FixtureManifest!
    var tensors: [String: MLXArray]!
    var model: DiffusionModel!

    override func setUpWithError() throws {
        let dir = Self.fixtureDir
        let weightsURL = dir.appendingPathComponent("weights.safetensors")
        let tensorsURL = dir.appendingPathComponent("tensors.safetensors")
        let manifestURL = dir.appendingPathComponent("manifest.json")

        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsURL.path),
            "Fixtures missing at \(dir.path) — run: python3 Tools/generate_core_fixtures.py")

        manifest = try JSONDecoder().decode(
            FixtureManifest.self, from: Data(contentsOf: manifestURL))
        tensors = try MLX.loadArrays(url: tensorsURL)
        try XCTSkipUnless(
            tensors["forward.logits"] != nil,
            "forward.* fixtures missing — regenerate with: python3 Tools/generate_core_fixtures.py")

        model = DiffusionModel(config: manifest.config)
        try model.loadWeights(from: weightsURL)
    }

    /// Full stack forward vs corrected reference logits.
    func testFullForwardParity() throws {
        let inputIds = tensors["forward.input_ids"]!.asType(.int32)
        let refLogits = tensors["forward.logits"]!.asType(.float32)   // [1, L, V]
        let refArgmax = tensors["forward.argmax"]!.asType(.int32)     // [1, L]

        let logits = model.model.logits(forTokens: inputIds, blockLength: manifest.blockLength)
        eval(logits)

        XCTAssertEqual(logits.shape, refLogits.shape, "forward logits shape mismatch")

        // (1) Top-1 token agreement must be exactly 100% (hard gate, both FP32 and BF16 paths).
        let argmax = logits.argMax(axis: -1).asType(.int32)
        let agree = (argmax .== refArgmax).asType(.float32)
        let agreement = agree.mean().item(Float.self)
        let total = refArgmax.size
        let matched = Int((agree.sum().item(Float.self)).rounded())
        XCTAssertEqual(
            agreement, 1.0,
            "top-1 token agreement \(matched)/\(total) (\(agreement * 100)%) — expected 100%")

        // (2) Max softmax-prob deviation within tolerance. FP32 toy is ~float-epsilon; the
        // 1e-2 bound is the BF16 headroom the milestone anticipated.
        let refProbs = softmax(refLogits, axis: -1)
        let probs = softmax(logits, axis: -1)
        let maxProbDev = abs(probs - refProbs).max().item(Float.self)
        let maxLogitDev = abs(logits - refLogits).max().item(Float.self)
        print("[M4] full-forward parity (\(manifest.dtype), block \(manifest.blockLength)): "
            + "top-1 \(matched)/\(total), max |Δprob| \(maxProbDev), max |Δlogit| \(maxLogitDev)")
        XCTAssertLessThanOrEqual(
            maxProbDev, 1e-2,
            "max softmax-prob deviation \(maxProbDev) exceeds tolerance (max |Δlogit| \(maxLogitDev))")
    }
}
