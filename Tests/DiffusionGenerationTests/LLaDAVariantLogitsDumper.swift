import XCTest
import Foundation
import MLX
import DiffusionCore
import DiffusionModel
@testable import DiffusionGeneration

/// M8 E5/E6 — variant logits on the FIXED drift corpus (m8-logbook): loads a model
/// variant (lm_head-quantized, or an alternative artefact) and computes logits for the
/// exact windows in `scratch/m8_drift/corpus.safetensors`. Comparing on identical inputs
/// isolates *logit drift* from *trajectory divergence* (a variant's own generations would
/// diverge at the first flipped token — that effect is measured separately via bench arms).
///
/// Env:
///   NEODIFFUSION_M8_VARIANT   required — name for the output file (e.g. "headq4")
///   NEODIFFUSION_M8_MODEL_DIR optional — artefact dir (default the g64 artefact)
///   NEODIFFUSION_M8_QUANT_HEAD=1 optional — apply in-memory lm_head 4-bit after load
///
/// Output: scratch/m8_drift/variant_<name>.safetensors ("<key>.logits" per window)
final class LLaDAVariantLogitsDumper: XCTestCase {

    static let corpusDir = URL(fileURLWithPath:
        "./scratch/m8_drift")
    static let defaultModelDir =
        "./models/llada2-1-mini-4bit"

    func testDumpVariantLogits() throws {
        let env = ProcessInfo.processInfo.environment
        guard let variant = env["NEODIFFUSION_M8_VARIANT"] else {
            throw XCTSkip("opt-in: set NEODIFFUSION_M8_VARIANT=<name> (and optionally "
                + "NEODIFFUSION_M8_MODEL_DIR / NEODIFFUSION_M8_QUANT_HEAD=1)")
        }
        let modelDir = URL(fileURLWithPath: env["NEODIFFUSION_M8_MODEL_DIR"]
            ?? Self.defaultModelDir)
        let container = try DiffusionModel.load(from: modelDir)
        if env["NEODIFFUSION_M8_QUANT_HEAD"] == "1" {
            container.quantizeLMHead()
            print("[m8-variant] lm_head quantized in memory")
        }

        let manifestData = try Data(contentsOf:
            Self.corpusDir.appendingPathComponent("manifest.json"))
        let manifest = try JSONSerialization.jsonObject(with: manifestData) as! [String: Any]
        let windows = manifest["windows"] as! [[String: Any]]
        let corpus = try MLX.loadArrays(url:
            Self.corpusDir.appendingPathComponent("corpus.safetensors"))

        var out: [String: MLXArray] = [:]
        for entry in windows {
            let key = entry["key"] as! String
            let activeLen = entry["activeLen"] as! Int
            let ids = corpus["\(key).ids"]!
            let W = ids.dim(ids.ndim - 1)
            let logits = container.model.logits(forTokens: ids, blockLength: 32)
            let active = logits[0..., (W - activeLen)..., 0...].asType(.float16)
            eval(active)
            out["\(key).logits"] = active
        }
        let url = Self.corpusDir.appendingPathComponent("variant_\(variant).safetensors")
        try MLX.save(arrays: out, url: url)
        print("[m8-variant] wrote \(out.count) windows to \(url.path)")
        XCTAssertEqual(out.count, windows.count)
    }
}
