import Foundation
import MLX
import MLXNN
import DiffusionCore

/// Container housing the Sumi model config, modules, and loading logic for the converted
/// 4-bit artefact (sumi-plan.md §4 S2). Mirrors the LLaDA `DiffusionModel` container.
public class SumiDiffusionModel {
    public let config: SumiConfig
    public let model: SumiModel

    public init(config: SumiConfig) {
        self.config = config
        self.model = SumiModel(config: config)
    }

    /// Loads a converted model directory (`config.json` + `model.safetensors`).
    public static func load(from directory: URL) throws -> SumiDiffusionModel {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(SumiConfig.self, from: configData)
        let container = SumiDiffusionModel(config: config)
        try container.loadWeights(from: directory.appendingPathComponent("model.safetensors"))
        return container
    }

    /// Loads weights from a safetensors file. If the checkpoint is quantized (`.scales` keys
    /// present), quantizable layers are converted first and packed quantized weights are
    /// normalised to MLX's layout (see `sanitize`).
    public func loadWeights(from url: URL) throws {
        var arrays = try MLX.loadArrays(url: url)

        let isQuantized = arrays.keys.contains { $0.hasSuffix(".scales") }
        if isQuantized {
            let quant = config.quantization
            quantizeModel(groupSize: quant?.groupSize ?? 64, bits: quant?.bits ?? 4)
            arrays = Self.sanitize(arrays)
        }

        let parameters = ModuleParameters.unflattened(arrays)
        try model.update(parameters: parameters, verify: .all)
        eval(model)
    }

    /// Converts the quantizable Linear layers to their quantized forms. Kept 16-bit per the
    /// sumi-plan §4 keep-list: `lm_head` and embeddings (Embedding is not a Linear, so only
    /// the head needs excluding); norm weights are raw parameters, never touched.
    public func quantizeModel(groupSize: Int = 64, bits: Int = 4) {
        MLXNN.quantize(
            model: model,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            filter: { path, module in
                guard module is Linear else { return false }
                if path.contains("lm_head") { return false }
                return true
            }
        )
    }

    /// Quantizes the (F16-loaded) output head in memory — the checkpoint keeps `lm_head`
    /// 16-bit per the keep-list, so this must run **after** `loadWeights`. Confidence
    /// quality is sensitive to head precision (M8-sweep caveat inherited from the LLaDA
    /// plan); gate with the SumiLMHeadPrecisionTests quality eval before adopting.
    public func quantizeLMHead(groupSize: Int = 64, bits: Int = 4) {
        MLXNN.quantize(
            model: model,
            groupSize: groupSize,
            bits: bits,
            mode: .affine,
            filter: { path, module in module is Linear && path.contains("lm_head") }
        )
        eval(model)
    }

    /// The converted artefact stores packed 4-bit weights as `U8 [out, in/2]` (safetensors
    /// byte view), while MLX's `QuantizedLinear` expects `U32 [out, in·bits/32]`. The byte
    /// content is identical (little-endian, low-nibble-first) — reinterpret in place.
    static func sanitize(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var result = arrays
        for (key, value) in arrays where key.hasSuffix(".weight") && value.dtype == .uint8 {
            guard arrays["\(String(key.dropLast(7))).scales"] != nil else { continue }
            result[key] = value.view(dtype: .uint32)
        }
        return result
    }
}
