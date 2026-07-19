import Foundation
import MLX
import MLXNN
import Tokenizers
import DiffusionCore

/// Backward-compatibility wrapper for early stubs.
public struct DiffusionModelConfig {
    public let vocabSize: Int
    public let hiddenDim: Int
    public let numLayers: Int

    public init(vocabSize: Int = 32000, hiddenDim: Int = 1024, numLayers: Int = 12) {
        self.vocabSize = vocabSize
        self.hiddenDim = hiddenDim
        self.numLayers = numLayers
    }

    public func toLLaDA2MoeConfig() -> LLaDA2MoeConfig {
        return LLaDA2MoeConfig(
            vocabSize: vocabSize,
            hiddenSize: hiddenDim,
            numHiddenLayers: numLayers
        )
    }
}

/// The DiffusionModel container housing the LLaDA2 MoE model config, modules, and loading logic.
public class DiffusionModel {
    public let config: LLaDA2MoeConfig
    public let model: LLaDA2MoeModel
    public var tokenizer: DiffusionTokenizer?

    public init(config: LLaDA2MoeConfig) {
        self.config = config
        self.model = LLaDA2MoeModel(config: config)
        self.tokenizer = nil
    }

    /// Convenience initializer to support legacy stubs.
    public convenience init(config: DiffusionModelConfig) {
        self.init(config: config.toLLaDA2MoeConfig())
    }

    /// Quantization metadata the conversion script records in the artefact's `config.json`
    /// (`Tools/convert_weights_streaming.py`). Decoded separately from ``LLaDA2MoeConfig``
    /// because it is NeoDiffusion artefact metadata, not an HF configuration key.
    public struct QuantizationConfig: Codable {
        public let bits: Int
        public let groupSize: Int
        /// Routed-expert overrides (M8 sweep artefacts: g32 / 6-bit experts). Absent in
        /// the uniform g64 artefact → experts use `bits`/`groupSize`.
        public let expertBits: Int?
        public let expertGroupSize: Int?
        /// Quant format. Absent in pre-mxfp4 artefacts → `.affine`. `.mxfp4` (Step 4c-ii
        /// diagnostic) stores e8m0 scales and no biases; the load path threads this into
        /// `MLXNN.quantize` so the built modules match the artefact's tensor set.
        public let mode: QuantizationMode?
        enum CodingKeys: String, CodingKey {
            case bits
            case groupSize = "group_size"
            case expertBits = "expert_bits"
            case expertGroupSize = "expert_group_size"
            case mode
        }
    }

    private struct ArtefactConfig: Codable {
        let quantization: QuantizationConfig?
    }

    /// Loads a converted model directory (`config.json` + `model.safetensors`), the standard
    /// entry point for the 4-bit dev artefact (phase-2 §3). Quantization parameters come from
    /// the artefact's `quantization` block (default 4-bit group-64).
    public static func load(from directory: URL) throws -> DiffusionModel {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        let config = try JSONDecoder().decode(LLaDA2MoeConfig.self, from: configData)
        let quant = try JSONDecoder().decode(ArtefactConfig.self, from: configData).quantization
        let container = DiffusionModel(config: config)
        try container.loadWeights(
            from: directory.appendingPathComponent("model.safetensors"),
            quantization: quant)
        return container
    }

    /// Loads weights from a safetensors file and applies them to the model.
    ///
    /// The checkpoint stores routed experts per-expert (`...experts.{e}.gate_proj.weight`);
    /// this is stacked into the gathered `[numExperts, out, in]` layout the `SwitchLinear`
    /// blocks expect (phase-2 §2.5, `gather_qmm` dispatch). If the checkpoint is quantized
    /// (contains `.scales` keys), matching layers are converted to their quantized forms and
    /// packed weights normalised to MLX's layout (see `sanitize`) before the parameter update.
    public func loadWeights(from url: URL, quantization: QuantizationConfig? = nil) throws {
        var rawArrays = try MLX.loadArrays(url: url)

        let isQuantized = rawArrays.keys.contains { $0.hasSuffix(".scales") }
        if isQuantized {
            quantizeModel(
                groupSize: quantization?.groupSize ?? 64, bits: quantization?.bits ?? 4,
                expertGroupSize: quantization?.expertGroupSize,
                expertBits: quantization?.expertBits,
                quantMode: quantization?.mode ?? .affine)
            rawArrays = Self.sanitize(rawArrays)
        }

        var arrays = ExpertWeightStacking.stack(rawArrays)

        // NEODIFFUSION_PRECAST_SCALES=1 — diagnostic, default off, **REJECTED 2026-07-19**
        // (step5 logbook post-close addendum): the premise was that serving x is f32, making
        // `gather_qmm`'s per-call `astype(scales, out_type)` cost ~2.9 GB/forward. Measured:
        // serving x is f16 → promote(f16, f16) = f16 → the casts short-circuit already and
        // stock pays NOTHING. Flipping this flag FORCES f32 promotion instead: +0.89 GB peak,
        // ms/forward wash (0.996), and real trajectory changes (10/12 prompts) from the
        // precision increase — measured `scratch/precast/`. Kept only as the measurement's
        // provenance; do not enable in serving.
        if ProcessInfo.processInfo.environment["NEODIFFUSION_PRECAST_SCALES"] == "1" {
            var castCount = 0
            for (key, value) in arrays
            where key.contains("experts.")
                && (key.hasSuffix(".scales") || key.hasSuffix(".biases"))
                && value.dtype != .float32
            {
                arrays[key] = value.asType(.float32)
                castCount += 1
            }
            // Effective echo (AGENTS.md): logs must show the pre-cast actually applied.
            print("[precast-scales] routed-expert scales/biases pre-cast to f32 (\(castCount) tensors)")
        }

        let parameters = ModuleParameters.unflattened(arrays)
        try model.update(parameters: parameters, verify: .all)
        eval(model)
    }

    /// Converts the quantizable Linear / SwitchLinear layers to their quantized forms.
    ///
    /// Kept 16-bit per §2.7: output head, shared experts. The router gate weight is a raw
    /// FP32 parameter (not a Module), so it is never touched here. `expertGroupSize`/
    /// `expertBits` override the routed experts only (M8 sweep artefacts) — realized via
    /// MLXNN's per-module-params `quantize(model:filter:)` overload in a **single** pass:
    /// a second pass touching only the MoE layers is a *sparse* `update(modules:)` on the
    /// `layers` array, which MLXNN rejects with `unexpectedStructure` (found by test —
    /// m8-logbook, `testModelQuantizationFilter`).
    public func quantizeModel(
        groupSize: Int = 64, bits: Int = 4,
        expertGroupSize: Int? = nil, expertBits: Int? = nil,
        quantMode: QuantizationMode = .affine
    ) {
        // NB: the parameter is `quantMode`, not `mode`. The filter's return tuple has a
        // `mode:` label; a parameter also named `mode` gets shadowed by that label inside
        // the closure and silently resolves to the enum's first-declared-elsewhere value
        // (observed: mxfp4 leaked in, crashing `[quantize] mxfp4 requires group 32`).
        MLXNN.quantize(
            model: model,
            filter: { path, module -> (groupSize: Int, bits: Int, mode: QuantizationMode)? in
                if path.contains("lm_head") || path.contains("shared_experts") {
                    return nil
                }
                if module is SwitchLinear {
                    return (expertGroupSize ?? groupSize, expertBits ?? bits, quantMode)
                }
                if module is Linear {
                    return (groupSize, bits, quantMode)
                }
                return nil
            }
        )
    }

    /// Quantizes the (16-bit-loaded) output head in memory — the artefact keeps `lm_head`
    /// 16-bit per §2.7, so this must run **after** `loadWeights`. M8 sweep axis (E5):
    /// confidence quality is sensitive to head precision; gate with the margin-conditioned
    /// drift analysis (m8-logbook) before adopting. No artefact change needed (Sumi quirk 6).
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
    /// byte view), while MLX's quantized layers expect `U32 [out, in·bits/32]`. The byte
    /// content is identical (little-endian, low-nibble-first) — reinterpret in place (Sumi
    /// campaign quirk 3; validated against a ground-truth `mx.quantize` dequantize).
    /// Runs on the raw per-expert keys, before ``ExpertWeightStacking``.
    static func sanitize(_ arrays: [String: MLXArray]) -> [String: MLXArray] {
        var result = arrays
        for (key, value) in arrays where key.hasSuffix(".weight") && value.dtype == .uint8 {
            guard arrays["\(String(key.dropLast(7))).scales"] != nil else { continue }
            result[key] = value.view(dtype: .uint32)
        }
        return result
    }
}
