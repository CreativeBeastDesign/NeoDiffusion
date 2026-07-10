import XCTest
import MLX
import MLXNN
import DiffusionCore
@testable import DiffusionModel

/// Loader-side gates for the converted 4-bit artefact (phase-2 M1 loader + Sumi quirk 3).
///
/// The conversion itself is verified bit-perfect against `mx.quantize` of the BF16 source
/// (2026-07-09, see llada-4bit memory note / phase-2 §3); these tests cover the *Swift*
/// side: the U8→U32 packed-weight reinterpret and the end-to-end quantized load path,
/// at toy scale so they run in the default suite.
final class LLaDA2QuantizedLoaderTests: XCTestCase {

    /// `sanitize` must be an exact byte reinterpret: quantize a matrix, view its packed U32
    /// weight as the artefact's U8 layout, sanitize back, and demand bitwise identity.
    func testSanitizeReinterpretsPackedWeights() throws {
        let w = MLXRandom.normal([64, 128])
        let (wq, scales, maybeBiases) = MLX.quantized(w, groupSize: 64, bits: 4)
        let biases = try XCTUnwrap(maybeBiases, "affine quantization emits biases")
        eval(wq, scales, biases)
        XCTAssertEqual(wq.dtype, .uint32)

        // Simulate the artefact: U8 [out, in/2] byte view + the scales key that marks the
        // tensor as quantized.
        let artefact: [String: MLXArray] = [
            "layer.weight": wq.view(dtype: .uint8),
            "layer.scales": scales,
            "layer.biases": biases,
        ]
        XCTAssertEqual(artefact["layer.weight"]!.shape, [64, 128 / 2])

        let sanitized = DiffusionModel.sanitize(artefact)
        let restored = sanitized["layer.weight"]!
        XCTAssertEqual(restored.dtype, .uint32)
        XCTAssertEqual(restored.shape, wq.shape)
        XCTAssertTrue(allClose(restored.asType(.int64), wq.asType(.int64)).item(Bool.self),
                      "U8→U32 view must restore the packed weight bit-exactly")

        // A U8 tensor with no companion .scales (not quantized) must pass through untouched.
        let plain: [String: MLXArray] = ["other.weight": MLXArray.zeros([4, 4], dtype: .uint8)]
        XCTAssertEqual(DiffusionModel.sanitize(plain)["other.weight"]!.dtype, .uint8)
    }

    /// End-to-end quantized load at toy scale: write an artefact-shaped safetensors file
    /// (per-expert keys, U8-packed weights, F16 scales, F32 router) from a random model,
    /// load it through `loadWeights`, and check logits match the source model.
    func testQuantizedArtefactRoundTrip() throws {
        let config = LLaDA2MoeConfig(
            vocabSize: 128, hiddenSize: 64, intermediateSize: 128, numHiddenLayers: 2,
            numAttentionHeads: 4, numKeyValueHeads: 2, rmsNormEps: 1e-6,
            ropeTheta: 600_000, padTokenId: 126, numExperts: 4, numSharedExperts: 1,
            numExpertsPerTok: 2, nGroup: 2, topkGroup: 1, moeIntermediateSize: 64,
            firstKDenseReplace: 1, headDim: 16, partialRotaryFactor: 0.5)

        // Source model with random weights, quantized in memory — the ground truth.
        let source = DiffusionModel(config: config)
        var randomized: [String: MLXArray] = [:]
        for (key, value) in source.model.parameters().flattened() {
            randomized[key] = MLXRandom.normal(value.shape) * 0.05
        }
        try source.model.update(
            parameters: ModuleParameters.unflattened(randomized), verify: .all)
        source.quantizeModel()
        eval(source.model)

        // Emit the artefact layout: unstack experts back to per-expert keys, view packed
        // U32 weights as U8, cast scales/biases to F16 (the conversion script's dtypes).
        var artefact: [String: MLXArray] = [:]
        for (key, value) in source.model.parameters().flattened() {
            let isPacked = value.dtype == .uint32
            let isScaleOrBias = key.hasSuffix(".scales") || key.hasSuffix(".biases")
            func emit(_ k: String, _ v: MLXArray) {
                if isPacked { artefact[k] = v.view(dtype: .uint8) }
                else if isScaleOrBias { artefact[k] = v.asType(.float16) }
                else { artefact[k] = v }
            }
            if let range = key.range(of: #"experts\."#, options: .regularExpression),
               !key.contains("shared_experts"), value.ndim == 3 {
                for e in 0 ..< value.dim(0) {
                    let perExpert = key.replacingCharacters(in: range, with: "experts.\(e).")
                    emit(perExpert, value[e])
                }
            } else {
                emit(key, value)
            }
        }
        eval(Array(artefact.values))

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("llada-loader-test-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let weightsURL = dir.appendingPathComponent("model.safetensors")
        try MLX.save(arrays: artefact, url: weightsURL)

        // Load through the artefact path and compare logits against the source.
        let loaded = DiffusionModel(config: config)
        try loaded.loadWeights(from: weightsURL)

        let ids = MLXArray((0 ..< 32).map { Int32($0 % 128) }).reshaped(1, 32)
        let expected = source.model.logits(forTokens: ids, blockLength: 16)
        let actual = loaded.model.logits(forTokens: ids, blockLength: 16)
        eval(expected, actual)

        // F16 scales/biases (vs the source's in-memory dtype) bound the tolerance.
        let maxDelta = abs(expected - actual).max().item(Float.self)
        XCTAssertLessThan(maxDelta, 1e-2,
            "quantized artefact round-trip logits diverged (max |Δ| = \(maxDelta))")
    }
}
