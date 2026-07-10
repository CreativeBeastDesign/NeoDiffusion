import XCTest
import MLX
import MLXNN
@testable import DiffusionCore

/// sumi-M2' acceptance (sumi-plan.md §3 S1.2): off-by-one softmax formulations agree with
/// each other and with the reference `softmax_one`; the Sumi core modules (full RoPE,
/// RMSNorm at Sumi's ε, SwiGLU MLP, OffByOneAttention) diff against reference intermediates
/// dumped by `Tools/generate_sumi_fixtures.py`.
///
/// Config-driven from the fixture's manifest.json (nothing hardcoded to the toy shape), with
/// a `NEODIFFUSION_SUMI_FIXTURE_DIR` override — the same tests cover the deferred Studio
/// BF16 real-weight run (sumi-plan.md §7).
///
/// Regenerate fixtures with:
///   scratch/sumi-venv/bin/python Tools/generate_sumi_fixtures.py
final class SumiCoreFixtureTests: XCTestCase {

    struct Manifest: Decodable {
        struct Config: Decodable {
            let hiddenSize: Int
            let intermediateSize: Int
            let numAttentionHeads: Int
            let numKeyValueHeads: Int
            let headDim: Int
            let rmsNormEps: Float
            let vocabSize: Int
            let ropeParameters: Rope

            struct Rope: Decodable {
                let ropeTheta: Float
                enum CodingKeys: String, CodingKey { case ropeTheta = "rope_theta" }
            }

            enum CodingKeys: String, CodingKey {
                case hiddenSize = "hidden_size"
                case intermediateSize = "intermediate_size"
                case numAttentionHeads = "num_attention_heads"
                case numKeyValueHeads = "num_key_value_heads"
                case headDim = "head_dim"
                case rmsNormEps = "rms_norm_eps"
                case vocabSize = "vocab_size"
                case ropeParameters = "rope_parameters"
            }
        }
        let config: Config
        let dtype: String
    }

    static var fixtureDir: URL {
        if let override = ProcessInfo.processInfo.environment["NEODIFFUSION_SUMI_FIXTURE_DIR"] {
            return URL(fileURLWithPath: override)
        }
        return URL(fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/sumi_core_fixtures")
    }

    var manifest: Manifest!
    var weights: [String: MLXArray]!
    var tensors: [String: MLXArray]!
    /// FP32 fixtures gate tight (1e-5); BF16 real-weight fixtures get M3-style headroom.
    var atol: Float { manifest.dtype == "float32" ? 1e-5 : 1e-3 }
    var rtol: Float { manifest.dtype == "float32" ? 1e-5 : 1e-2 }

    override func setUpWithError() throws {
        let dir = Self.fixtureDir
        let manifestURL = dir.appendingPathComponent("manifest.json")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: manifestURL.path),
            "Sumi fixtures missing — run: scratch/sumi-venv/bin/python Tools/generate_sumi_fixtures.py")
        manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        weights = try MLX.loadArrays(url: dir.appendingPathComponent("weights.safetensors"))
        tensors = try MLX.loadArrays(url: dir.appendingPathComponent("tensors.safetensors"))
    }

    // MARK: - Helpers (CoreFixtureTests conventions)

    private func subWeights(_ prefix: String) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        for (key, value) in weights where key.hasPrefix(prefix) {
            result[String(key.dropFirst(prefix.count))] = value
        }
        return result
    }

    private func load<M: Module>(_ module: M, _ prefix: String) throws -> M {
        let params = ModuleParameters.unflattened(subWeights(prefix))
        try module.update(parameters: params, verify: .all)
        eval(module)
        return module
    }

    private func assertClose(
        _ actual: MLXArray, _ expected: MLXArray,
        atol: Float, rtol: Float, _ label: String,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertEqual(actual.shape, expected.shape, "\(label): shape mismatch", file: file, line: line)
        let a = actual.asType(.float32)
        let e = expected.asType(.float32)
        let diff = abs(a - e)
        let tol = atol + rtol * abs(e)
        let maxAbs = diff.max().item(Float.self)
        let worstAllowed = (diff - tol).max().item(Float.self)
        XCTAssertLessThanOrEqual(
            worstAllowed, 0,
            "\(label): exceeds tolerance (max abs diff \(maxAbs), atol \(atol), rtol \(rtol))",
            file: file, line: line)
    }

    private func rope() -> PartialRotaryEmbedding {
        PartialRotaryEmbedding(
            headDim: manifest.config.headDim,
            partialRotaryFactor: 1.0,  // Sumi: full rotary
            ropeTheta: manifest.config.ropeParameters.ropeTheta)
    }

    private func makeAttention() -> OffByOneAttention {
        OffByOneAttention(
            hiddenSize: manifest.config.hiddenSize,
            numHeads: manifest.config.numAttentionHeads,
            numKVHeads: manifest.config.numKeyValueHeads,
            headDim: manifest.config.headDim)
    }

    // MARK: - Off-by-one softmax formulations

    /// Formulation A (sink oracle) ≡ formulation B (multiplicative, hot path) ≡ reference
    /// `softmax_one`, including the outlier-logit rows (+40 / −35) where the sink term
    /// underflows / dominates (metal-shader-guide §2.4 training wheels).
    func testSoftmaxOneFormulations() throws {
        let input = tensors["softmax_one.input"]!.asType(.float32)
        let expected = tensors["softmax_one.output"]!

        let multiplicative = softmaxOne(input)
        let sink = softmaxOneSink(input)

        let abDiff = abs(multiplicative - sink).max().item(Float.self)
        XCTAssertLessThanOrEqual(abDiff, 1e-6, "formulation A vs B max abs diff \(abDiff)")

        assertClose(multiplicative, expected, atol: 1e-6, rtol: 1e-6, "softmax_one vs reference")
    }

    // MARK: - Full RoPE (θ=500k, rotaryDim == headDim fast path)

    func testFullRoPE() throws {
        let (cos, sin) = rope().cosSin(positionIds: tensors["rope.position_ids"]!)
        assertClose(cos, tensors["rope.cos"]!, atol: atol, rtol: rtol, "rope.cos")
        assertClose(sin, tensors["rope.sin"]!, atol: atol, rtol: rtol, "rope.sin")

        let qOut = PartialRotaryEmbedding.apply(tensors["rope.q_in"]!, cos: cos, sin: sin)
        let kOut = PartialRotaryEmbedding.apply(tensors["rope.k_in"]!, cos: cos, sin: sin)
        assertClose(qOut, tensors["rope.q_out"]!, atol: atol, rtol: rtol, "rope.q_out")
        assertClose(kOut, tensors["rope.k_out"]!, atol: atol, rtol: rtol, "rope.k_out")
    }

    // MARK: - RMSNorm (reused LLaDA2RMSNorm at Sumi's ε)

    func testRMSNorm() throws {
        let norm = try load(
            LLaDA2RMSNorm(dimensions: manifest.config.hiddenSize, eps: manifest.config.rmsNormEps),
            "model.layers.0.input_layernorm.")
        let out = norm(tensors["rmsnorm.input"]!)
        assertClose(out, tensors["rmsnorm.output"]!, atol: atol, rtol: rtol, "rmsnorm")
    }

    // MARK: - SwiGLU MLP (reused LLaDA2MLP; Sumi is dense, mlp_bias=false)

    func testMLP() throws {
        let mlp = try load(
            LLaDA2MLP(
                hiddenSize: manifest.config.hiddenSize,
                intermediateSize: manifest.config.intermediateSize),
            "model.layers.0.mlp.")
        let out = mlp(tensors["mlp.input"]!)
        assertClose(out, tensors["mlp.output"]!, atol: 1e-4, rtol: 1e-4, "mlp")
    }

    // MARK: - OffByOneAttention (the S1.2 gate)

    func testOffByOneAttention() throws {
        let attn = try load(makeAttention(), "model.layers.0.self_attn.")
        let (cos, sin) = rope().cosSin(positionIds: tensors["rope.position_ids"]!)
        let out = attn(tensors["attn.input"]!, mask: nil, cos: cos, sin: sin)
        assertClose(out, tensors["attn.output"]!, atol: atol, rtol: 1e-3, "off_by_one_attention")
    }

    // MARK: - Decoder layer composition (pre-norm residual wiring)

    func testDecoderLayers() throws {
        let (cos, sin) = rope().cosSin(positionIds: tensors["rope.position_ids"]!)

        func makeLayer() -> SumiDecoderLayer {
            SumiDecoderLayer(
                hiddenSize: manifest.config.hiddenSize,
                intermediateSize: manifest.config.intermediateSize,
                rmsNormEps: manifest.config.rmsNormEps,
                attention: makeAttention())
        }

        let layer0 = try load(makeLayer(), "model.layers.0.")
        let out0 = layer0(tensors["layer0.input"]!, cos: cos, sin: sin)
        assertClose(out0, tensors["layer0.output"]!, atol: 1e-4, rtol: 1e-3, "layer0")

        let layer1 = try load(makeLayer(), "model.layers.1.")
        let out1 = layer1(out0, cos: cos, sin: sin)
        assertClose(out1, tensors["layer1.output"]!, atol: 1e-4, rtol: 1e-3, "layer1")
    }

    /// The hot path (fused SDPA × sigmoid(LSE)) and the sink-oracle attention must agree —
    /// pins the algebraic substitution at the attention level, not just row-wise.
    func testAttendVsSinkOracle() throws {
        let (cos, sin) = rope().cosSin(positionIds: tensors["rope.position_ids"]!)
        let q = PartialRotaryEmbedding.apply(tensors["rope.q_in"]!, cos: cos, sin: sin)
        let k = PartialRotaryEmbedding.apply(tensors["rope.k_in"]!, cos: cos, sin: sin)
        let v = tensors["rope.k_in"]!
        let scale = pow(Float(manifest.config.headDim), -0.5)

        let hot = OffByOneAttention.attend(queries: q, keys: k, values: v, scale: scale, mask: nil)
        let oracle = OffByOneAttention.attendSinkOracle(
            queries: q, keys: k, values: v, scale: scale, mask: nil)
        let d = abs(hot.asType(.float32) - oracle.asType(.float32)).max().item(Float.self)
        XCTAssertLessThanOrEqual(d, 1e-6, "attend vs sink oracle max abs diff \(d)")
    }

    /// The chunked running-max/sum LSE must be exact across tile boundaries — the fixture
    /// sequence fits one default chunk, so force multi-tile with small chunk sizes here
    /// (incl. a non-divisor size to cover the ragged final tile).
    func testChunkedLSEAndFastAttend() throws {
        let (cos, sin) = rope().cosSin(positionIds: tensors["rope.position_ids"]!)
        let q = PartialRotaryEmbedding.apply(tensors["rope.q_in"]!, cos: cos, sin: sin)
        let k = PartialRotaryEmbedding.apply(tensors["rope.k_in"]!, cos: cos, sin: sin)
        let scale = pow(Float(manifest.config.headDim), -0.5)

        let reference = OffByOneAttention.rowLogSumExp(
            queries: q, keys: k, scale: scale, chunkSize: 1_000_000)  // single tile
        for chunk in [4, 7, 8] {  // 7 does not divide seq_len 24: ragged tail covered
            let chunked = OffByOneAttention.rowLogSumExp(
                queries: q, keys: k, scale: scale, chunkSize: chunk)
            let d = abs(chunked - reference).max().item(Float.self)
            XCTAssertLessThanOrEqual(d, 1e-5, "chunked LSE (chunk \(chunk)) max abs diff \(d)")
        }
    }
}
