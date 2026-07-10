import XCTest
import MLX
import MLXNN
@testable import DiffusionCore

/// M3 acceptance (phase-2 §4): every §2 core block is diffed against reference intermediates
/// dumped by `Tools/generate_core_fixtures.py`. Toy config, random seeded weights, FP32.
///
/// Regenerate fixtures with:
///   python3 Tools/generate_core_fixtures.py
final class CoreFixtureTests: XCTestCase {

    // Toy config (scratch/dummy_hf/config.json), mirrored here so the test needs no JSON decode.
    static let hidden = 128
    static let heads = 4
    static let kvHeads = 1
    static let headDim = 32
    static let eps: Float = 1e-6
    static let ropeTheta: Float = 600000
    static let partialRotary: Float = 0.5
    static let numExperts = 4
    static let expertsPerTok = 2
    static let nGroup = 2
    static let topkGroup = 1
    static let scaling: Float = 2.5
    static let moeInter = 64
    static let denseInter = 256
    static let blockLength = 16
    static let seqLength = 48

    static let fixtureDir = URL(fileURLWithPath:
        "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/core_fixtures")

    var weights: [String: MLXArray]!
    var tensors: [String: MLXArray]!

    override func setUpWithError() throws {
        let weightsURL = Self.fixtureDir.appendingPathComponent("weights.safetensors")
        let tensorsURL = Self.fixtureDir.appendingPathComponent("tensors.safetensors")
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: weightsURL.path),
            "Core fixtures missing — run: python3 Tools/generate_core_fixtures.py")
        weights = try MLX.loadArrays(url: weightsURL)
        tensors = try MLX.loadArrays(url: tensorsURL)
    }

    // MARK: - Helpers

    /// Weights under `prefix`, with the prefix stripped from the keys.
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

    /// Asserts `actual ≈ expected` within `atol + rtol*|expected|`, reporting the max deviation.
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
        let worstAllowed = (diff - tol).max().item(Float.self)  // > 0 iff some element exceeds tol
        XCTAssertLessThanOrEqual(
            worstAllowed, 0,
            "\(label): exceeds tolerance (max abs diff \(maxAbs), atol \(atol), rtol \(rtol))",
            file: file, line: line)
    }

    // MARK: - 2.2 RMSNorm

    func testRMSNorm() throws {
        let norm = try load(
            LLaDA2RMSNorm(dimensions: Self.hidden, eps: Self.eps),
            "model.layers.1.input_layernorm.")
        let out = norm(tensors["rmsnorm.input"]!)
        assertClose(out, tensors["rmsnorm.output"]!, atol: 1e-5, rtol: 1e-5, "rmsnorm")
    }

    // MARK: - 2.3 Partial RoPE

    func testPartialRoPE() throws {
        let rope = PartialRotaryEmbedding(
            headDim: Self.headDim, partialRotaryFactor: Self.partialRotary, ropeTheta: Self.ropeTheta)
        let (cos, sin) = rope.cosSin(positionIds: tensors["rope.position_ids"]!)
        assertClose(cos, tensors["rope.cos"]!, atol: 1e-5, rtol: 1e-5, "rope.cos")
        assertClose(sin, tensors["rope.sin"]!, atol: 1e-5, rtol: 1e-5, "rope.sin")

        let qOut = PartialRotaryEmbedding.apply(tensors["rope.q_in"]!, cos: cos, sin: sin)
        let kOut = PartialRotaryEmbedding.apply(tensors["rope.k_in"]!, cos: cos, sin: sin)
        assertClose(qOut, tensors["rope.q_out"]!, atol: 1e-5, rtol: 1e-5, "rope.q_out")
        assertClose(kOut, tensors["rope.k_out"]!, atol: 1e-5, rtol: 1e-5, "rope.k_out")
    }

    // MARK: - Block-diffusion mask

    func testBlockDiffusionMask() {
        let strict = BlockDiffusionMask.build(
            totalLength: Self.seqLength, blockLength: Self.blockLength,
            semantics: .strict, dtype: .float32)
        let ref = tensors["mask.additive"]!  // generator stored the strict 0/-inf form
        XCTAssertEqual(strict.shape, ref.shape)
        // Elementwise equality handles the -inf entries (|-inf - -inf| would be NaN);
        // in IEEE, -inf == -inf is true.
        XCTAssertTrue((strict .== ref).all().item(Bool.self), "mask.strict differs from reference")
    }

    // MARK: - 2.3 Attention (fused QKV split + qk-norm + partial RoPE + GQA SDPA)

    func testAttention() throws {
        let attn = try load(
            LLaDA2Attention(
                hiddenSize: Self.hidden, numHeads: Self.heads, numKVHeads: Self.kvHeads,
                headDim: Self.headDim, rmsNormEps: Self.eps, useQkNorm: true,
                useQkvBias: false, useDenseBias: false),
            "model.layers.1.attention.")
        let rope = PartialRotaryEmbedding(
            headDim: Self.headDim, partialRotaryFactor: Self.partialRotary, ropeTheta: Self.ropeTheta)
        let (cos, sin) = rope.cosSin(positionIds: tensors["rope.position_ids"]!)
        let mask = BlockDiffusionMask.build(
            totalLength: Self.seqLength, blockLength: Self.blockLength,
            semantics: .strict, dtype: .float32)
        let out = attn(tensors["attn.input"]!, mask: mask, cos: cos, sin: sin)
        assertClose(out, tensors["attn.output"]!, atol: 1e-4, rtol: 1e-3, "attention")
    }

    // MARK: - 2.4 Dense FFN (layer 0)

    func testDenseFFN() throws {
        let ffn = try load(
            LLaDA2MLP(hiddenSize: Self.hidden, intermediateSize: Self.denseInter),
            "model.layers.0.mlp.")
        let out = ffn(tensors["ffn.input"]!)
        assertClose(out, tensors["ffn.output"]!, atol: 1e-4, rtol: 1e-4, "dense_ffn")
    }

    // MARK: - 2.5 Router / gate

    func testRouterGate() throws {
        let gate = try load(makeGate(), "model.layers.1.mlp.gate.")
        let input = tensors["gate.input"]!.reshaped(-1, Self.hidden)
        let (indices, gWeights, logits) = gate(input)

        assertClose(logits, tensors["gate.logits"]!, atol: 1e-4, rtol: 1e-4, "gate.logits")

        // Selection + weights are order-independent: scatter into a dense [T, E] vector and diff.
        let denseActual = denseWeights(indices: indices, weights: gWeights)
        let denseExpected = denseWeights(
            indices: tensors["gate.topk_idx"]!, weights: tensors["gate.topk_weight"]!)
        assertClose(denseActual, denseExpected, atol: 1e-4, rtol: 1e-4, "gate.weights")
    }

    // MARK: - 2.5 Full MoE block (router + gathered experts + shared expert)

    func testMoEBlock() throws {
        let block = try loadMoEBlock("model.layers.1.mlp.")
        let out = block(tensors["moe.input"]!)
        assertClose(out, tensors["moe.output"]!, atol: 1e-4, rtol: 1e-3, "moe_block")
    }

    // MARK: - 2.1 Embeddings

    func testEmbeddings() throws {
        let embedding = Embedding(embeddingCount: 1000, dimensions: Self.hidden)
        try embedding.update(
            parameters: ModuleParameters.unflattened(
                ["weight": weights["model.word_embeddings.weight"]!]),
            verify: .all)
        eval(embedding)
        let out = embedding(tensors["embed.ids"]!)
        assertClose(out, tensors["embed.output"]!, atol: 1e-5, rtol: 1e-5, "embeddings")
    }

    // MARK: - 2.6 Output head (FP32 logits)

    func testLMHead() throws {
        let head = Linear(Self.hidden, 1000, bias: false)
        try head.update(
            parameters: ModuleParameters.unflattened(["weight": weights["lm_head.weight"]!]),
            verify: .all)
        eval(head)
        let out = head(tensors["lmhead.input"]!).asType(.float32)
        assertClose(out, tensors["lmhead.logits"]!, atol: 1e-4, rtol: 1e-4, "lm_head")
    }

    // MARK: - Decoder layer composition

    func testDecoderLayers() throws {
        let rope = PartialRotaryEmbedding(
            headDim: Self.headDim, partialRotaryFactor: Self.partialRotary, ropeTheta: Self.ropeTheta)
        let (cos, sin) = rope.cosSin(positionIds: tensors["rope.position_ids"]!)
        let mask = BlockDiffusionMask.build(
            totalLength: Self.seqLength, blockLength: Self.blockLength,
            semantics: .strict, dtype: .float32)

        let layer0 = try loadDecoderLayer("model.layers.0.", dense: true)
        let out0 = layer0(tensors["layer0.input"]!, mask: mask, cos: cos, sin: sin)
        assertClose(out0, tensors["layer0.output"]!, atol: 1e-4, rtol: 1e-3, "layer0")

        let layer1 = try loadDecoderLayer("model.layers.1.", dense: false)
        let out1 = layer1(tensors["layer0.output"]!, mask: mask, cos: cos, sin: sin)
        assertClose(out1, tensors["layer1.output"]!, atol: 1e-4, rtol: 1e-3, "layer1")
    }

    // MARK: - Construction helpers

    private func makeGate() -> LLaDA2MoEGate {
        LLaDA2MoEGate(
            hiddenSize: Self.hidden, numExperts: Self.numExperts,
            numExpertsPerTok: Self.expertsPerTok, nGroup: Self.nGroup,
            topkGroup: Self.topkGroup, routedScalingFactor: Self.scaling)
    }

    private func makeMoEBlock() -> LLaDA2SparseMoEBlock {
        LLaDA2SparseMoEBlock(
            hiddenSize: Self.hidden, moeIntermediateSize: Self.moeInter,
            numExperts: Self.numExperts, numSharedExperts: 1,
            numExpertsPerTok: Self.expertsPerTok, nGroup: Self.nGroup,
            topkGroup: Self.topkGroup, routedScalingFactor: Self.scaling)
    }

    private func makeAttention() -> LLaDA2Attention {
        LLaDA2Attention(
            hiddenSize: Self.hidden, numHeads: Self.heads, numKVHeads: Self.kvHeads,
            headDim: Self.headDim, rmsNormEps: Self.eps, useQkNorm: true,
            useQkvBias: false, useDenseBias: false)
    }

    private func loadMoEBlock(_ prefix: String) throws -> LLaDA2SparseMoEBlock {
        let block = makeMoEBlock()
        let stacked = ExpertWeightStacking.stack(subWeights(prefix))
        try block.update(parameters: ModuleParameters.unflattened(stacked), verify: .all)
        eval(block)
        return block
    }

    private func loadDecoderLayer(_ prefix: String, dense: Bool) throws -> LLaDA2DecoderLayer {
        let mlp: Module = dense
            ? LLaDA2MLP(hiddenSize: Self.hidden, intermediateSize: Self.denseInter)
            : makeMoEBlock()
        let layer = LLaDA2DecoderLayer(
            hiddenSize: Self.hidden, rmsNormEps: Self.eps,
            attention: makeAttention(), mlp: mlp)
        let stacked = ExpertWeightStacking.stack(subWeights(prefix))
        try layer.update(parameters: ModuleParameters.unflattened(stacked), verify: .all)
        eval(layer)
        return layer
    }

    /// Scatter top-k `weights` at `indices` into a dense `[T, numExperts]` vector (FP32).
    private func denseWeights(indices: MLXArray, weights: MLXArray) -> MLXArray {
        let T = indices.dim(0)
        let iota = MLXArray(0 ..< Int32(Self.numExperts))               // [E]
        let oneHot = (indices.expandedDimensions(axis: -1) .== iota)    // [T, k, E]
            .asType(.float32)
        let contrib = oneHot * weights.asType(.float32).expandedDimensions(axis: -1)  // [T, k, E]
        return contrib.sum(axis: 1).reshaped(T, Self.numExperts)
    }
}

extension CoreFixtureTests {
    /// The forward pass uses MLX's fused SDPA (`attend`); this pins its agreement with the
    /// explicit reference attention (`attendReference`) on the block-diffusion 0/-inf mask,
    /// so a future MLX change that breaks the fused path is caught here.
    func testSDPAvsManual() throws {
        let mask = BlockDiffusionMask.build(
            totalLength: Self.seqLength, blockLength: Self.blockLength, semantics: .strict, dtype: .float32)
        let q = tensors["rope.q_out"]!, k = tensors["rope.k_out"]!, v = tensors["rope.k_in"]!
        let scale: Float = pow(Float(Self.headDim), -0.5)
        let fused = LLaDA2Attention.attend(queries: q, keys: k, values: v, scale: scale, mask: mask)
        let reference = LLaDA2Attention.attendReference(queries: q, keys: k, values: v, scale: scale, mask: mask)
        assertCloseFused(fused, reference)
    }

    private func assertCloseFused(_ a: MLXArray, _ b: MLXArray) {
        let d = abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
        XCTAssertLessThanOrEqual(d, 1e-4, "fused SDPA vs reference attention max abs diff \(d)")
    }
}
