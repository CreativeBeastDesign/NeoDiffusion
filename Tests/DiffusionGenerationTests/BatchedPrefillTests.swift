import XCTest
import MLX
import MLXNN
import DiffusionCore
@testable import DiffusionModel
@testable import DiffusionGeneration

/// E-012 batched prefill (`GenerationParams.batchedPrefillBlocks`, default 1 = off).
///
/// Batched prefill captures several prompt blocks per forward under the block-causal
/// `BlockDiffusionMask.activeWindowMask` and commits the chunk's K/V block by block. It is
/// mathematically identical to per-block prefill; these toy-config FP32 gates check that:
///   • cache-level: every layer's committed K/V after a batched prefill matches the per-block
///     prefill to FP32 round-off, for several chunk sizes incl. chunk > prompt;
///   • KV-quantized modes: same committed length, dequantized K/V close (a ~1e-6 input change can
///     flip an int4 rounding bucket, so bitwise equality is not expected);
///   • end to end: `generateCached` output and steps identical on FP32, and the effective echo
///     (`effectiveBatchedPrefillBlocks`, `prefillForwards`) reports what ran.
final class BatchedPrefillTests: XCTestCase {

    static let B = 4
    static let genLength = 16
    /// 7 whole prompt blocks + a 2-token tail sharing the first generated block.
    static func prompt(_ B: Int) -> [Int] { (0 ..< (7 * B + 2)).map { 3 + ($0 * 37) % 190 } }

    /// head_dim 16 for the FP32 gates; the KV-quantized gate needs head_dim 64 (MLX quantize
    /// groups of 64 along the last axis), as in `SessionPrefixCacheTests.randomHeadDim64Model`.
    private func makeModel(headDim: Int = 16) -> LLaDA2MoeModel {
        let config = LLaDA2MoeConfig(
            vocabSize: 200, hiddenSize: 64, intermediateSize: 128, numHiddenLayers: 2,
            numAttentionHeads: 4, numKeyValueHeads: 2, rmsNormEps: 1e-6,
            ropeTheta: 600_000, padTokenId: 198, numExperts: 4, numSharedExperts: 1,
            numExpertsPerTok: 2, nGroup: 2, topkGroup: 1, moeIntermediateSize: 32,
            firstKDenseReplace: 1, headDim: headDim, partialRotaryFactor: 0.5)
        let container = DiffusionModel(config: config)
        MLXRandom.seed(11)
        var randomized: [String: MLXArray] = [:]
        for (key, value) in container.model.parameters().flattened() {
            randomized[key] = MLXRandom.normal(value.shape) * 0.05
        }
        try! container.model.update(
            parameters: ModuleParameters.unflattened(randomized), verify: .all)
        return container.model
    }

    /// Per-block prefill exactly as the engine's default path does it.
    private func prefillPerBlock(_ model: LLaDA2MoeModel, quant: KVCacheQuantization, B: Int = BatchedPrefillTests.B) -> ExactPrefixCache {
        let prompt = Self.prompt(B)
        let cache = ExactPrefixCache(layerCount: model.layerCount, usePool: false,
                                     initialPoolCapacity: 256, quantization: quant)
        for b in 0 ..< prompt.count / B {
            let ids = MLXArray(prompt[b * B ..< (b + 1) * B].map { Int32($0) }).reshaped(1, B)
            let pos = MLXArray(Int32(b * B) ..< Int32((b + 1) * B)).expandedDimensions(axis: 0)
            _ = model(ids, positionIds: pos, caches: cache.layers)
            cache.commitBlock()
        }
        return cache
    }

    /// Mirrors the engine's batched chunk loop (`DiffusionEngine+Entry.swift`) at cache level;
    /// `testGenerateCachedIdenticalAndEchoed` covers the engine path itself.
    private func prefillBatched(_ model: LLaDA2MoeModel, chunk: Int, quant: KVCacheQuantization, B: Int = BatchedPrefillTests.B) -> ExactPrefixCache {
        let prompt = Self.prompt(B)
        let cache = ExactPrefixCache(layerCount: model.layerCount, usePool: false,
                                     initialPoolCapacity: 256, quantization: quant)
        let blocks = prompt.count / B
        var b = 0
        while b < blocks {
            let n = min(chunk, blocks - b)
            let ids = MLXArray(prompt[b * B ..< (b + n) * B].map { Int32($0) }).reshaped(1, n * B)
            let pos = MLXArray(Int32(b * B) ..< Int32((b + n) * B)).expandedDimensions(axis: 0)
            let mask: MLXArray? = n > 1
                ? BlockDiffusionMask.activeWindowMask(prefixLen: cache.committedLength, activeLen: n * B, blockLength: B)
                : nil
            _ = model(ids, positionIds: pos, caches: cache.layers, mask: mask)
            let kv = cache.layers.map { ($0.pendingKeys!, $0.pendingValues!) }
            for j in 0 ..< n {
                for (layer, p) in zip(cache.layers, kv) {
                    layer.pendingKeys = p.0[0..., 0..., (j * B) ..< ((j + 1) * B), 0...]
                    layer.pendingValues = p.1[0..., 0..., (j * B) ..< ((j + 1) * B), 0...]
                }
                cache.commitBlock()
            }
            b += n
        }
        return cache
    }

    private func maxAbsDiff(_ a: MLXArray, _ b: MLXArray) -> Float {
        abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self)
    }

    func testCommittedKVMatchesPerBlockPrefillFP32() {
        let model = makeModel()
        let ref = prefillPerBlock(model, quant: .none)
        for chunk in [2, 3, 7, 16] {
            let got = prefillBatched(model, chunk: chunk, quant: .none)
            XCTAssertEqual(got.committedLength, ref.committedLength, "chunk \(chunk): committed length")
            for (l, (g, r)) in zip(got.layers, ref.layers).enumerated() {
                let dk = maxAbsDiff(g.keys!, r.keys!), dv = maxAbsDiff(g.values!, r.values!)
                XCTAssertLessThan(dk, 1e-5, "chunk \(chunk) layer \(l): keys max |Δ| \(dk)")
                XCTAssertLessThan(dv, 1e-5, "chunk \(chunk) layer \(l): values max |Δ| \(dv)")
            }
        }
    }

    func testCommittedKVCloseUnderKVQuantization() {
        let model = makeModel(headDim: 64)
        for quant in [KVCacheQuantization.k4v8, .kivi4, .k8v4] {
            let ref = prefillPerBlock(model, quant: quant, B: 32)
            let got = prefillBatched(model, chunk: 3, quant: quant, B: 32)
            XCTAssertEqual(got.committedLength, ref.committedLength, "\(quant): committed length")
            for (l, (g, r)) in zip(got.layers, ref.layers).enumerated() {
                let scale = abs(r.keys!.asType(.float32)).max().item(Float.self)
                let dk = maxAbsDiff(g.keys!, r.keys!), dv = maxAbsDiff(g.values!, r.values!)
                // At most a rounding-bucket flip on a few elements: bound relative to the range.
                XCTAssertLessThan(dk, 0.2 * scale, "\(quant) layer \(l): keys max |Δ| \(dk) (range \(scale))")
                XCTAssertLessThan(dv, 0.2 * max(scale, 1e-3), "\(quant) layer \(l): values max |Δ| \(dv)")
            }
        }
    }

    func testGenerateCachedIdenticalAndEchoed() {
        let model = makeModel()
        var base = GenerationParams.mode(.q, blockLength: Self.B, genLength: Self.genLength,
                                         maskId: 199, eosId: 198)
        base.batchedPrefillBlocks = 1
        let ref = DiffusionEngine(model: model, speculationK: 1).generateCached(prompt: Self.prompt(Self.B), params: base)
        XCTAssertEqual(ref.metrics.effectiveBatchedPrefillBlocks, 1)
        XCTAssertEqual(ref.metrics.prefillForwards, 7)
        for chunk in [2, 3, 7, 16] {
            var p = base
            p.batchedPrefillBlocks = chunk
            let out = DiffusionEngine(model: model, speculationK: 1).generateCached(prompt: Self.prompt(Self.B), params: p)
            XCTAssertEqual(out.finalSequence, ref.finalSequence, "chunk \(chunk): final sequence")
            XCTAssertEqual(out.stepsPerBlock, ref.stepsPerBlock, "chunk \(chunk): steps per block")
            XCTAssertEqual(out.metrics.effectiveBatchedPrefillBlocks, chunk)
            XCTAssertEqual(out.metrics.prefillForwards, (7 + chunk - 1) / chunk, "chunk \(chunk): prefill forwards")
        }
    }
}
