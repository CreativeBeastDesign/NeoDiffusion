import XCTest
@testable import DiffusionModel

/// sumi-M1' acceptance (sumi-plan.md §3 S1.1): the config decodes the real
/// `Tools/reference/sumi/config.json` with the documented values, and absent keys take the
/// Python-class defaults from `configuration_sumi.py` (incl. the `__post_init__` resolution
/// of `head_dim` / `num_key_value_heads`).
final class SumiConfigTests: XCTestCase {

    static let referenceConfigURL = URL(
        fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/Tools/reference/sumi/config.json")

    func testSumiConfigLoads() throws {
        let data = try Data(contentsOf: Self.referenceConfigURL)
        let config = try JSONDecoder().decode(SumiConfig.self, from: data)

        XCTAssertEqual(config.hiddenSize, 4096)
        XCTAssertEqual(config.numHiddenLayers, 36)
        XCTAssertEqual(config.vocabSize, 100278)
        XCTAssertEqual(config.numAttentionHeads, 32)
        XCTAssertEqual(config.numKeyValueHeads, 8)
        XCTAssertEqual(config.headDim, 128)
        XCTAssertEqual(config.intermediateSize, 12288)
        XCTAssertEqual(config.maxPositionEmbeddings, 4864)
        XCTAssertEqual(config.ropeTheta, 500_000)
        XCTAssertEqual(config.ropeParameters.ropeType, "default")
        // config.json explicitly overrides the 1e-6 Python default.
        XCTAssertEqual(config.rmsNormEps, 1e-5)
        XCTAssertEqual(config.bosTokenId, 100256)
        XCTAssertEqual(config.eosTokenId, 100257)
        XCTAssertEqual(config.padTokenId, 100277)
        XCTAssertFalse(config.tieWordEmbeddings)
        XCTAssertFalse(config.attentionBias)
        XCTAssertFalse(config.addQkvBias)
        XCTAssertFalse(config.qkvBias)
        XCTAssertFalse(config.mlpBias)
        // Present in config.json as `true`, but generation always runs cache-disabled
        // (generation_sumi.py passes use_cache=False on every forward).
        XCTAssertTrue(config.useCache)
    }

    func testSumiConfigDefaults() throws {
        // Minimal config: absent keys must take configuration_sumi.py defaults.
        let jsonStr = """
            {
              "vocab_size": 200,
              "hidden_size": 64,
              "num_hidden_layers": 2,
              "num_attention_heads": 4
            }
            """
        let config = try JSONDecoder().decode(SumiConfig.self, from: jsonStr.data(using: .utf8)!)

        // __post_init__ semantics: head_dim = hidden/heads, kv heads = heads (MHA).
        XCTAssertEqual(config.headDim, 16)
        XCTAssertEqual(config.numKeyValueHeads, 4)
        // Python-class defaults.
        XCTAssertEqual(config.rmsNormEps, 1e-6)
        XCTAssertEqual(config.intermediateSize, 11008)
        XCTAssertEqual(config.maxPositionEmbeddings, 2048)
        XCTAssertFalse(config.useCache)
        XCTAssertEqual(config.bosTokenId, 1)
        XCTAssertEqual(config.eosTokenId, 2)
        XCTAssertNil(config.padTokenId)
        XCTAssertEqual(config.ropeTheta, 10000.0)
        XCTAssertEqual(config.attentionDropout, 0.0)
    }
}
