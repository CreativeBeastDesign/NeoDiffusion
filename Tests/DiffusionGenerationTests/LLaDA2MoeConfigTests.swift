import XCTest
import MLX
import MLXNN
import DiffusionCore
@testable import DiffusionModel

final class LLaDA2MoeConfigTests: XCTestCase {
    
    func testConfigDecodingDefaults() throws {
        // Mock config.json with a few fields overridden, leaving others to use the defaults
        let jsonStr = """
        {
          "vocab_size": 157184,
          "hidden_size": 2048,
          "num_hidden_layers": 2,
          "num_attention_heads": 16,
          "num_key_value_heads": 4,
          "rms_norm_eps": 1e-06,
          "pad_token_id": 156892
        }
        """
        
        let data = jsonStr.data(using: .utf8)!
        let config = try JSONDecoder().decode(LLaDA2MoeConfig.self, from: data)
        
        // Assert overridden fields
        XCTAssertEqual(config.vocabSize, 157184)
        XCTAssertEqual(config.hiddenSize, 2048)
        XCTAssertEqual(config.numHiddenLayers, 2)
        XCTAssertEqual(config.rmsNormEps, 1e-06)
        XCTAssertEqual(config.padTokenId, 156892)
        
        // Assert python-class fallbacks for missing fields
        XCTAssertTrue(config.useQkNorm) // defaults to true
        XCTAssertEqual(config.numExperts, 16) // defaults to 16
        XCTAssertEqual(config.routedScalingFactor, 2.5) // defaults to 2.5
    }
    
    func testModelQuantizationFilter() {
        let config = LLaDA2MoeConfig(
            vocabSize: 1000,
            hiddenSize: 128,
            numHiddenLayers: 2,
            numAttentionHeads: 4,
            numKeyValueHeads: 1,
            numExperts: 4,
            numSharedExperts: 1,
            // Divisible by the 4-bit quantization group size (64); matches the toy fixtures.
            moeIntermediateSize: 64,
            firstKDenseReplace: 1
        )

        let container = DiffusionModel(config: config)
        
        // Before quantization: verify layers are normal Linear
        XCTAssertFalse(container.model.lmHead is QuantizedLinear)

        let preLayer1 = container.model.model.layers[1]
        XCTAssertFalse(preLayer1.attention.queryKeyValue is QuantizedLinear)
        
        // Run model quantization
        container.quantizeModel()
        
        // After quantization:
        // Output head should remain unquantized (Linear)
        XCTAssertFalse(container.model.lmHead is QuantizedLinear)
        
        // Fetch layer reference after quantization completes, to see replacement instances
        let postLayer1 = container.model.model.layers[1]
        
        // Attention QKV and dense layers should be quantized
        XCTAssertTrue(postLayer1.attention.queryKeyValue is QuantizedLinear)
        XCTAssertTrue(postLayer1.attention.dense is QuantizedLinear)
        
        // Shared experts should remain unquantized (Linear)
        let moeBlock = postLayer1.mlp as! LLaDA2SparseMoEBlock
        XCTAssertNotNil(moeBlock.sharedExperts)
        XCTAssertTrue(moeBlock.sharedExperts?.gateProj is Linear)
        XCTAssertFalse(moeBlock.sharedExperts?.gateProj is QuantizedLinear)

        // Routed experts are dispatched through a gathered SwitchLinear and should be
        // quantized (QuantizedSwitchLinear) after the quantization pass.
        XCTAssertTrue(moeBlock.experts.gateProj is QuantizedSwitchLinear)

        // Router gate weight stays FP32 (it is a raw parameter, not a Quantizable module).
        XCTAssertEqual(moeBlock.gate.weight.dtype, .float32)
    }
    
    func testWeightsLoader() throws {
        let mlxDir = URL(fileURLWithPath: "./scratch/dummy_mlx")
        let configURL = mlxDir.appendingPathComponent("config.json")
        let weightsURL = mlxDir.appendingPathComponent("model.safetensors")
        
        // Skip test if dummy weights are not generated yet (setup in command execution)
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            print("Skipping testWeightsLoader: dummy weights not generated at \(weightsURL.path)")
            return
        }
        
        // 1. Decode config
        let configData = try Data(contentsOf: configURL)
        let config = try JSONDecoder().decode(LLaDA2MoeConfig.self, from: configData)
        
        // 2. Load model and load weights
        let container = DiffusionModel(config: config)
        try container.loadWeights(from: weightsURL)
        
        // 3. Verify that the parameters are loaded and shapes match
        let layer1 = container.model.model.layers[1]
        
        // queryKeyValue in layer 1 should be a QuantizedLinear
        XCTAssertTrue(layer1.attention.queryKeyValue is QuantizedLinear)
        let qkv = layer1.attention.queryKeyValue as! QuantizedLinear
        
        // Shape of QKV weight in dummy config: Q=4 heads * 32 dim, K=1 head * 32 dim, V=1 head * 32 dim -> QKV total dims = 6 * 32 = 192.
        // Quantized 4-bit weight shape is [192, 128 / 8] -> [192, 16]
        XCTAssertEqual(qkv.weight.shape, [192, 16])
        // Scales shape is [192, 128 / 64] -> [192, 2]
        XCTAssertEqual(qkv.scales.shape, [192, 2])
        
        // lmHead stays unquantized (Linear); verify its weight shape survived loading.
        XCTAssertEqual(container.model.lmHead.weight.shape, [1000, 128])
    }
}
