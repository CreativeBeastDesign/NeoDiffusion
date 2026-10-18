import Foundation
import MLX
import DiffusionCore
import DiffusionModel
import DiffusionGeneration

public enum ConnectionTarget: Codable, Equatable {
    case local
    case remote(host: String, port: Int)
    
    public var description: String {
        switch self {
        case .local:
            return "Local Engine"
        case .remote(let host, let port):
            return "\(host):\(port)"
        }
    }
}

public protocol InferenceBackend: AnyObject {
    var name: String { get }
    var isLocal: Bool { get }
    
    func generate(
        prompt: String,
        config: SettingsSchema,
        onTrace: @escaping (DiffusionEngine.StepTrace) -> Void,
        onTokenStream: @escaping ([Int]) -> Void,
        onTextStream: @escaping (String) -> Void
    ) async throws -> DiffusionEngine.Output
}

extension GenerationParams {
    public static func from(schema: SettingsSchema, maskId: Int, eosId: Int) -> GenerationParams {
        let kvQuant = KVCacheQuantization(rawValue: schema.kvQuantization) ?? .none
        var params = GenerationParams(
            threshold: schema.threshold,
            editingThreshold: schema.editingThreshold,
            maxPostSteps: schema.maxPostSteps,
            numToTransfer: schema.numToTransfer,
            eosEarlyStop: schema.eosEarlyStop,
            temperature: schema.temperature,
            blockLength: schema.blockLength,
            genLength: schema.genLength,
            maskId: maskId,
            eosId: eosId,
            nBuf: schema.nBuf,
            tauAdd: schema.tauAdd,
            tauSemi: schema.tauSemi,
            speculation: schema.speculation == "s2d2" ? .s2d2 : .none,
            tauSpan: schema.tauSpan,
            dynamicTauAlpha: schema.dynamicTauAlpha,
            eosEarlyExit: schema.eosEarlyExit,
            elasticCacheEnabled: false,
            jotEnabled: schema.jotEnabled,
            jotK: schema.jotK,
            jotThreshold: schema.jotThreshold,
            jotFaithful: schema.jotFaithful,
            moeCapacityRatio: schema.moeCapacityRatio,
            subBlockCommit: schema.subBlockCommit,
            subBlockMinPrefix: schema.subBlockMinPrefix,
            flashBlockEnabled: schema.flashBlockEnabled,
            flashBlockTau: schema.flashBlockTau,
            temporalVotingEnabled: schema.temporalVotingEnabled,
            temporalVotingAlpha: schema.temporalVotingAlpha,
            temporalVotingCutoff: schema.temporalVotingCutoff,
            iceEnabled: schema.iceEnabled,
            iceTau: schema.iceTau,
            iceNt: schema.iceNt,
            iceThinkingLength: schema.iceThinkingLength,
            creditDecodingEnabled: schema.creditDecodingEnabled,
            creditAlpha: schema.creditAlpha,
            creditBeta: schema.creditBeta,
            creditGamma: schema.creditGamma,
            csa2Enabled: schema.csa2Enabled,
            kvQuantization: kvQuant,
            skipRedundantCommitForward: schema.skipRedundantCommitForward
        )
        if let preset = CachePreset(rawValue: schema.cachePreset), preset != .exactFast {
            params.applyCachePreset(preset)
        }
        return params
    }
}

public class LocalInferenceBackend: InferenceBackend {
    public let name = "Local Engine"
    public let isLocal = true
    
    public var engine: DiffusionEngine?
    public var tokenizer: DiffusionTokenizer?
    private var isLoaded = false
    private let queue = DispatchQueue(label: "local.inference.backend")
    
    public init() {}
    
    public func loadModelIfNeeded() async throws {
        if isLoaded { return }
        
        let repoRoot = FileManager.default.currentDirectoryPath
        let defaultModelPath = "\(repoRoot)/models/llada2-1-mini-4bit"
        let defaultTokenizerPath = "\(repoRoot)/models/llada2-1-mini"
        
        let modelURL = URL(fileURLWithPath: defaultModelPath)
        let tokenizerURL = URL(fileURLWithPath: defaultTokenizerPath)
        
        if FileManager.default.fileExists(atPath: modelURL.path) && FileManager.default.fileExists(atPath: tokenizerURL.path) {
            print("[LocalBackend] Loading real model from \(modelURL.path)...")
            let container = try DiffusionModel.load(from: modelURL)
            let tok = try await DiffusionTokenizer.from(modelFolder: tokenizerURL)
            self.engine = DiffusionEngine(model: container.model, speculationK: 4)
            self.tokenizer = tok
        } else {
            print("[LocalBackend] Real model files not found. Creating a dummy local configuration.")
            let config = DiffusionModelConfig(vocabSize: 32000, hiddenDim: 512, numLayers: 4)
            let container = DiffusionModel(config: config)
            // Create temporary dir for dummy tokenizer files
            let tempDir = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("dummy_tok")
            try? FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            // Save empty config/vocab if needed, or try loading sumi/dummy if they exist
            let tok = try await DiffusionTokenizer.from(pretrained: "gpt2") // fallback standard tokenizer
            self.engine = DiffusionEngine(model: container.model, speculationK: 4)
            self.tokenizer = tok
        }
        isLoaded = true
    }
    
    public func generate(
        prompt: String,
        config: SettingsSchema,
        onTrace: @escaping (DiffusionEngine.StepTrace) -> Void,
        onTokenStream: @escaping ([Int]) -> Void,
        onTextStream: @escaping (String) -> Void
    ) async throws -> DiffusionEngine.Output {
        try await loadModelIfNeeded()
        
        guard let engine = self.engine, let tokenizer = self.tokenizer else {
            throw NSError(domain: "LocalInferenceBackend", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not initialized"])
        }
        
        let promptIds = tokenizer.encode(text: prompt)
        let maskId = tokenizer.maskId
        let eosId = tokenizer.eosId
        let params = GenerationParams.from(schema: config, maskId: maskId, eosId: eosId)
        
        engine.onTrace = { trace in
            onTrace(trace)
        }
        
        return try await withCheckedThrowingContinuation { continuation in
            queue.async {
                do {
                    var currentTextLength = 0
                    let output = engine.generateCached(prompt: promptIds, params: params) { blockIds in
                        onTokenStream(blockIds)
                        
                        // Also compute decoded incremental text to stream back
                        let text = tokenizer.decode(tokens: blockIds)
                        if text.count > currentTextLength {
                            let delta = String(text.dropFirst(currentTextLength))
                            onTextStream(delta)
                            currentTextLength = text.count
                        }
                    }
                    continuation.resume(returning: output)
                } catch {
                    continuation.resume(throwing: error)
                }
            }
        }
    }
}

public class RemoteInferenceBackend: InferenceBackend {
    public let name: String
    public let isLocal = false
    
    private let host: String
    private let port: Int
    
    public init(host: String, port: Int) {
        self.host = host
        self.port = port
        self.name = "Remote: \(host):\(port)"
    }
    
    public func generate(
        prompt: String,
        config: SettingsSchema,
        onTrace: @escaping (DiffusionEngine.StepTrace) -> Void,
        onTokenStream: @escaping ([Int]) -> Void,
        onTextStream: @escaping (String) -> Void
    ) async throws -> DiffusionEngine.Output {
        let url = URL(string: "http://\(host):\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        
        let requestBody: [String: Any] = [
            "model": "llada2-1-mini",
            "messages": [
                ["role": "user", "content": prompt]
            ],
            "temperature": Double(config.temperature),
            "max_tokens": config.genLength,
            "stream": true,
            "neodiffusion_mode": config.editingThreshold > 0.1 ? "q" : "s",
            "cache_preset": config.cachePreset,
            "kv_quantization": config.kvQuantization
        ]
        
        request.httpBody = try JSONSerialization.data(withJSONObject: requestBody)
        
        let (bytes, response) = try await URLSession.shared.bytes(for: request)
        guard let httpResponse = response as? HTTPURLResponse, httpResponse.statusCode == 200 else {
            throw NSError(domain: "RemoteInferenceBackend", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid server response"])
        }
        
        var generatedText = ""
        
        for try await line in bytes.lines {
            guard line.hasPrefix("data: ") else { continue }
            let dataStr = line.dropFirst(6).trimmingCharacters(in: .whitespacesAndNewlines)
            if dataStr == "[DONE]" { break }
            
            guard let data = dataStr.data(using: .utf8) else { continue }
            if let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
               let choices = json["choices"] as? [[String: Any]],
               let first = choices.first,
               let delta = first["delta"] as? [String: Any],
               let content = delta["content"] as? String {
                generatedText += content
                onTextStream(content)
            }
        }
        
        // Return dummy engine output structure since we don't have local metrics
        return DiffusionEngine.Output(
            tokens: [],
            finalSequence: [],
            blockCommits: [],
            stepsPerBlock: [],
            syncPoints: 0,
            metrics: DiffusionEngine.Metrics(
                prefillSeconds: 0,
                denoiseSeconds: 0,
                commitSeconds: 0,
                postStepsPerBlock: [],
                forwardsEvaluated: 0,
                transfersPerStep: [],
                editsPerStep: [],
                meanTransferConfidencePerStep: [],
                eosBlockIndex: nil,
                logicalStepsTotal: 0,
                dualActiveSteps: 0,
                activationSteps: [],
                trailingStarvedStepsPerBlock: [],
                singleActiveDenoiseSeconds: 0,
                dualActiveDenoiseSeconds: 0,
                acceptedPerStep: [],
                tokensProcessedInForwards: 0,
                effectiveNBuf: config.nBuf,
                effectiveTauAdd: config.tauAdd,
                effectiveTauSemi: config.tauSemi,
                effectiveSpeculationK: config.speculationK,
                effectiveSpeculation: config.speculation,
                effectiveTauSpan: config.tauSpan,
                effectiveDynamicTauAlpha: config.dynamicTauAlpha,
                effectiveEosEarlyExit: config.eosEarlyExit,
                effectiveJotEnabled: config.jotEnabled,
                effectiveJotK: config.jotK,
                effectiveJotThreshold: config.jotThreshold,
                effectiveJotFaithful: config.jotFaithful,
                effectiveMoeCapacityRatio: config.moeCapacityRatio,
                effectiveIceEnabled: config.iceEnabled,
                effectiveIceTau: config.iceTau,
                effectiveIceNt: config.iceNt,
                effectiveIceThinkingLength: config.iceThinkingLength,
                effectiveCreditDecodingEnabled: config.creditDecodingEnabled,
                effectiveCreditAlpha: config.creditAlpha,
                effectiveCreditBeta: config.creditBeta,
                effectiveCreditGamma: config.creditGamma
            )
        )
    }
}
