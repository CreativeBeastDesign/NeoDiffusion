import Foundation
import Combine

public enum Stability: String, Codable, Sendable {
    case stable
    case experimental
}

public struct FlagDescriptor: Identifiable, Sendable {
    public var id: String { key }
    public let key: String
    public let label: String
    public let category: String
    public let stability: Stability
    public let description: String
}

public struct SettingsSchema: Codable, Equatable, Sendable {
    // Core parameters (Stable)
    public var threshold: Float = 0.7
    public var editingThreshold: Float = 0.5
    public var maxPostSteps: Int = 16
    public var numToTransfer: Int = 1
    public var eosEarlyStop: Bool = false
    public var temperature: Float = 0.0
    public var blockLength: Int = 32
    public var genLength: Int = 128
    
    // MultiBD parameters
    public var nBuf: Int = 1
    public var tauAdd: Float = 2.0
    public var tauSemi: Float = 0.9
    
    // Speculation
    public var speculation: String = "none" // "none" or "s2d2"
    public var speculationK: Int = 4
    public var tauSpan: Int = 1
    
    // Dynamic Tau
    public var dynamicTauAlpha: Float = 0.0
    public var eosEarlyExit: Bool = false
    
    // JOT
    public var jotEnabled: Bool = false
    public var jotK: Int = 2
    public var jotThreshold: Float = 0.9
    public var jotFaithful: Bool = false
    public var subBlockCommit: Bool = false
    public var subBlockMinPrefix: Int = 8
    public var moeCapacityRatio: Float = 0.0
    
    // FlashBlock
    public var flashBlockEnabled: Bool = false
    public var flashBlockTau: Int = 4
    
    // Temporal Voting
    public var temporalVotingEnabled: Bool = false
    public var temporalVotingAlpha: Float = 0.0
    public var temporalVotingCutoff: Float = 0.9
    
    // ICE
    public var iceEnabled: Bool = false
    public var iceTau: Float = 0.9
    public var iceNt: Int = 3
    public var iceThinkingLength: Int = 96
    
    // Credit Decoding
    public var creditDecodingEnabled: Bool = false
    public var creditAlpha: Float = 0.5
    public var creditBeta: Float = 0.9
    public var creditGamma: Float = 0.5
    
    // Caching & KV Compression
    public var cachePreset: String = "exact-fast"
    public var skipRedundantCommitForward: Bool = true
    public var kvQuantization: String = "none" // "none", "fp8", "k8v4", "fp4"
    public var csa2Enabled: Bool = false
    
    public init() {}
    
    // Custom decoder initializer to allow robust loading when config.json has a subset of fields
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.threshold = (try? container.decode(Float.self, forKey: .threshold)) ?? 0.7
        self.editingThreshold = (try? container.decode(Float.self, forKey: .editingThreshold)) ?? 0.5
        self.maxPostSteps = (try? container.decode(Int.self, forKey: .maxPostSteps)) ?? 16
        self.numToTransfer = (try? container.decode(Int.self, forKey: .numToTransfer)) ?? 1
        self.eosEarlyStop = (try? container.decode(Bool.self, forKey: .eosEarlyStop)) ?? false
        self.temperature = (try? container.decode(Float.self, forKey: .temperature)) ?? 0.0
        self.blockLength = (try? container.decode(Int.self, forKey: .blockLength)) ?? 32
        self.genLength = (try? container.decode(Int.self, forKey: .genLength)) ?? 128
        
        self.nBuf = (try? container.decode(Int.self, forKey: .nBuf)) ?? 1
        self.tauAdd = (try? container.decode(Float.self, forKey: .tauAdd)) ?? 2.0
        self.tauSemi = (try? container.decode(Float.self, forKey: .tauSemi)) ?? 0.9
        
        self.speculation = (try? container.decode(String.self, forKey: .speculation)) ?? "none"
        self.speculationK = (try? container.decode(Int.self, forKey: .speculationK)) ?? 4
        self.tauSpan = (try? container.decode(Int.self, forKey: .tauSpan)) ?? 1
        
        self.dynamicTauAlpha = (try? container.decode(Float.self, forKey: .dynamicTauAlpha)) ?? 0.0
        self.eosEarlyExit = (try? container.decode(Bool.self, forKey: .eosEarlyExit)) ?? false
        
        self.jotEnabled = (try? container.decode(Bool.self, forKey: .jotEnabled)) ?? false
        self.jotK = (try? container.decode(Int.self, forKey: .jotK)) ?? 2
        self.jotThreshold = (try? container.decode(Float.self, forKey: .jotThreshold)) ?? 0.9
        self.jotFaithful = (try? container.decode(Bool.self, forKey: .jotFaithful)) ?? false
        self.subBlockCommit = (try? container.decode(Bool.self, forKey: .subBlockCommit)) ?? false
        self.subBlockMinPrefix = (try? container.decode(Int.self, forKey: .subBlockMinPrefix)) ?? 8
        self.moeCapacityRatio = (try? container.decode(Float.self, forKey: .moeCapacityRatio)) ?? 0.0
        
        self.flashBlockEnabled = (try? container.decode(Bool.self, forKey: .flashBlockEnabled)) ?? false
        self.flashBlockTau = (try? container.decode(Int.self, forKey: .flashBlockTau)) ?? 4
        
        self.temporalVotingEnabled = (try? container.decode(Bool.self, forKey: .temporalVotingEnabled)) ?? false
        self.temporalVotingAlpha = (try? container.decode(Float.self, forKey: .temporalVotingAlpha)) ?? 0.0
        self.temporalVotingCutoff = (try? container.decode(Float.self, forKey: .temporalVotingCutoff)) ?? 0.9
        
        self.iceEnabled = (try? container.decode(Bool.self, forKey: .iceEnabled)) ?? false
        self.iceTau = (try? container.decode(Float.self, forKey: .iceTau)) ?? 0.9
        self.iceNt = (try? container.decode(Int.self, forKey: .iceNt)) ?? 3
        self.iceThinkingLength = (try? container.decode(Int.self, forKey: .iceThinkingLength)) ?? 96
        
        self.creditDecodingEnabled = (try? container.decode(Bool.self, forKey: .creditDecodingEnabled)) ?? false
        self.creditAlpha = (try? container.decode(Float.self, forKey: .creditAlpha)) ?? 0.5
        self.creditBeta = (try? container.decode(Float.self, forKey: .creditBeta)) ?? 0.9
        self.creditGamma = (try? container.decode(Float.self, forKey: .creditGamma)) ?? 0.5
        
        self.cachePreset = (try? container.decode(String.self, forKey: .cachePreset)) ?? "exact-fast"
        self.skipRedundantCommitForward = (try? container.decode(Bool.self, forKey: .skipRedundantCommitForward)) ?? true
        self.kvQuantization = (try? container.decode(String.self, forKey: .kvQuantization)) ?? "none"
        self.csa2Enabled = (try? container.decode(Bool.self, forKey: .csa2Enabled)) ?? false
    }
    
    public static var allFlags: [FlagDescriptor] {
        return [
            FlagDescriptor(key: "threshold", label: "τ_mask Threshold", category: "Core", stability: .stable, description: "Confidence threshold for unmasking"),
            FlagDescriptor(key: "editingThreshold", label: "τ_edit Threshold", category: "Core", stability: .stable, description: "Confidence threshold for editing"),
            FlagDescriptor(key: "maxPostSteps", label: "Max Post Steps", category: "Core", stability: .stable, description: "Max refinement steps after block is unmasked"),
            FlagDescriptor(key: "numToTransfer", label: "Num To Transfer", category: "Core", stability: .stable, description: "Minimum tokens to unmask per step"),
            FlagDescriptor(key: "eosEarlyStop", label: "EOS Early Stop", category: "Core", stability: .stable, description: "Stop when EOS is unmasked"),
            FlagDescriptor(key: "temperature", label: "Temperature", category: "Core", stability: .stable, description: "Sampling temperature"),
            FlagDescriptor(key: "blockLength", label: "Block Length", category: "Core", stability: .stable, description: "Block granularity (e.g. 32)"),
            FlagDescriptor(key: "genLength", label: "Gen Length", category: "Core", stability: .stable, description: "Max tokens to generate"),
            
            FlagDescriptor(key: "cachePreset", label: "Cache Preset", category: "Caching", stability: .stable, description: "exact-fast (default, output-identical) or legacy. k8v4-clean / fp8-clean / kivi4-clean / dual-k8v4 change outputs, have no quality sheet, and need a server started with --allow-experimental-cache"),
            FlagDescriptor(key: "skipRedundantCommitForward", label: "Clean-Step Commit", category: "Caching", stability: .stable, description: "Commit verified clean-step KV directly without a redundant forward"),
            FlagDescriptor(key: "kvQuantization", label: "KV Cache Quantization", category: "Caching", stability: .experimental, description: "Prefix KV quantization (affine int, not float): none (FP16), fp8=int8 (-47%), k8v4 (-59%), fp4=int4 (-72%), kivi4 (-70%). Changes outputs; quality gate open; server needs --allow-experimental-cache"),
            FlagDescriptor(key: "csa2Enabled", label: "CSA2 Intra-Block Reuse", category: "Caching", stability: .experimental, description: "Depth-aware periodic intra-block KV reuse with Q-only projection"),
            
            FlagDescriptor(key: "nBuf", label: "Buffer Slots (nBuf)", category: "MultiBD", stability: .experimental, description: "Number of active blocks concurrently refined"),
            FlagDescriptor(key: "tauAdd", label: "τ_add Activation", category: "MultiBD", stability: .experimental, description: "Progress threshold to activate next block"),
            FlagDescriptor(key: "tauSemi", label: "τ_semi Completion", category: "MultiBD", stability: .experimental, description: "Progress threshold for fallback gating"),
            
            FlagDescriptor(key: "speculation", label: "Speculation Mode", category: "Speculation", stability: .experimental, description: "Speculative decoding kind (none or s2d2)"),
            FlagDescriptor(key: "speculationK", label: "Speculation K", category: "Speculation", stability: .experimental, description: "Batch size for speculative loop control"),
            FlagDescriptor(key: "tauSpan", label: "τ_span Speculation", category: "Speculation", stability: .experimental, description: "Min contiguous masked span to verify"),
            
            FlagDescriptor(key: "dynamicTauAlpha", label: "Dynamic τ_mask Alpha", category: "Dynamic Tau", stability: .experimental, description: "Adaptation strength of confidence threshold"),
            FlagDescriptor(key: "eosEarlyExit", label: "EOS Early Exit", category: "Dynamic Tau", stability: .experimental, description: "Fill active window with EOS after first EOS is found"),
            
            FlagDescriptor(key: "jotEnabled", label: "JOT Enabled", category: "JOT", stability: .experimental, description: "Enable Just-on-Time token freezing"),
            FlagDescriptor(key: "jotK", label: "JOT K-steps", category: "JOT", stability: .experimental, description: "Steps a token must remain stable to freeze"),
            FlagDescriptor(key: "jotThreshold", label: "JOT Threshold", category: "JOT", stability: .experimental, description: "Confidence threshold to freeze token"),
            FlagDescriptor(key: "jotFaithful", label: "JOT Faithful", category: "JOT", stability: .experimental, description: "Hold pre-freeze K/V representation constant"),
            FlagDescriptor(key: "subBlockCommit", label: "Sub-block Commit", category: "JOT", stability: .experimental, description: "Commit frozen prefixes early to cut attention query"),
            FlagDescriptor(key: "subBlockMinPrefix", label: "Sub-block Min Prefix", category: "JOT", stability: .experimental, description: "Min frozen prefix to trigger sub-block commit"),
            FlagDescriptor(key: "moeCapacityRatio", label: "MoE Capacity Ratio", category: "JOT", stability: .experimental, description: "Gather fraction of active tokens to skip expert GEMMs"),
            
            FlagDescriptor(key: "flashBlockEnabled", label: "FlashBlock Enabled", category: "FlashBlock", stability: .experimental, description: "Enable FlashBlock attention caching"),
            FlagDescriptor(key: "flashBlockTau", label: "FlashBlock Tau", category: "FlashBlock", stability: .experimental, description: "Dirty token threshold to refresh cache"),
            
            FlagDescriptor(key: "temporalVotingEnabled", label: "Temporal Voting Enabled", category: "Temporal Voting", stability: .experimental, description: "Enable Temporal Self-Consistency Voting"),
            FlagDescriptor(key: "temporalVotingAlpha", label: "Temporal Voting Alpha", category: "Temporal Voting", stability: .experimental, description: "Decay parameter alpha for exponential step-weighting"),
            FlagDescriptor(key: "temporalVotingCutoff", label: "Temporal Voting Cutoff", category: "Temporal Voting", stability: .experimental, description: "Cutoff ratio below which outputs are discarded"),
            
            FlagDescriptor(key: "iceEnabled", label: "ICE Enabled", category: "ICE", stability: .experimental, description: "Enable In-Place Chain-of-Thought prompting"),
            FlagDescriptor(key: "iceTau", label: "ICE Tau", category: "ICE", stability: .experimental, description: "Confidence threshold for ICE early exit"),
            FlagDescriptor(key: "iceNt", label: "ICE Reasoning Steps", category: "ICE", stability: .experimental, description: "Number of reasoning steps in templates"),
            FlagDescriptor(key: "iceThinkingLength", label: "ICE Thinking Length", category: "ICE", stability: .experimental, description: "Length of the reasoning/thinking section"),
            
            FlagDescriptor(key: "creditDecodingEnabled", label: "Credit Decoding Enabled", category: "Credit Decoding", stability: .experimental, description: "Enable Credit Decoding"),
            FlagDescriptor(key: "creditAlpha", label: "Credit Alpha", category: "Credit Decoding", stability: .experimental, description: "Boost scale alpha for credit decoding"),
            FlagDescriptor(key: "creditBeta", label: "Credit Beta", category: "Credit Decoding", stability: .experimental, description: "Decay discount factor beta for credit decoding"),
            FlagDescriptor(key: "creditGamma", label: "Credit Gamma", category: "Credit Decoding", stability: .experimental, description: "Exponent gamma for concave transform")
        ]
    }
}

public class Settings: ObservableObject {
    @Published public var config: SettingsSchema
    
    public static let shared = Settings()
    
    private init() {
        self.config = SettingsSchema()
        loadConfigJSON()
    }
    
    private var appSupportDirectory: URL {
        let paths = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)
        let appSupportDir = paths[0].appendingPathComponent("NeoDiffusion", isDirectory: true)
        try? FileManager.default.createDirectory(at: appSupportDir, withIntermediateDirectories: true)
        return appSupportDir
    }
    
    public var configFileURL: URL {
        return appSupportDirectory.appendingPathComponent("config.json")
    }
    
    public func loadConfigJSON() {
        let fileURL = configFileURL
        guard FileManager.default.fileExists(atPath: fileURL.path) else {
            saveConfigJSON()
            return
        }
        
        do {
            let data = try Data(contentsOf: fileURL)
            let decoder = JSONDecoder()
            let decoded = try decoder.decode(SettingsSchema.self, from: data)
            self.config = decoded
            print("Loaded config.json successfully from \(fileURL.path)")
        } catch {
            print("Failed to load config.json: \(error). Reverting to compile-time defaults.")
        }
    }
    
    public func saveConfigJSON() {
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = .prettyPrinted
            let data = try encoder.encode(self.config)
            try data.write(to: configFileURL)
            print("Saved config.json to \(configFileURL.path)")
        } catch {
            print("Failed to save config.json: \(error)")
        }
    }
    
    public func resetToDefaults() {
        self.config = SettingsSchema()
        saveConfigJSON()
    }
}
