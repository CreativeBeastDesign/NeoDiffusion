import Foundation
import Hummingbird
import HTTPTypes
import NIOCore
import MLX
import DiffusionModel
import DiffusionGeneration

// Conform classes to unchecked Sendable to allow safe capture in Hummingbird's @Sendable route closures
extension DiffusionTokenizer: @unchecked Sendable {}
extension DiffusionEngine: @unchecked Sendable {}

// MARK: - Host-aware speculationK default (final-plan F-l)

func sysctlStringValue(_ name: String) -> String {
    var size = 0
    guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "" }
    var buf = [CChar](repeating: 0, count: size)
    guard sysctlbyname(name, &buf, &size, nil, 0) == 0 else { return "" }
    return String(cString: buf)
}

/// The K=1-vs-K=4 loop-speculation decision, resolved from host analytics.
///
/// Measured (final-plan F-k/F-l, 2026-07-17, single-process interleaved, byte-identical output):
/// on the M2 Ultra (Mac14,14, 192 GB) K=1 beats the K=4 default by +6–17% because the host is
/// compute-bound and CPU-GPU syncs are cheap, so speculative run-ahead is pure overhead. K is
/// output-invariant, so this is a free speedup. On sync-bound hosts (M1 MacBook, 16 GB) K>1 can
/// still pay, so the default stays 4 there. The finding was only measured on the Ultra, so the
/// heuristic is deliberately conservative: physical memory ≥ 64 GB (Ultra/Max-class) ⇒ K=1.
/// Always overridable by `--speculation-k` / `NEODIFFUSION_SPECULATION_K`.
func hostAwareDefaultSpeculationK() -> (k: Int, model: String, memGB: Double, studioClass: Bool) {
    var mem: UInt64 = 0; var size = MemoryLayout<UInt64>.size
    sysctlbyname("hw.memsize", &mem, &size, nil, 0)
    let memGB = Double(mem) / 1_073_741_824.0
    let model = sysctlStringValue("hw.model")
    let studioClass = memGB >= 64.0
    return (studioClass ? 1 : 4, model, memGB, studioClass)
}

// MARK: - OpenAI Chat Completion API Models

struct ChatCompletionRequest: Codable {
    struct Message: Codable {
        let role: String
        let content: String
    }
    
    let model: String
    let messages: [Message]
    let temperature: Float?
    let maxTokens: Int?
    let stream: Bool?
    let neodiffusionMode: String?
    
    enum CodingKeys: String, CodingKey {
        case model
        case messages
        case temperature
        case maxTokens = "max_tokens"
        case stream
        case neodiffusionMode = "neodiffusion_mode"
    }
}

struct ChatCompletionResponse: Codable, ResponseEncodable {
    struct Choice: Codable {
        struct Message: Codable {
            let role: String
            let content: String
        }
        let index: Int
        let message: Message
        let logprobs: [String: Double]? = nil
        let finishReason: String
        
        enum CodingKeys: String, CodingKey {
            case index
            case message
            case logprobs
            case finishReason = "finish_reason"
        }
    }
    struct Usage: Codable {
        let promptTokens: Int
        let completionTokens: Int
        let totalTokens: Int
        
        enum CodingKeys: String, CodingKey {
            case promptTokens = "prompt_tokens"
            case completionTokens = "completion_tokens"
            case totalTokens = "total_tokens"
        }
    }
    
    let id: String
    var object: String = "chat.completion"
    let created: Int
    let model: String
    let choices: [Choice]
    let usage: Usage
}

struct ChatCompletionChunk: Codable {
    struct Choice: Codable {
        struct Delta: Codable {
            let role: String?
            let content: String?
        }
        let index: Int
        let delta: Delta
        let logprobs: [String: Double]? = nil
        let finishReason: String?
        
        enum CodingKeys: String, CodingKey {
            case index
            case delta
            case logprobs
            case finishReason = "finish_reason"
        }
    }
    
    let id: String
    var object: String = "chat.completion.chunk"
    let created: Int
    let model: String
    let choices: [Choice]
}

struct ModelListResponse: Codable, ResponseEncodable {
    struct ModelItem: Codable {
        let id: String
        let object: String = "model"
        let created: Int = 1719792000
        let ownedBy: String = "neodiffusion"
        
        enum CodingKeys: String, CodingKey {
            case id
            case object
            case created
            case ownedBy = "owned_by"
        }
    }
    var object: String = "list"
    let data: [ModelItem]
}

// MARK: - Concurrency Control

actor RequestQueue {
    private var isRunning = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func acquire() async {
        if !isRunning {
            isRunning = true
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    func release() {
        if !waiters.isEmpty {
            let next = waiters.removeFirst()
            next.resume()
        } else {
            isRunning = false
        }
    }
}

// MARK: - Server Main

@main
struct NeoDiffusionServer {
    static func main() async throws {
        // Line-buffered stdout to ensure logs flush immediately when redirected
        setvbuf(stdout, nil, _IOLBF, 0)
        
        // Local CLI argument helpers
        func argValue(_ name: String) -> String? {
            guard let i = CommandLine.arguments.firstIndex(of: name),
                  i + 1 < CommandLine.arguments.count
            else { return nil }
            return CommandLine.arguments[i + 1]
        }

        func hasFlag(_ name: String) -> Bool {
            CommandLine.arguments.contains(name)
        }
        
        let repoRoot = FileManager.default.currentDirectoryPath
        let modelPath = argValue("--model") 
            ?? ProcessInfo.processInfo.environment["NEODIFFUSION_MODEL_DIR"] 
            ?? "\(repoRoot)/models/llada2-1-mini-4bit"
        let tokenizerPath = argValue("--tokenizer") 
            ?? ProcessInfo.processInfo.environment["NEODIFFUSION_TOKENIZER_DIR"] 
            ?? "\(repoRoot)/models/llada2-1-mini"
        let host = argValue("--host") 
            ?? ProcessInfo.processInfo.environment["NEODIFFUSION_HOST"] 
            ?? "127.0.0.1"
        let portString = argValue("--port") 
            ?? ProcessInfo.processInfo.environment["NEODIFFUSION_PORT"] 
            ?? "8080"
        let port = Int(portString) ?? 8080
        // Host-aware default (final-plan F-l): K=1 on Studio/Ultra-class hardware (free +6–17%,
        // output-invariant), K=4 otherwise. Explicit --speculation-k / env always wins.
        let autoK = hostAwareDefaultSpeculationK()
        let specKOverride = argValue("--speculation-k")
            ?? ProcessInfo.processInfo.environment["NEODIFFUSION_SPECULATION_K"]
        let speculationK = specKOverride.flatMap(Int.init) ?? autoK.k
        print(String(format: "speculationK = %d (%@ — host %@, %.0f GB, %@-class)",
                     speculationK,
                     specKOverride != nil ? "forced via flag/env"
                        : "host-aware default, F-l",
                     autoK.model.isEmpty ? "unknown" : autoK.model, autoK.memGB,
                     autoK.studioClass ? "Studio/Ultra" : "standard"))
        let blockLength = Int(argValue("--block-length") ?? "32") ?? 32
        let noEarlyStop = hasFlag("--no-early-stop")
        
        let modelDir = URL(fileURLWithPath: modelPath)
        let tokenizerDir = URL(fileURLWithPath: tokenizerPath)
        
        print("=========================================")
        print("  NeoDiffusion OpenAI-compatible Server  ")
        print("=========================================")
        
        print("Loading model from \(modelDir.path)...")
        let container = try DiffusionModel.load(from: modelDir)
        print("Loading tokenizer from \(tokenizerDir.path)...")
        let tokenizer = try await DiffusionTokenizer.from(modelFolder: tokenizerDir)
        
        let engine = DiffusionEngine(model: container.model, speculationK: speculationK)
        // Hoisted out of the route closure: `DiffusionModel` is not Sendable, and the
        // handler only needs this immutable value.
        let modelVocabSize = container.config.vocabSize
        let requestQueue = RequestQueue()
        
        let router = Router()
        
        // Enable CORS for API requests
        let cors = CORSMiddleware<BasicRequestContext>(allowOrigin: .all)
        router.add(middleware: cors)
        
        router.get("health") { request, context in
            return "OK"
        }
        
        router.get("v1/models") { request, context -> ModelListResponse in
            return ModelListResponse(data: [ModelListResponse.ModelItem(id: "llada2-1-mini")])
        }
        
        router.post("v1/chat/completions") { request, context -> Response in
            print("[Server] Received completions request")
            let chatRequest: ChatCompletionRequest
            do {
                chatRequest = try await request.decode(as: ChatCompletionRequest.self, context: context)
            } catch {
                print("[Server] Failed to decode JSON request: \(error)")
                throw HTTPError(.badRequest, message: "Failed to decode JSON request: \(error.localizedDescription)")
            }
            
            let modeStr = chatRequest.neodiffusionMode ?? "q"
            guard let mode = GenerationParams.Mode(rawValue: modeStr) else {
                print("[Server] Invalid neodiffusion_mode: \(modeStr)")
                throw HTTPError(.badRequest, message: "Invalid neodiffusion_mode. Supported values: 'q' or 's'")
            }
            
            let genLength = chatRequest.maxTokens ?? 128
            let promptIds: [Int]
            do {
                let mappedMessages = chatRequest.messages.map { ["role": $0.role, "content": $0.content] }
                promptIds = try tokenizer.applyChatTemplate(messages: mappedMessages)
            } catch {
                print("[Server] Failed to apply chat template: \(error)")
                throw HTTPError(.badRequest, message: "Failed to apply chat template to messages: \(error.localizedDescription)")
            }
            
            // Safeguard against vocabulary mismatches (primarily for dev testing on the 1000-vocab dummy model)
            let vocabSize = modelVocabSize
            let safePromptIds = promptIds.map { $0 >= vocabSize ? ($0 % vocabSize) : $0 }
            let safeMaskId = tokenizer.maskId >= vocabSize ? (tokenizer.maskId % vocabSize) : tokenizer.maskId
            let safeEosId = tokenizer.eosId >= vocabSize ? (tokenizer.eosId % vocabSize) : tokenizer.eosId
            
            let params = GenerationParams.mode(
                mode,
                blockLength: blockLength,
                genLength: genLength,
                maskId: safeMaskId,
                eosId: safeEosId,
                eosEarlyStop: !noEarlyStop,
                temperature: chatRequest.temperature ?? 0.0
            )
            
            let isStream = chatRequest.stream ?? false
            print("[Server] Request details: model=\(chatRequest.model), stream=\(isStream), mode=\(modeStr), maxTokens=\(genLength), vocabSize=\(vocabSize)")
            
            if isStream {
                return try await handleStreamingRequest(
                    engine: engine,
                    tokenizer: tokenizer,
                    promptIds: safePromptIds,
                    params: params,
                    modelName: chatRequest.model,
                    requestQueue: requestQueue
                )
            } else {
                return try await handleBlockingRequest(
                    engine: engine,
                    tokenizer: tokenizer,
                    promptIds: safePromptIds,
                    params: params,
                    modelName: chatRequest.model,
                    requestQueue: requestQueue
                )
            }
        }
        
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )
        
        print("Server running on http://\(host):\(port)")
        print("Ready for requests.")
        print("=========================================")
        
        try await app.runService()
    }
}

// MARK: - Route Handlers

func handleBlockingRequest(
    engine: DiffusionEngine,
    tokenizer: DiffusionTokenizer,
    promptIds: [Int],
    params: GenerationParams,
    modelName: String,
    requestQueue: RequestQueue
) async throws -> Response {
    let requestId = "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
    let createdTime = Int(Date().timeIntervalSince1970)
    
    print("[Blocking] Attempting to acquire queue lock for \(requestId)")
    await requestQueue.acquire()
    print("[Blocking] Acquired queue lock for \(requestId)")
    defer {
        Task {
            print("[Blocking] Releasing queue lock for \(requestId)")
            await requestQueue.release()
        }
    }
    
    if Task.isCancelled {
        print("[Blocking] Request cancelled before execution")
        throw HTTPError(.requestTimeout, message: "Request cancelled while queued")
    }
    
    print("[Blocking] Starting generation for \(requestId)")
    let output = engine.generateCached(prompt: promptIds, params: params, streamBlock: nil)
    print("[Blocking] Generation complete for \(requestId), tokens: \(output.tokens.count)")
    let generatedText = tokenizer.decode(tokens: output.tokens)
    
    let responseObj = ChatCompletionResponse(
        id: requestId,
        created: createdTime,
        model: modelName,
        choices: [
            .init(
                index: 0,
                message: .init(role: "assistant", content: generatedText),
                finishReason: "stop"
            )
        ],
        usage: .init(
            promptTokens: promptIds.count,
            completionTokens: output.tokens.count,
            totalTokens: promptIds.count + output.tokens.count
        )
    )
    
    let responseData = try JSONEncoder().encode(responseObj)
    return Response(
        status: .ok,
        headers: [.contentType: "application/json; charset=utf-8"],
        body: .init(byteBuffer: ByteBuffer(bytes: responseData))
    )
}

func handleStreamingRequest(
    engine: DiffusionEngine,
    tokenizer: DiffusionTokenizer,
    promptIds: [Int],
    params: GenerationParams,
    modelName: String,
    requestQueue: RequestQueue
) async throws -> Response {
    let requestId = "chatcmpl-\(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(24))"
    let createdTime = Int(Date().timeIntervalSince1970)
    
    print("[Streaming] Preparing stream for \(requestId)")
    
    let tokenStream = AsyncStream<[Int]> { continuation in
        let task = Task {
            if Task.isCancelled {
                print("[Streaming Task] Task was cancelled before running")
                continuation.finish()
                return
            }
            
            print("[Streaming Task] Attempting to acquire queue lock for \(requestId)")
            await requestQueue.acquire()
            print("[Streaming Task] Acquired queue lock for \(requestId)")
            defer {
                Task {
                    print("[Streaming Task] Releasing queue lock for \(requestId)")
                    await requestQueue.release()
                }
            }
            
            if Task.isCancelled {
                print("[Streaming Task] Task was cancelled during queue wait")
                continuation.finish()
                return
            }
            
            print("[Streaming Task] Starting generation cached for \(requestId)")
            _ = engine.generateCached(prompt: promptIds, params: params) { blockIds in
                print("[Streaming Task] Yielding block with \(blockIds.count) tokens")
                continuation.yield(blockIds)
            }
            print("[Streaming Task] Generation finished for \(requestId)")
            continuation.finish()
        }
        
        continuation.onTermination = { @Sendable reason in
            print("[Streaming Task] Continuation terminated: \(reason)")
            task.cancel()
        }
    }
    
    var headers = HTTPFields()
    headers[.contentType] = "text/event-stream"
    headers[.cacheControl] = "no-cache"
    headers[.connection] = "keep-alive"
    
    return Response(
        status: .ok,
        headers: headers,
        body: ResponseBody { writer in
            print("[ResponseBody] Starting to write response for \(requestId)")
            // Send initial delta with role: assistant
            let initialChunk = ChatCompletionChunk(
                id: requestId,
                created: createdTime,
                model: modelName,
                choices: [
                    .init(
                        index: 0,
                        delta: .init(role: "assistant", content: ""),
                        finishReason: nil
                    )
                ]
            )
            try await writer.write(sseString(for: initialChunk))
            print("[ResponseBody] Wrote initial role chunk for \(requestId)")
            
            let promptLength = promptIds.count
            let B = params.blockLength
            let eosId = params.eosId
            var currentBlockIndex = promptLength / B
            var ended = false
            
            for try await blockIds in tokenStream {
                print("[ResponseBody] Received block from tokenStream for \(requestId)")
                let blockStart = currentBlockIndex * B
                let promptOffset = max(0, promptLength - blockStart)
                
                if promptOffset < blockIds.count {
                    var generatedInBlock = Array(blockIds[promptOffset...])
                    
                    if let firstEosIndex = generatedInBlock.firstIndex(of: eosId) {
                        generatedInBlock = Array(generatedInBlock.prefix(firstEosIndex + 1))
                        ended = true
                    }
                    
                    if !generatedInBlock.isEmpty {
                        let text = tokenizer.decode(tokens: generatedInBlock)
                        if !text.isEmpty {
                            let chunk = ChatCompletionChunk(
                                id: requestId,
                                created: createdTime,
                                model: modelName,
                                choices: [
                                    .init(
                                        index: 0,
                                        delta: .init(role: nil, content: text),
                                        finishReason: ended ? "stop" : nil
                                    )
                                ]
                            )
                            try await writer.write(sseString(for: chunk))
                            print("[ResponseBody] Sent chunk: \"\(text.replacingOccurrences(of: "\n", with: "\\n"))\"")
                        }
                    }
                }
                
                currentBlockIndex += 1
                if ended {
                    break
                }
            }
            
            if !ended {
                let terminalChunk = ChatCompletionChunk(
                    id: requestId,
                    created: createdTime,
                    model: modelName,
                    choices: [
                        .init(
                            index: 0,
                            delta: .init(role: nil, content: nil),
                            finishReason: "length"
                        )
                    ]
                )
                try await writer.write(sseString(for: terminalChunk))
                print("[ResponseBody] Sent terminal length chunk")
            }
            
            try await writer.write(ByteBuffer(string: "data: [DONE]\n\n"))
            print("[ResponseBody] Sent [DONE] marker for \(requestId)")
        }
    )
}

func sseString(for chunk: ChatCompletionChunk) -> ByteBuffer {
    let data = try! JSONEncoder().encode(chunk)
    let json = String(data: data, encoding: .utf8)!
    return ByteBuffer(string: "data: \(json)\n\n")
}
