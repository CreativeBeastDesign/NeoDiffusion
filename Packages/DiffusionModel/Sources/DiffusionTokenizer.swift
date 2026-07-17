import Foundation
import Hub
import Tokenizers

/// A helper class wrapping swift-transformers AutoTokenizer for the LLaDA2 MoE Model.
public final class DiffusionTokenizer: @unchecked Sendable {
    public let tokenizer: any Tokenizer
    
    /// Special token IDs mandated by the model architecture:
    /// - maskId: 156895
    /// - eosId: 156892
    /// - padId: 156892
    public let maskId: Int = 156895
    public let eosId: Int = 156892
    public let padId: Int = 156892
    
    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }
    
    /// Asynchronously load the tokenizer from a Hugging Face Hub repository ID.
    public static func from(pretrained repoId: String) async throws -> DiffusionTokenizer {
        let tokenizer = try await AutoTokenizer.from(pretrained: repoId)
        return DiffusionTokenizer(tokenizer: tokenizer)
    }
    
    /// Asynchronously load the tokenizer from a local model directory.
    public static func from(modelFolder url: URL) async throws -> DiffusionTokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: url)
        return DiffusionTokenizer(tokenizer: tokenizer)
    }
    
    /// Encode a string to an array of token IDs.
    public func encode(text: String) -> [Int] {
        return tokenizer.encode(text: text)
    }
    
    /// Decode an array of token IDs back into a string.
    public func decode(tokens: [Int]) -> String {
        return tokenizer.decode(tokens: tokens)
    }
    
    /// Format conversation messages into a string and encode them.
    public func applyChatTemplate(messages: [[String: String]]) throws -> [Int] {
        let chatMessages: [Message] = messages.map { dict in
            dict.mapValues { $0 as any Sendable }
        }
        return try tokenizer.applyChatTemplate(messages: chatMessages)
    }
    
    /// Format conversation messages into a string and encode them using the raw Hub Message typealias.
    public func applyChatTemplate(messages: [Message]) throws -> [Int] {
        return try tokenizer.applyChatTemplate(messages: messages)
    }
}
