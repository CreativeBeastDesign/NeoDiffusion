import Foundation
import Hub
import Tokenizers

/// Tokenizer wrapper for Sumi (`tohoku-nlp/sumi-7b`): a standard `tokenizers`-backend BPE
/// (cl100k-style byte-level, OLMo-family added tokens) shipped as a plain `tokenizer.json`.
///
/// Sumi has **no mask token** (uniform-state diffusion) and no chat template (base model);
/// the special ids that matter at inference are bos/eos (the mid-canvas `[EOS, BOS]` anchor)
/// and pad (used as trim filler).
public class SumiTokenizer {
    public let tokenizer: any Tokenizer

    /// Special token IDs from config.json / tokenizer.json (verified 2026-07-08):
    /// - bos `<|beginoftext|>` = 100256
    /// - eos `<|endoftext|>` = 100257
    /// - pad `<|pad|>` = 100277
    public let bosId: Int = 100256
    public let eosId: Int = 100257
    public let padId: Int = 100277

    public init(tokenizer: any Tokenizer) {
        self.tokenizer = tokenizer
    }

    /// Asynchronously load the tokenizer from a local model directory
    /// (needs `tokenizer.json` + `tokenizer_config.json`).
    public static func from(modelFolder url: URL) async throws -> SumiTokenizer {
        let tokenizer = try await AutoTokenizer.from(modelFolder: url)
        return SumiTokenizer(tokenizer: tokenizer)
    }

    /// Asynchronously load the tokenizer from a Hugging Face Hub repository ID.
    public static func from(pretrained repoId: String) async throws -> SumiTokenizer {
        let tokenizer = try await AutoTokenizer.from(pretrained: repoId)
        return SumiTokenizer(tokenizer: tokenizer)
    }

    /// Encode a string. With `addSpecialTokens` (the default, matching HF `tokenizer(text)`),
    /// the tokenizer's `TemplateProcessing` post-processor prepends `<|beginoftext|>` — the
    /// form prompts are fed to `generate` in the reference quickstart.
    public func encode(text: String, addSpecialTokens: Bool = true) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: addSpecialTokens)
    }

    public func decode(tokens: [Int]) -> String {
        tokenizer.decode(tokens: tokens)
    }
}
