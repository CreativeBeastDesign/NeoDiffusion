import XCTest
import Hub
import Tokenizers
@testable import DiffusionModel

/// sumi-M1' acceptance, tokenizer half (sumi-plan.md §3 S1.1): encode/decode parity with the
/// HF tokenizer on a 100-case text suite (multilingual, code, whitespace, special-token-looking
/// text). Fixtures come from `Tools/generate_sumi_tokenizer_fixtures.py`; Sumi is a base model
/// with no chat template, so the suite is text-only.
///
/// Regenerate fixtures with:
///   scratch/sumi-venv/bin/python Tools/generate_sumi_tokenizer_fixtures.py
final class SumiTokenizerTests: XCTestCase {

    struct Fixture: Decodable {
        struct Header: Decodable {
            let bosTokenId: Int
            let eosTokenId: Int
            let padTokenId: Int
            let vocabSize: Int

            enum CodingKeys: String, CodingKey {
                case bosTokenId = "bos_token_id"
                case eosTokenId = "eos_token_id"
                case padTokenId = "pad_token_id"
                case vocabSize = "vocab_size"
            }
        }
        struct Case: Decodable {
            let text: String
            let tokens: [Int]
            let tokensWithSpecials: [Int]
            let decoded: String

            enum CodingKeys: String, CodingKey {
                case text
                case tokens
                case tokensWithSpecials = "tokens_with_specials"
                case decoded
            }
        }
        let header: Header
        let cases: [Case]
    }

    static let fixtureURL = URL(
        fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/scratch/sumi_tokenizer_fixtures.json")
    static let tokenizerDir = URL(
        fileURLWithPath:
            "/Users/andrebarlocher/Documents/Swift/NeoDiffusion/Tools/reference/sumi")

    func testTokenizerParity() async throws {
        try XCTSkipUnless(
            FileManager.default.fileExists(atPath: Self.fixtureURL.path),
            "Sumi tokenizer fixtures missing — run: scratch/sumi-venv/bin/python Tools/generate_sumi_tokenizer_fixtures.py")

        let fixture = try JSONDecoder().decode(
            Fixture.self, from: Data(contentsOf: Self.fixtureURL))
        XCTAssertEqual(fixture.cases.count, 100, "Fixture should contain exactly 100 test cases.")

        let tokenizer = try await SumiTokenizer.from(modelFolder: Self.tokenizerDir)

        // Special ids: the Swift constants must agree with the Python tokenizer's report.
        XCTAssertEqual(tokenizer.bosId, fixture.header.bosTokenId)
        XCTAssertEqual(tokenizer.eosId, fixture.header.eosTokenId)
        XCTAssertEqual(tokenizer.padId, fixture.header.padTokenId)

        for (index, testCase) in fixture.cases.enumerated() {
            // Raw encode (no specials) — canvas/token-level parity form.
            let encoded = tokenizer.encode(text: testCase.text, addSpecialTokens: false)
            XCTAssertEqual(
                encoded, testCase.tokens,
                "Encode mismatch at index \(index): '\(testCase.text.prefix(60))'")

            // Default encode — the TemplateProcessing post-processor prepends BOS
            // (the prompt form the reference quickstart feeds `generate`).
            let encodedSpecial = tokenizer.encode(text: testCase.text)
            XCTAssertEqual(
                encodedSpecial, testCase.tokensWithSpecials,
                "Encode-with-specials mismatch at index \(index)")

            let decoded = tokenizer.decode(tokens: testCase.tokens)
            XCTAssertEqual(
                decoded, testCase.decoded,
                "Decode mismatch at index \(index)")
        }
    }
}
