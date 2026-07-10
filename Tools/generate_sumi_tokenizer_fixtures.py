#!/usr/bin/env python3
"""Sumi tokenizer fixture generator (sumi-plan.md §3 S1.1).

Encodes a 100-case text suite (multilingual, code, whitespace edge cases) with the HF
tokenizer from Tools/reference/sumi/ and writes scratch/sumi_tokenizer_fixtures.json for
SumiTokenizerTests. Sumi is a base model with no chat template, so the suite is text-only
(unlike the LLaDA fixtures).

Run with the Sumi venv (transformers >= 5.8):
  scratch/sumi-venv/bin/python Tools/generate_sumi_tokenizer_fixtures.py
"""

import json
import os

from transformers import PreTrainedTokenizerFast

REFERENCE_DIR = os.path.join(os.path.dirname(__file__), "reference", "sumi")
OUT_PATH = os.path.join(os.path.dirname(__file__), "..", "scratch", "sumi_tokenizer_fixtures.json")

TEXTS = [
    # Plain English
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, there was a diffusion language model.",
    "Numbers: 1, 22, 333, 4444, 55555, 1234567890.",
    "Punctuation!? ... ;: -- (parentheses) [brackets] {braces} <angle>",
    "Contractions: don't, can't, won't, it's, we're, they've, I'll, he'd.",
    "MixedCase CamelCase snake_case kebab-case SCREAMING_SNAKE.",
    "A sentence ending without period",
    "  Leading spaces",
    "Trailing spaces  ",
    "Multiple   internal    spaces",
    "word",
    "a",
    " ",
    "",
    # Newlines / whitespace structure
    "Line one\nLine two\nLine three",
    "Windows line endings\r\nSecond line",
    "Tabs\tbetween\twords",
    "\n\n\n",
    "Paragraph one.\n\nParagraph two.",
    "Ends with newline\n",
    # Code
    "def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)",
    "for (int i = 0; i < 10; i++) { printf(\"%d\\n\", i); }",
    "let x: [Int] = (0..<10).map { $0 * $0 }",
    "SELECT id, name FROM users WHERE age >= 21 ORDER BY name;",
    "import numpy as np\narr = np.zeros((3, 4), dtype=np.float32)",
    "<html><body><p class=\"x\">Hi &amp; bye</p></body></html>",
    "{\"key\": \"value\", \"list\": [1, 2, 3], \"nested\": {\"a\": null}}",
    "#!/bin/bash\necho \"$HOME\" | grep -o '/.*'",
    "x = lambda a, b=2: a ** b  # comment",
    "match value {\n    Some(v) => v,\n    None => 0,\n}",
    "public static void main(String[] args) throws IOException {}",
    # Math / symbols
    "E = mc^2 and ∑_{i=1}^{n} i = n(n+1)/2",
    "α β γ δ ε ζ η θ — ∀x ∈ ℝ: x² ≥ 0",
    "Price: $19.99, €15.50, ¥2000, £12.00, ₹500",
    "5 ± 0.3 × 10⁻⁶ ÷ 2 ≈ 2.5e-6",
    # Multilingual
    "Grüße aus Zürich — schöne Föhnlage heute.",
    "Le français est une langue romane parlée dans le monde entier.",
    "El niño comió mañana y ñoquis con jalapeños.",
    "Português: ação, coração, não, pão.",
    "Italiano: perché, città, più, così.",
    "こんにちは、世界！日本語のテキストです。",
    "漢字とひらがなとカタカナが混ざった文章。",
    "中文测试：你好，世界！这是一个分词测试。",
    "简体中文和繁體中文的混合文本。",
    "안녕하세요 세계! 한국어 토큰화 테스트입니다.",
    "Привет, мир! Это тест токенизации на русском.",
    "Ελληνικά: γεια σου κόσμε, καλημέρα!",
    "مرحبا بالعالم! هذا اختبار للتقطيع.",
    "שלום עולם! זהו מבחן טוקניזציה.",
    "हिन्दी में नमस्ते दुनिया!",
    "ไทย: สวัสดีชาวโลก ทดสอบการตัดคำ",
    "Tiếng Việt: Xin chào thế giới, đây là bài kiểm tra.",
    "Türkçe: Merhaba dünya, güzel bir gün!",
    "Polski: Zażółć gęślą jaźń.",
    "Čeština: Příliš žluťoučký kůň úpěl ďábelské ódy.",
    # Emoji / unusual unicode
    "Emoji: 😀 🚀 🧪 👨‍👩‍👧‍👦 🇯🇵 ❤️",
    "Zero-width​joiner and non-breaking space.",
    "Combining: é (é) vs é (precomposed).",
    "Box drawing: ┌─┬─┐ │ ├─┼─┤ └─┴─┘",
    "Fullwidth: ＨＥＬＬＯ　ＷＯＲＬＤ １２３",
    # Special-token-looking text (must NOT map to special ids when encoding plain text)
    "<|endoftext|>",
    "<|beginoftext|>",
    "Text with <|endoftext|> in the middle.",
    "<|im_start|>user\nHello<|im_end|>",
    "|||PHONE_NUMBER|||",
    # Long-ish / repetitive
    "word " * 50,
    "ab" * 100,
    "The " * 30 + "end.",
    "1234567890" * 12,
    "=" * 80,
    "-" * 79 + "x",
    # URLs / emails / paths
    "Visit https://example.com/path?query=value&x=1#frag for details.",
    "Email me at andre.baerlocher@example.com or user+tag@sub.domain.org.",
    "Path: /usr/local/bin/python3 and C:\\Program Files\\App\\bin.exe",
    "git clone git@github.com:tohoku-nlp/sumi.git && cd sumi",
    # Quotes and apostrophes
    "“Smart quotes” and ‘single ones’ versus \"plain\" and 'simple'.",
    "It’s the model’s best guess — or is it?",
    # Mixed content
    "Der Preis beträgt 42,50 € (inkl. MwSt.) — siehe §3 Abs. 2.",
    "The 東京 subway carries ~8M passengers/day (2019年データ).",
    "f(x) = ln(x) has derivative 1/x for x > 0. QED.",
    "RFC 2119: MUST, SHOULD, MAY — normative keywords.",
    "v2.31.4-beta+build.2026.07.08",
    "CH-8001 Zürich, Bahnhofstrasse 1",
    "a.b.c.d.e.f.g.h.i.j",
    "camelCaseIdentifier.methodCall(argumentOne, argument_two)",
    "0x1A2B3C4D 0b101010 0o777 1_000_000",
    "TODO(andre): fix the off-by-one softmax // FIXME later",
    "Sumi (墨) means ink — as in ink-wash painting (水墨画).",
    "The GIDD loss combines KL and Itakura–Saito divergences.",
    "Attention is all you need... plus one (the sink).",
    "prompt | canvas | anchor: EOS,BOS at prompt_len + budget",
    "α_t = sigmoid(log SNR_t); β_t = 1 − α_t; u_t = β_t / V",
    "Step 128/128: log_snr −9 → +9, linear schedule.",
    "🤖 Robots write código in 日本語 sometimes — прекрасно!",
    "Novel word: supercalifragilisticexpialidocious antidisestablishmentarianism.",
    "Base64: aGVsbG8gd29ybGQ= and hex deadbeefcafe0123.",
    "Nested \"quotes 'inside' quotes\" and \\escaped\\ backslashes \\n literal.",
    "Roman numerals: MMXXVI, XLII; fractions: ½ ⅓ ¾; degrees: 25°C, 77°F.",
    "End of fixture suite. ✓",
]


def main():
    # Load from tokenizer.json directly: AutoTokenizer.from_pretrained on the reference dir
    # trips the trust_remote_code prompt because config.json carries an auto_map.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(REFERENCE_DIR, "tokenizer.json"),
        bos_token="<|beginoftext|>",
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
    )
    fixtures = []
    for text in TEXTS:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # With specials: the tokenizer.json TemplateProcessing post-processor prepends BOS —
        # the form prompts take in the reference quickstart (`tokenizer(prompt)`).
        tokens_special = tokenizer.encode(text, add_special_tokens=True)
        decoded = tokenizer.decode(tokens, skip_special_tokens=False)
        fixtures.append({
            "type": "text",
            "text": text,
            "tokens": tokens,
            "tokens_with_specials": tokens_special,
            "decoded": decoded,
        })

    assert len(fixtures) == 100, f"expected 100 cases, got {len(fixtures)}"

    # Special-token id sanity, recorded in the fixture header consumed by the Swift test.
    header = {
        "bos_token_id": tokenizer.convert_tokens_to_ids("<|beginoftext|>"),
        "eos_token_id": tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        "pad_token_id": tokenizer.convert_tokens_to_ids("<|pad|>"),
        "vocab_size": len(tokenizer),
    }

    out = {"header": header, "cases": fixtures}
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"Wrote {len(fixtures)} cases to {os.path.abspath(OUT_PATH)}")
    print(f"Special ids: {header}")


if __name__ == "__main__":
    main()
