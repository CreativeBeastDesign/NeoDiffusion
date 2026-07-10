#!/usr/bin/env python3
import os
import json
from transformers import AutoTokenizer

def generate_fixtures(out_path):
    print("Loading Hugging Face tokenizer for LLaDA2.1-mini...")
    tokenizer = AutoTokenizer.from_pretrained("inclusionAI/LLaDA2.1-mini", trust_remote_code=True)
    
    # Define 100 diverse test cases
    test_cases = []
    
    # 1. English sentences (1-20)
    english_sentences = [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the way we write software.",
        "Metal is a low-overhead hardware-accelerated graphics and compute API designed by Apple.",
        "Swift is a general-purpose, multi-paradigm, compiled programming language.",
        "NeoDiffusion is a high-performance Swift inference engine.",
        "Can you explain the difference between autoregressive and diffusion language models?",
        "Mixture of Experts uses a router to forward tokens to selected expert networks.",
        "ExactPrefixCache provides exact cache lookup under block-causal attention masks.",
        "Hummingbird is a lightweight Swift HTTP server framework.",
        "Apple Silicon provides unified memory architecture which is great for running LLMs.",
        "Quantization reduces the memory footprint of weights by packing them into 4-bit representations.",
        "A group size of 64 balances accuracy and performance during quantization.",
        "The attention mechanism dynamically scales key-value representations based on queries.",
        "RoPE (Rotary Position Embeddings) encodes absolute positions with rotation matrices.",
        "Mac Studio with M2 Ultra has 192GB of unified memory.",
        "Jinja templates format conversational inputs into raw strings for tokenization.",
        "Masked language models predict tokens that are masked out of the input sequence.",
        "Speedy mode uses a lower confidence threshold to accelerate generation steps.",
        "Denoising starts with fully masked sequences and iteratively recovers original text."
    ]
    for text in english_sentences:
        tokens = tokenizer.encode(text)
        test_cases.append({"type": "text", "text": text, "tokens": tokens})
        
    # 2. Multilingual sentences (21-45)
    multilingual_sentences = [
        # German
        "Guten Tag! Wie geht es Ihnen heute?",
        "Die Optimierung von Inferenzmotoren auf Apple Silicon erfordert genaue Metal-Kernel-Implementierung.",
        # French
        "Bonjour, comment ça va ?",
        "L'intelligence artificielle est un domaine de recherche passionnant.",
        # Spanish
        "Hola, ¿cómo estás?",
        "El motor de inferencia está optimizado para la arquitectura unificada de Apple.",
        # Italian
        "Buongiorno! Come stai?",
        "La quantizzazione riduce l'impronta di memoria del modello.",
        # Portuguese
        "Olá, tudo bem?",
        "A computação de alto desempenho usa aceleradores de hardware.",
        # Russian
        "Привет, мир! Как дела?",
        "Диффузионные языковые модели показывают отличные результаты.",
        # Chinese
        "你好，世界！今天天气怎么样？",
        "混合专家模型（MoE）通过路由选择不同的专家来进行计算。",
        # Japanese
        "こんにちは、世界！調子はどうですか？",
        "ロータリー位置エンコーディングは、トランスフォーマーモデルで広く使用されています。",
        # Korean
        "안녕하세요, 세계! 만나서 반갑습니다.",
        "애플 실리콘은 높은 메모리 대역폭을 제공합니다.",
        # Arabic
        "مرحبا يا عالم! كيف حالك اليوم؟",
        "نماذج انتشار اللغة هي بديل واعد للنماذج التقليدية.",
        # Hindi
        "नमस्ते दुनिया! आप कैसे हैं?",
        "मशीन लर्निंग मॉडल को चलाने के लिए बहुत अधिक मेमोरी की आवश्यकता होती है.",
        # Mixed
        "Hello 世界, how are you today? Wie geht's?",
        "User <|mask|> is typing in Chinese: 你好",
        "Tokenizer handles emoji 🚀 and special symbols like € or ¥."
    ]
    for text in multilingual_sentences:
        tokens = tokenizer.encode(text)
        test_cases.append({"type": "text", "text": text, "tokens": tokens})
        
    # 3. Code snippets (46-70)
    code_snippets = [
        "def hello_world():\n    print('Hello, world!')",
        "struct ModelConfig: Codable {\n    let vocabSize: Int\n    let hiddenSize: Int\n}",
        "class RMSNorm(nn.Module):\n    def __init__(self, dim, eps=1e-6):\n        super().__init__()\n        self.weight = nn.Parameter(torch.ones(dim))",
        "let array = MLXArray.zeros([2, 4], dtype: .float32)",
        "func quantize(model: Module) {\n    MLXNN.quantize(model: model, bits: 4)\n}",
        "import Foundation\nimport MLX\nimport MLXNN",
        "for i in range(10):\n    if i % 2 == 0:\n        print(f'{i} is even')",
        "protocol Quantizable {\n    func toQuantized(groupSize: Int, bits: Int) -> Module\n}",
        "const express = require('express');\nconst app = express();",
        "<html>\n<body>\n<h1>Hello</h1>\n</body>\n</html>",
        "body {\n    font-family: 'Inter', sans-serif;\n    color: #333;\n}",
        "int main() {\n    std::cout << \"Hello C++\" << std::endl;\n    return 0;\n}",
        "SELECT * FROM users WHERE active = 1 ORDER BY created_at DESC;",
        "JSON.stringify({ status: 'ok', count: 42 });",
        "assert x.shape == (B, L, H), f'Expected shape, got {x.shape}'",
        "try {\n    let data = try Data(contentsOf: url)\n} catch {\n    print(error)\n}",
        "// TODO: implement dynamic KV caching logic\n// WARN: do not cache active block",
        "#pragma once\n#include <metal_stdlib>",
        "kernel void my_kernel(device float* in [[buffer(0)]], device float* out [[buffer(1)]])",
        "let config = try JSONDecoder().decode(LLaDA2MoeConfig.self, from: data)",
        "class Router(nn.Module):\n    def forward(self, x):\n        logits = self.gate(x)\n        return torch.sigmoid(logits)",
        "git commit -m \"feat: implement milestone 1 weights loader\"",
        "curl -s https://huggingface.co/inclusionAI/LLaDA2.1-mini",
        "pip install numpy mlx huggingface_hub safetensors",
        "let matches = try regex.firstMatch(in: text)"
    ]
    for text in code_snippets:
        tokens = tokenizer.encode(text)
        test_cases.append({"type": "text", "text": text, "tokens": tokens})
        
    # 4. Chat Templates (71-100)
    chat_templates = [
        # User only
        [{"role": "user", "content": "Hello!"}],
        [{"role": "user", "content": "How do I implement Mixture of Experts?"}],
        # System + User
        [
            {"role": "system", "content": "You are a helpful coding assistant."},
            {"role": "user", "content": "Write a python function to compute RMSNorm."}
        ],
        [
            {"role": "system", "content": "Always respond in German."},
            {"role": "user", "content": "What is the capital of France?"}
        ],
        # User + Assistant
        [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2 + 2 = 4."}
        ],
        # System + User + Assistant + User
        [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "Solve for x: 2x = 10."},
            {"role": "assistant", "content": "Divide both sides by 2: x = 5."},
            {"role": "user", "content": "Thank you!"}
        ],
        # Templates with reasoning / think tag
        [
            {"role": "user", "content": "Why is the sky blue?"},
            {"role": "assistant", "content": "<think>\nRayleigh scattering.\n</think>\nThe sky is blue due to Rayleigh scattering."}
        ],
        # Multiple turns
        [
            {"role": "user", "content": "Tell me a joke."},
            {"role": "assistant", "content": "Why don't scientists trust atoms? Because they make up everything!"},
            {"role": "user", "content": "Explain it."}
        ]
    ]
    
    # Duplicate chat cases to reach 100 total cases
    # We have 20 english + 25 multilingual + 25 code + 8 chat = 78 cases.
    # Let's add more chat variations to reach exactly 100 cases.
    additional_chats = [
        # Simple greetings
        [{"role": "user", "content": f"Greeting test turn {i}."}] for i in range(10)
    ] + [
        # System + user variations
        [
            {"role": "system", "content": f"System persona role {i}."},
            {"role": "user", "content": "Hello."}
        ] for i in range(12)
    ]
    
    chat_templates.extend(additional_chats)
    
    for msgs in chat_templates:
        # Generate the formatted string using HF apply_chat_template
        formatted_str = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        # Tokenize the formatted string
        tokens = tokenizer.encode(formatted_str)
        # Format for saving: messages is a list of dicts, tokens is list of ints
        test_cases.append({
            "type": "chat",
            "messages": msgs,
            "formatted_text": formatted_str,
            "tokens": tokens
        })
        
    print(f"Total test cases generated: {len(test_cases)}")
    
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(test_cases, f, indent=2)
    print(f"Saved fixtures to {out_path}")

if __name__ == "__main__":
    generate_fixtures("scratch/tokenizer_test_fixtures.json")
