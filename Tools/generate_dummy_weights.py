#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
from safetensors.numpy import save_file

def generate_dummy_model(out_dir, num_layers=2, hidden_size=128, num_experts=4):
    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Create a minimal configuration mirroring configuration_llada2_moe.py
    config = {
        "architectures": ["LLaDA2MoeModelLM"],
        "vocab_size": 1000,
        "hidden_size": hidden_size,
        "intermediate_size": 256,
        "num_hidden_layers": num_layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": hidden_size // 4,
        "rms_norm_eps": 1e-06,
        "pad_token_id": 999,
        "num_experts": num_experts,
        "num_experts_per_tok": 2,
        "num_shared_experts": 1,
        "n_group": 2,
        "topk_group": 1,
        "moe_intermediate_size": 64,
        "first_k_dense_replace": 1,
        "use_bias": False,
        "use_qk_norm": True,
        "rope_theta": 600000,
        "partial_rotary_factor": 0.5,
        "model_type": "llada2_moe"
    }
    
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    tensors = {}
    
    # Embedding: shape [vocab_size, hidden_size]
    tensors["model.word_embeddings.weight"] = np.random.randn(1000, hidden_size).astype(np.float32)
    # Output norm: shape [hidden_size]
    tensors["model.norm.weight"] = np.ones(hidden_size).astype(np.float32)
    # Output head: shape [vocab_size, hidden_size]
    tensors["lm_head.weight"] = np.random.randn(1000, hidden_size).astype(np.float32)
    
    # Layers
    for i in range(num_layers):
        # Norms: shape [hidden_size]
        tensors[f"model.layers.{i}.input_layernorm.weight"] = np.ones(hidden_size).astype(np.float32)
        tensors[f"model.layers.{i}.post_attention_layernorm.weight"] = np.ones(hidden_size).astype(np.float32)
        
        # Attention
        # query_key_value: Q=4 heads * 32 dim, K=1 head * 32 dim, V=1 head * 32 dim -> QKV total dims = 6 * 32 = 192
        qkv_dim = (4 + 2 * 1) * (hidden_size // 4)
        tensors[f"model.layers.{i}.attention.query_key_value.weight"] = np.random.randn(qkv_dim, hidden_size).astype(np.float32)
        # qk-norm is per-head-dim (reference: LLaDA2MoeRMSNorm(head_dim)), NOT per num_heads*head_dim
        tensors[f"model.layers.{i}.attention.query_layernorm.weight"] = np.ones(hidden_size // 4).astype(np.float32)
        tensors[f"model.layers.{i}.attention.key_layernorm.weight"] = np.ones(hidden_size // 4).astype(np.float32)
        tensors[f"model.layers.{i}.attention.dense.weight"] = np.random.randn(hidden_size, 4 * (hidden_size // 4)).astype(np.float32)
        
        # MLP / FFN
        if i == 0:
            # Layer 0 is Dense FFN (intermediate size 5120, let's use 256 for toy configuration)
            tensors[f"model.layers.0.mlp.gate_proj.weight"] = np.random.randn(256, hidden_size).astype(np.float32)
            tensors[f"model.layers.0.mlp.up_proj.weight"] = np.random.randn(256, hidden_size).astype(np.float32)
            tensors[f"model.layers.0.mlp.down_proj.weight"] = np.random.randn(hidden_size, 256).astype(np.float32)
        else:
            # Layers 1..N are MoE Blocks
            # Router: gate weight [num_experts, hidden_size], expert bias [num_experts]
            tensors[f"model.layers.{i}.mlp.gate.weight"] = np.random.randn(num_experts, hidden_size).astype(np.float32)
            tensors[f"model.layers.{i}.mlp.gate.expert_bias"] = np.zeros(num_experts).astype(np.float32)
            
            # Shared expert: intermediate size 64
            tensors[f"model.layers.{i}.mlp.shared_experts.gate_proj.weight"] = np.random.randn(64, hidden_size).astype(np.float32)
            tensors[f"model.layers.{i}.mlp.shared_experts.up_proj.weight"] = np.random.randn(64, hidden_size).astype(np.float32)
            tensors[f"model.layers.{i}.mlp.shared_experts.down_proj.weight"] = np.random.randn(hidden_size, 64).astype(np.float32)
            
            # Experts: num_experts, intermediate size 64
            for e in range(num_experts):
                tensors[f"model.layers.{i}.mlp.experts.{e}.gate_proj.weight"] = np.random.randn(64, hidden_size).astype(np.float32)
                tensors[f"model.layers.{i}.mlp.experts.{e}.up_proj.weight"] = np.random.randn(64, hidden_size).astype(np.float32)
                tensors[f"model.layers.{i}.mlp.experts.{e}.down_proj.weight"] = np.random.randn(hidden_size, 64).astype(np.float32)
                
    # Save safetensors
    save_file(tensors, os.path.join(out_dir, "model.safetensors"))
    print(f"Generated dummy model at {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dummy model weights for LLaDA2 MoE testing.")
    parser.add_argument("--out-dir", required=True, help="Output directory to save dummy weights")
    args = parser.parse_args()
    
    generate_dummy_model(args.out_dir)
