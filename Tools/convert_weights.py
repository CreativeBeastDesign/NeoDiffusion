#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import mlx.core as mx
from safetensors.numpy import load_file, save_file

def should_quantize(key):
    """
    Decides whether a weight tensor should be quantized to 4-bit.
    Per precision rules:
    - Quantize: QKV, attention dense, dense FFN (layer 0), routed expert weights.
    - Keep: embeddings, norms, router gate, shared experts, lm_head.
    """
    if not key.endswith(".weight"):
        return False
    if "word_embeddings" in key:
        return False
    if "norm" in key:
        return False
    if "lm_head" in key:
        return False
    if "shared_experts" in key:
        return False
    if key.endswith(".mlp.gate.weight"): # MoE router weights (keep as float32)
        return False
    return True

def convert_and_quantize(src_dir, dest_dir, bits=4, group_size=64):
    os.makedirs(dest_dir, exist_ok=True)
    
    # 1. Read config.json
    config_path = os.path.join(src_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {src_dir}")
        
    with open(config_path, "r") as f:
        config = json.load(f)
        
    # Append quantization parameters to configuration
    config["quantization"] = {
        "bits": bits,
        "group_size": group_size,
        "mode": "affine"
    }
    
    # Write updated config to destination
    with open(os.path.join(dest_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
        
    # 2. Collect safetensors weight files from source directory
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_files = sorted(list(set(index_data["weight_map"].values())))
    else:
        weight_files = [f for f in os.listdir(src_dir) if f.endswith(".safetensors")]
        
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {src_dir}")
        
    print(f"Processing weight files: {weight_files}")
    
    converted_tensors = {}
    for wf in weight_files:
        src_file = os.path.join(src_dir, wf)
        print(f"Reading weights from {src_file}...")
        tensors = load_file(src_file)
        
        for k, v in tensors.items():
            # Convert NumPy array to MLX array
            x = mx.array(v)
            
            if should_quantize(k):
                # Quantize using MLX
                # mx.quantize returns: (quantized_weight, scales, biases)
                w_q, scales, biases = mx.quantize(x, group_size=group_size, bits=bits, mode="affine")
                
                # Store packed weight
                converted_tensors[k] = np.array(w_q)
                
                # Append scales and biases
                base_name = k[:-7] # Remove '.weight'
                converted_tensors[f"{base_name}.scales"] = np.array(scales.astype(mx.float16))
                converted_tensors[f"{base_name}.biases"] = np.array(biases.astype(mx.float16))
                
                print(f"  Quantized: {k} (shape: {w_q.shape}, scales: {scales.shape}, biases: {biases.shape})")
            else:
                # Keep unquantized. Router weights should be float32, others float16.
                if "mlp.gate" in k:
                    x = x.astype(mx.float32)
                else:
                    x = x.astype(mx.float16)
                converted_tensors[k] = np.array(x)
                print(f"  Kept unquantized: {k} (shape: {x.shape}, dtype: {x.dtype})")
                
    # Save the output to a single consolidated safetensors file
    out_file = os.path.join(dest_dir, "model.safetensors")
    print(f"Saving converted weights to {out_file}...")
    save_file(converted_tensors, out_file)
    print("Weight conversion complete successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and quantize Hugging Face LLaDA2 MoE weights to MLX 4-bit format.")
    parser.add_argument("--src-dir", required=True, help="Path to HF model source directory containing config.json and safetensors files")
    parser.add_argument("--dest-dir", required=True, help="Destination directory to save quantized weights and updated config")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    
    args = parser.parse_args()
    
    try:
        convert_and_quantize(args.src_dir, args.dest_dir, bits=args.bits, group_size=args.group_size)
    except Exception as e:
        print(f"Error during weight conversion: {e}")
        exit(1)
