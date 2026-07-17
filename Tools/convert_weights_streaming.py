#!/usr/bin/env python3
import os
import json
import struct
import gc
import argparse
import numpy as np
import mlx.core as mx
from safetensors import safe_open # kept for reference, not used for loading

def should_quantize(key):
    """
    Decides whether a weight tensor should be quantized to 4-bit.
    """
    if not key.endswith(".weight"):
        return False
    if "word_embeddings" in key or "embed_tokens" in key or "wte" in key:
        return False
    if "norm" in key:
        return False
    if "lm_head" in key:
        return False
    if "shared_experts" in key:
        return False
    if key.endswith(".mlp.gate.weight") or key.endswith(".gate.weight"):
        return False
    return True

def read_safetensors_header(file_path):
    """Reads only the JSON header of a safetensors file without loading any weights."""
    with open(file_path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)
        return json.loads(header_bytes.decode("utf-8")), header_size

def is_routed_expert(key):
    """Routed-expert tensors (M8 sweep axis): `...mlp.experts.<e>...`, never shared."""
    return ".experts." in key and "shared_experts" not in key


def convert_and_quantize_streaming(src_dir, dest_dir, bits=4, group_size=64,
                                   expert_bits=None, expert_group_size=None):
    """expert_bits/expert_group_size override the quantization of ROUTED EXPERT tensors
    only (M8 E6/E7 sweep axes: g32 / 6-bit experts, rest stays at bits/group_size)."""
    expert_bits = expert_bits or bits
    expert_group_size = expert_group_size or group_size

    def quant_params(key):
        if is_routed_expert(key):
            return expert_bits, expert_group_size
        return bits, group_size

    os.makedirs(dest_dir, exist_ok=True)

    # 1. Parse config.json
    config_path = os.path.join(src_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    config["quantization"] = {"bits": bits, "group_size": group_size, "mode": "affine",
                              "expert_bits": expert_bits,
                              "expert_group_size": expert_group_size}
    with open(os.path.join(dest_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # 2. Build map of tensor_name -> shard_file
    index_path = os.path.join(src_dir, "model.safetensors.index.json")
    weight_map = {}
    if os.path.exists(index_path):
        with open(index_path, "r") as f:
            index_data = json.load(f)
        weight_map = index_data["weight_map"]
    else:
        safetensors_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".safetensors")])
        for sf in safetensors_files:
            header, _ = read_safetensors_header(os.path.join(src_dir, sf))
            for k in header.keys():
                if k != "__metadata__":
                    weight_map[k] = sf

    # 3. Read shapes & dtypes of all tensors (extremely lightweight)
    original_metadata = {}
    shard_headers = {}
    for k, sf in weight_map.items():
        if sf not in shard_headers:
            shard_headers[sf], _ = read_safetensors_header(os.path.join(src_dir, sf))
        original_metadata[k] = shard_headers[sf][k]

    # 4. Pre-calculate the output offsets and shapes
    dtype_bytes = {"F32": 4, "F16": 2, "BF16": 2, "I32": 4, "I16": 2, "I8": 1, "U8": 1}
    output_tensors = []

    for k, info in original_metadata.items():
        shape = info["shape"]

        if not should_quantize(k):
            out_dtype = "F32" if (k.endswith(".gate.weight") or "gate.expert_bias" in k) else "F16"
            size_bytes = int(np.prod(shape)) * dtype_bytes[out_dtype]
            output_tensors.append({
                "name": k, "shape": shape, "dtype": out_dtype, "size_bytes": size_bytes,
                "source_name": k, "type": "cast"
            })
        else:
            M, N = shape
            base_name = k[:-7] # Remove '.weight'
            bits_k, group_k = quant_params(k)

            # Packed weights: N*bits/8 bytes per row (U8 byte view of MLX's U32 packing).
            w_q_size = M * (N * bits_k // 8)
            scales_size = M * (N // group_k) * 2
            biases_size = M * (N // group_k) * 2

            output_tensors.append({
                "name": k, "shape": [M, N * bits_k // 8], "dtype": "U8", "size_bytes": w_q_size,
                "source_name": k, "type": "q_weight"
            })
            output_tensors.append({
                "name": f"{base_name}.scales", "shape": [M, N // group_k], "dtype": "F16", "size_bytes": scales_size,
                "source_name": k, "type": "q_scales"
            })
            output_tensors.append({
                "name": f"{base_name}.biases", "shape": [M, N // group_k], "dtype": "F16", "size_bytes": biases_size,
                "source_name": k, "type": "q_biases"
            })

    # Calculate sequential byte offsets
    header_dict = {}
    current_offset = 0
    for out in output_tensors:
        start = current_offset
        end = current_offset + out["size_bytes"]
        header_dict[out["name"]] = {
            "dtype": out["dtype"],
            "shape": out["shape"],
            "data_offsets": [start, end]
        }
        current_offset = end

    # 5. Write Header
    header_json = json.dumps(header_dict, separators=(",", ":"))
    header_bytes = header_json.encode("utf-8")

    # Pad header to 64-byte boundary to guarantee tensor memory alignment
    total_header_len = 8 + len(header_bytes)
    padding = (64 - total_header_len % 64) % 64
    header_bytes += b" " * padding
    actual_header_size = len(header_bytes)

    out_file = os.path.join(dest_dir, "model.safetensors")
    print(f"Writing streaming weights to {out_file}...")

    # Open input files lazily using MLX native loading (supports BF16)
    # Keep at most one shard loaded in memory to avoid paging/GPU Timeout
    current_shard_name = None
    current_shard_tensors = None

    def get_input_tensor(k):
        nonlocal current_shard_name, current_shard_tensors
        sf = weight_map[k]
        if sf != current_shard_name:
            if current_shard_tensors is not None:
                del current_shard_tensors
                gc.collect()
                mx.metal.clear_cache()
            current_shard_tensors = mx.load(os.path.join(src_dir, sf))
            current_shard_name = sf
        return current_shard_tensors[k]

    with open(out_file, "wb") as out_f:
        # Write uint64 header size
        out_f.write(struct.pack("<Q", actual_header_size))
        # Write JSON header
        out_f.write(header_bytes)

        # 6. Stream and Quantize Tensors
        # Process source-by-source to avoid loading the same input weight multiple times
        processed_sources = set()

        for out in output_tensors:
            src_name = out["source_name"]
            if src_name in processed_sources:
                continue

            # Load single tensor
            x = get_input_tensor(src_name)

            if should_quantize(src_name):
                # Run GPU-accelerated quantization (per-tensor params: expert override)
                bits_k, group_k = quant_params(src_name)
                w_q, scales, biases = mx.quantize(x, group_size=group_k, bits=bits_k)

                # CRITICAL: mx.eval forces compilation/execution immediately
                # so memory is not held by a growing lazy execution graph
                mx.eval(w_q, scales, biases)

                # Write to disk
                out_f.write(np.array(w_q).tobytes())
                out_f.write(np.array(scales.astype(mx.float16)).tobytes())
                out_f.write(np.array(biases.astype(mx.float16)).tobytes())

                print(f"Quantized and streamed: {src_name}")
            else:
                # Cast and write
                out_dtype = "F32" if (src_name.endswith(".gate.weight") or "gate.expert_bias" in src_name) else "F16"
                x_cast = x.astype(mx.float32 if out_dtype == "F32" else mx.float16)
                mx.eval(x_cast)

                out_f.write(np.array(x_cast).tobytes())
                print(f"Cast and streamed: {src_name}")

            processed_sources.add(src_name)

            # Clean memory immediately
            del x
            gc.collect()
            mx.metal.clear_cache() # Clear GPU memory allocations

    # Close handles
    if current_shard_tensors is not None:
        del current_shard_tensors
        gc.collect()
        mx.metal.clear_cache()
    print("Weight conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and quantize Hugging Face model weights to MLX 4-bit format in a streaming, low-memory fashion.")
    parser.add_argument("--src-dir", required=True, help="Path to HF model source directory containing config.json and safetensors files")
    parser.add_argument("--dest-dir", required=True, help="Destination directory to save quantized weights and updated config")
    parser.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    parser.add_argument("--group-size", type=int, default=64, help="Quantization group size (default: 64)")
    parser.add_argument("--expert-bits", type=int, default=None,
                        help="Override bits for routed-expert tensors (M8 sweep)")
    parser.add_argument("--expert-group-size", type=int, default=None,
                        help="Override group size for routed-expert tensors (M8 sweep)")

    args = parser.parse_args()

    try:
        convert_and_quantize_streaming(args.src_dir, args.dest_dir, bits=args.bits,
                                       group_size=args.group_size,
                                       expert_bits=args.expert_bits,
                                       expert_group_size=args.expert_group_size)
    except Exception as e:
        print(f"Error during weight conversion: {e}")
        exit(1)
