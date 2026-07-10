#!/usr/bin/env python3
"""M8 E3 — layer-streamed BF16 reference logits for the drift corpus (m8-logbook).

Recomputes reference logits for the windows dumped by `LLaDADriftCorpusDumper`
(scratch/m8_drift/corpus.safetensors) using the *reference* decoder implementation
(modeling_llada2_moe.py classes, maximum fidelity), with the ~33 GB BF16 checkpoint
streamed **layer by layer**: the whole corpus is batched through layer i before layer
i+1 loads, so the checkpoint is read once total (the Sumi §2 technique, corpus-batched).

Fits the 16 GB dev M1: at any moment only one layer's weights (~1.6 GB) plus the corpus
hidden states (~tens of MB) are resident. Attention runs under the strict 0/-inf block
mask (the corrected baseline — phase-2 §6 deviation 8), matching the engine.

Compute dtype: float32 on CPU (the parity-relevant paths are fp32 in the reference too;
BF16 storage is upcast per layer). Slow is fine — one-off fixture generation.

Usage:
  python3 Tools/m8_reference_logits.py \
      --hf-dir models/llada2-1-mini --corpus scratch/m8_drift --out scratch/m8_drift
Produces: reference_logits.safetensors (float32, per window key: "<key>.logitsRef")
"""
import argparse
import gc
import importlib
import json
import os
import shutil
import sys
import tempfile

import torch
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file


def load_reference_module(hf_dir):
    """The modeling file uses a relative import — stage the .py files into a synthetic
    package dir (same technique as generate_core_fixtures.import_reference_package,
    but sourced from the local checkpoint dir instead of the hub cache)."""
    pkg_dir = tempfile.mkdtemp(prefix="llada2_ref_")
    pkg_name = "llada2_reference"
    dest = os.path.join(pkg_dir, pkg_name)
    os.makedirs(dest)
    open(os.path.join(dest, "__init__.py"), "w").close()
    for fname in ["configuration_llada2_moe.py", "modeling_llada2_moe.py"]:
        shutil.copy(os.path.join(hf_dir, fname), os.path.join(dest, fname))
    sys.path.insert(0, pkg_dir)
    cfg_mod = importlib.import_module(f"{pkg_name}.configuration_llada2_moe")
    mod = importlib.import_module(f"{pkg_name}.modeling_llada2_moe")
    return cfg_mod, mod


def build_config(cfg_mod, hf_dir):
    with open(os.path.join(hf_dir, "config.json")) as f:
        raw = json.load(f)
    raw.pop("architectures", None)
    raw.pop("auto_map", None)
    raw.pop("model_type", None)
    config = cfg_mod.LLaDA2MoeConfig(**raw)
    # Built outside from_pretrained, _attn_implementation is None and the attention
    # forward's ALL_ATTENTION_FUNCTIONS lookup KeyErrors. Eager is the reference's own
    # path (fp32 softmax, additive mask) — exactly the wanted reference numerics.
    config._attn_implementation = "eager"
    return config


class ShardReader:
    """Random access to checkpoint tensors without loading whole shards."""

    def __init__(self, hf_dir):
        with open(os.path.join(hf_dir, "model.safetensors.index.json")) as f:
            self.weight_map = json.load(f)["weight_map"]
        self.hf_dir = hf_dir
        self.handles = {}

    def get(self, name):
        shard = self.weight_map[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(
                os.path.join(self.hf_dir, shard), framework="pt", device="cpu")
        return self.handles[shard].get_tensor(name)

    def state_dict_for(self, prefix):
        out = {}
        for name in self.weight_map:
            if name.startswith(prefix):
                out[name[len(prefix):]] = self.get(name).to(torch.float32)
        return out


class DequantArtefactReader:
    """Reads the converted 4-bit artefact and yields fp32-dequantized tensors —
    the pipeline-fidelity discriminator (m8-logbook): if THIS source reproduces the
    engine's corpus logits through the same python pipeline, the pipeline is faithful
    and any BF16-vs-4bit gap is real quantization drift, not an implementation bug.

    Layout: packed U8 [out, in/2] low-nibble-first (MLX affine 4-bit byte view),
    F16 scales/biases [out, in/64]; w = scale * q + bias per group of 64."""

    def __init__(self, artefact_dir, group_size=64):
        self.handle = safe_open(
            os.path.join(artefact_dir, "model.safetensors"), framework="pt", device="cpu")
        self.names = set(self.handle.keys())
        self.group = group_size

    def _dequant(self, base):
        packed = self.handle.get_tensor(f"{base}.weight")           # U8 [out, in/2]
        scales = self.handle.get_tensor(f"{base}.scales").to(torch.float32)
        biases = self.handle.get_tensor(f"{base}.biases").to(torch.float32)
        lo = (packed & 0x0F).to(torch.float32)
        hi = (packed >> 4).to(torch.float32)
        out_dim = packed.shape[0]
        q = torch.stack([lo, hi], dim=-1).reshape(out_dim, -1)      # [out, in]
        q = q.reshape(out_dim, -1, self.group)
        w = q * scales.unsqueeze(-1) + biases.unsqueeze(-1)
        return w.reshape(out_dim, -1)

    def get(self, name):
        if name in self.names:
            return self.handle.get_tensor(name)
        base = name[:-len(".weight")]
        assert name.endswith(".weight") and f"{base}.scales" in self.names, \
            f"{name}: not in artefact and not quantized"
        return self._dequant(base)

    def state_dict_for(self, prefix):
        out = {}
        bases = set()
        for name in self.names:
            if not name.startswith(prefix):
                continue
            if name.endswith(".scales") or name.endswith(".biases"):
                bases.add(name.rsplit(".", 1)[0])
            elif name.endswith(".weight") and f"{name[:-7]}.scales" in self.names:
                bases.add(name[:-7])
            else:
                out[name[len(prefix):]] = self.handle.get_tensor(name).to(torch.float32)
        for base in bases:
            out[f"{base[len(prefix):]}.weight"] = self._dequant(base)
        return out


def strict_block_mask(total_len, block_len, dtype=torch.float32):
    """0 on allowed pairs, -inf on future blocks (within-block bidirectional,
    cross-block causal — the corrected reference semantics)."""
    idx = torch.arange(total_len)
    qb = (idx // block_len).unsqueeze(1)
    kb = (idx // block_len).unsqueeze(0)
    mask = torch.zeros(total_len, total_len, dtype=dtype)
    mask[kb > qb] = float("-inf")
    return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--corpus", required=True, help="dir with corpus.safetensors + manifest.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dequant-artefact", default=None,
                    help="4-bit artefact dir: run the SAME pipeline on dequantized 4-bit "
                         "weights (pipeline-fidelity discriminator); output file becomes "
                         "reference_logits_dequant4bit.safetensors")
    args = ap.parse_args()

    torch.set_grad_enabled(False)
    cfg_mod, mod = load_reference_module(args.hf_dir)
    config = build_config(cfg_mod, args.hf_dir)
    if args.dequant_artefact:
        reader = DequantArtefactReader(args.dequant_artefact)
        out_name = "reference_logits_dequant4bit.safetensors"
    else:
        reader = ShardReader(args.hf_dir)
        out_name = "reference_logits.safetensors"

    with open(os.path.join(args.corpus, "manifest.json")) as f:
        manifest = json.load(f)["windows"]
    corpus = safe_open(
        os.path.join(args.corpus, "corpus.safetensors"), framework="pt", device="cpu")

    block_len = 32
    windows = []
    for entry in manifest:
        ids = corpus.get_tensor(f"{entry['key']}.ids").to(torch.long)  # [1, W]
        windows.append((entry, ids))
    print(f"[m8-ref] {len(windows)} windows; layers {config.num_hidden_layers}")

    # ---- Embeddings (streamed like a layer) ----
    emb_w = reader.get("model.word_embeddings.weight").to(torch.float32)
    hidden = []
    rotary = mod.LLaDA2MoeRotaryEmbedding(config=config)
    pos_embeds = []
    masks = []
    for entry, ids in windows:
        h = torch.nn.functional.embedding(ids, emb_w)
        W = ids.shape[1]
        position_ids = torch.arange(W).unsqueeze(0)
        cos, sin = rotary(h, position_ids)
        hidden.append(h)
        pos_embeds.append((cos, sin))
        masks.append(strict_block_mask(W, block_len))
    del emb_w
    gc.collect()
    print("[m8-ref] embeddings done")

    # ---- Decoder layers, streamed ----
    for li in range(config.num_hidden_layers):
        # meta-device construction + assign=True: weights become resident exactly once
        # (the fp32 copies from the shard reader). A normal fp32 construction would
        # transiently need ~13 GB per MoE layer (random init + loaded copy) — over budget.
        with torch.device("meta"):
            layer = mod.LLaDA2MoeDecoderLayer(config, li)
        layer.eval()
        sd = reader.state_dict_for(f"model.layers.{li}.")
        missing, unexpected = layer.load_state_dict(sd, strict=False, assign=True)
        real_missing = [m for m in missing if "rotary" not in m]
        if real_missing or unexpected:
            print(f"[m8-ref] layer {li} load: missing={real_missing} unexpected={unexpected}")
        # Any parameter still on meta was absent from the checkpoint — hard error, not skew.
        for name, p in list(layer.named_parameters()) + list(layer.named_buffers()):
            assert not p.is_meta, f"layer {li}: {name} missing from checkpoint"
        for wi, (entry, ids) in enumerate(windows):
            out = layer(
                hidden[wi],
                attention_mask=masks[wi],
                position_ids=torch.arange(ids.shape[1]).unsqueeze(0),
                position_embeddings=pos_embeds[wi],
            )
            hidden[wi] = out[0] if isinstance(out, tuple) else out
        del layer, sd
        gc.collect()
        print(f"[m8-ref] layer {li + 1}/{config.num_hidden_layers} done")

    # ---- Final norm + head (streamed) ----
    norm_w = reader.get("model.norm.weight").to(torch.float32)
    head_w = reader.get("lm_head.weight").to(torch.float32)
    eps = config.rms_norm_eps
    out_arrays = {}
    for wi, (entry, ids) in enumerate(windows):
        h = hidden[wi]
        var = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(var + eps) * norm_w
        active = entry["activeLen"]
        logits = h[0, -active:, :] @ head_w.T  # [B, V] float32
        out_arrays[f"{entry['key']}.logitsRef"] = logits.contiguous()
    save_file(out_arrays, os.path.join(args.out, out_name))
    print(f"[m8-ref] wrote {len(out_arrays)} logit sets to {args.out}/{out_name}")


if __name__ == "__main__":
    main()
