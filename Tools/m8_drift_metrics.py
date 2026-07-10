#!/usr/bin/env python3
"""M8 E4 — drift metrics: 4-bit engine logits vs streamed-BF16 reference (m8-logbook).

Metrics per the M8 plan (phase-2 §4 M8 item 2), computed over the drift corpus's active
blocks, plus the margin-conditioned analysis (Sumi §3.3): flips binned by the reference's
top-1/top-2 probability margin, so razor-margin flips (expected, benign) don't pool with
wide-margin flips (damage). Confidence-margin drift is reported because the Γ/Δ
thresholds consume exactly that quantity.

Usage:
  python3 Tools/m8_drift_metrics.py --corpus scratch/m8_drift \
      [--alt-logits KEYSUFFIX]   # compare an alternative 4-bit variant instead
"""
import argparse
import json
import os

import numpy as np
from safetensors import safe_open


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--logits-key", default="logits4bit",
                    help="corpus tensor suffix holding the candidate logits")
    ap.add_argument("--candidate-file", default=None,
                    help="variant_<name>.safetensors from LLaDAVariantLogitsDumper "
                         "(keys '<key>.logits'); overrides --logits-key source")
    ap.add_argument("--reference-file", default="reference_logits.safetensors",
                    help="reference logits file; pass a variant file + --reference-key "
                         "to compare two 4-bit variants directly")
    ap.add_argument("--candidate-key", default="logits")
    ap.add_argument("--reference-key", default="logitsRef")
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    with open(os.path.join(args.corpus, "manifest.json")) as f:
        manifest = json.load(f)["windows"]
    if args.candidate_file:
        cand = safe_open(os.path.join(args.corpus, args.candidate_file),
                         framework="np", device="cpu")
        args.logits_key = args.candidate_key
    else:
        cand = safe_open(os.path.join(args.corpus, "corpus.safetensors"),
                         framework="np", device="cpu")
    ref = safe_open(os.path.join(args.corpus, args.reference_file),
                    framework="np", device="cpu")

    top1_total, top1_agree = 0, 0
    topk_overlap = []
    dp_max, dp_mean = [], []
    margin_drift = []
    # margin bins per Sumi §3.3
    bins = [(0.0, 0.01), (0.01, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.01)]
    bin_flips = {b: [0, 0] for b in bins}  # (flips, total)

    for entry in manifest:
        key = entry["key"]
        lc = cand.get_tensor(f"{key}.{args.logits_key}").astype(np.float32)
        lr = ref.get_tensor(f"{key}.{args.reference_key}").astype(np.float32)
        if lc.ndim == 3: lc = lc[0]   # dumper keeps the batch dim
        if lr.ndim == 3: lr = lr[0]
        pc, pr = softmax(lc), softmax(lr)

        t1c, t1r = lc.argmax(-1), lr.argmax(-1)
        top1_total += len(t1r)
        top1_agree += int((t1c == t1r).sum())

        kc = np.argsort(-lc, axis=-1)[:, :args.topk]
        kr = np.argsort(-lr, axis=-1)[:, :args.topk]
        for row_c, row_r in zip(kc, kr):
            topk_overlap.append(len(set(row_c) & set(row_r)) / args.topk)

        dp = np.abs(pc - pr)
        dp_max.append(dp.max())
        dp_mean.append(dp.mean())

        # Confidence margin (top-1 prob − top-2 prob) — what Γ/Δ thresholds consume.
        sr = np.sort(pr, axis=-1)
        sc = np.sort(pc, axis=-1)
        m_r = sr[:, -1] - sr[:, -2]
        m_c = sc[:, -1] - sc[:, -2]
        margin_drift.extend(np.abs(m_c - m_r).tolist())

        flips = t1c != t1r
        for lo, hi in bins:
            sel = (m_r >= lo) & (m_r < hi)
            bin_flips[(lo, hi)][0] += int(flips[sel].sum())
            bin_flips[(lo, hi)][1] += int(sel.sum())

    md = np.array(margin_drift)
    print(f"windows: {len(manifest)} | positions: {top1_total}")
    print(f"top-1 agreement: {top1_agree}/{top1_total} = {top1_agree / top1_total:.4f}")
    print(f"top-{args.topk} overlap: mean {np.mean(topk_overlap):.4f} "
          f"min {np.min(topk_overlap):.2f}")
    print(f"|Δp|: max {np.max(dp_max):.4g} | mean-of-means {np.mean(dp_mean):.4g}")
    print(f"|Δmargin|: mean {md.mean():.4g} p99 {np.percentile(md, 99):.4g} "
          f"max {md.max():.4g}")
    print("top-1 flips by reference margin bin (flips/total):")
    for (lo, hi), (fl, tot) in bin_flips.items():
        rate = f"{fl / tot:.3f}" if tot else "n/a"
        print(f"  [{lo:.2f}, {hi:.2f}): {fl}/{tot} ({rate})")
    largest_flip_margin = 0.0
    # second pass for the calibration headline: largest reference margin that flipped
    for entry in manifest:
        key = entry["key"]
        lc = cand.get_tensor(f"{key}.{args.logits_key}").astype(np.float32)
        lr = ref.get_tensor(f"{key}.{args.reference_key}").astype(np.float32)
        if lc.ndim == 3: lc = lc[0]
        if lr.ndim == 3: lr = lr[0]
        pr = softmax(lr)
        sr = np.sort(pr, axis=-1)
        m_r = sr[:, -1] - sr[:, -2]
        flips = lc.argmax(-1) != lr.argmax(-1)
        if flips.any():
            largest_flip_margin = max(largest_flip_margin, float(m_r[flips].max()))
    print(f"largest reference margin that flipped: {largest_flip_margin:.4f} "
          f"(compare against p99 |Δmargin| = {np.percentile(md, 99):.4f} — flips confined "
          f"below the measured noise scale are the benign-quantization signature)")


if __name__ == "__main__":
    main()
