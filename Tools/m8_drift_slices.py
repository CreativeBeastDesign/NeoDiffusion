#!/usr/bin/env python3
"""M8 E4 follow-up — drift metrics sliced by window population (m8-logbook).

Sumi §3.2: aggregate metrics over mixed populations lie. The drift corpus mixes step-0
windows (active block fully masked → near-maximal-entropy predictions, flips cheap) with
late-step windows (mostly committed text, high-confidence predictions, flips = damage).
This slices top-1 agreement and margin-binned flips by step-in-block and by prompt.
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
    ap.add_argument("--candidate-file", default=None)
    ap.add_argument("--logits-key", default="logits4bit")
    ap.add_argument("--reference-file", default="reference_logits.safetensors")
    ap.add_argument("--reference-key", default="logitsRef")
    args = ap.parse_args()

    with open(os.path.join(args.corpus, "manifest.json")) as f:
        manifest = json.load(f)["windows"]
    if args.candidate_file:
        cand = safe_open(os.path.join(args.corpus, args.candidate_file),
                         framework="np", device="cpu")
        args.logits_key = "logits"
    else:
        cand = safe_open(os.path.join(args.corpus, "corpus.safetensors"),
                         framework="np", device="cpu")
    ref = safe_open(os.path.join(args.corpus, args.reference_file),
                    framework="np", device="cpu")

    def step_class(step):
        if step == 0:
            return "step0 (fully masked)"
        if step <= 4:
            return "early (1-4)"
        return "late (>=16)"

    slices = {}
    for entry in manifest:
        key = entry["key"]
        lc = cand.get_tensor(f"{key}.{args.logits_key}").astype(np.float32)
        lr = ref.get_tensor(f"{key}.{args.reference_key}").astype(np.float32)
        if lc.ndim == 3:
            lc = lc[0]
        if lr.ndim == 3:
            lr = lr[0]
        pr = softmax(lr)
        sr = np.sort(pr, axis=-1)
        m_r = sr[:, -1] - sr[:, -2]
        flips = lc.argmax(-1) != lr.argmax(-1)
        for sl in (step_class(entry["step"]), f"prompt:{entry['prompt']}"):
            d = slices.setdefault(sl, {"n": 0, "agree": 0,
                                       "confident_n": 0, "confident_flips": 0,
                                       "max_flip_margin": 0.0})
            d["n"] += len(flips)
            d["agree"] += int((~flips).sum())
            conf = m_r >= 0.20
            d["confident_n"] += int(conf.sum())
            d["confident_flips"] += int(flips[conf].sum())
            if flips.any():
                d["max_flip_margin"] = max(d["max_flip_margin"], float(m_r[flips].max()))

    print(f"{'slice':<26} {'top-1 agree':<14} {'flips@margin>=0.20':<20} {'max flip margin'}")
    for sl, d in sorted(slices.items()):
        agree = d["agree"] / d["n"]
        conf = (f"{d['confident_flips']}/{d['confident_n']} "
                f"({d['confident_flips'] / d['confident_n']:.3f})"
                if d["confident_n"] else "n/a")
        print(f"{sl:<26} {agree:<14.4f} {conf:<20} {d['max_flip_margin']:.4f}")


if __name__ == "__main__":
    main()
