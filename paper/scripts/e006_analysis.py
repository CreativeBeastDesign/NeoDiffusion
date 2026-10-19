#!/usr/bin/env python3
"""e006_analysis.py — E-006 weight-precision control: KV-mode step deltas across 4-bit / 8-bit / F16 weights.
Pre-registered 2026-10-19 with paper/experiments/E-006.md.

    python3 paper/scripts/e006_analysis.py <rows_4bit.jsonl> <rows_8bit.jsonl> <rows_f16.jsonl> [--suite gsm8k_fresh200 | --not-suite gsm8k_fresh200]

Rows: warmup excluded; envValid=false rows excluded and counted. A prompt enters an analysis only
if every arm it needs is valid in every artefact it needs. Bootstrap: 10,000 resamples, 95 %
percentile CI as e001_analysis.py (index 250 / 9749 of the sorted means); each CI uses its own
random.Random(20261018), so results do not depend on print order.

Quantities (per prompt p, arm X, artefact w; none = cache-clean-commit-k1):
  delta_w(X, p) = steps_w[X, p] − steps_w[none, p]
  diff_{w1−w2}(X, p) = delta_w1(X, p) − delta_w2(X, p)                    (weight interaction)
  key_{w1−w2}(p) = diff_{w1−w2}(k4v8, p) − diff_{w1−w2}(k8v4, p)          (Key specificity)
Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys

BASE = "cache-clean-commit-k1"
ARMS = {"k8v4": "cache-k8v4-clean-k1", "k4v8": "cache-k4v8-clean-k1",
        "fp4": "cache-fp4-clean-k1", "kivi4": "cache-kivi4-clean-k1"}
B, SEED = 10_000, 20261018


def ci(xs: list[float]) -> tuple[float, float, float]:
    rng = random.Random(SEED)
    means = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(B))
    return statistics.fmean(xs), means[int(0.025 * B)], means[int(0.975 * B) - 1]


def fmt(t: tuple[float, float, float], n: int) -> str:
    return f"n={n:3d} mean {t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def load(path: str, keep) -> tuple[dict, int, int]:
    steps, invalid, total = {}, 0, 0
    for line in open(path):
        r = json.loads(line)
        if r.get("warmupIncluded") or int(r.get("run", 0)) < 0 or not keep(r.get("suite", "")):
            continue
        total += 1
        if not r.get("envValid"):
            invalid += 1
            continue
        steps[(r["arm"], r["promptId"])] = r["logicalSteps"]
    return steps, invalid, total


def main() -> int:
    args = sys.argv[1:]
    keep, label = (lambda s: True), "all suites"
    if "--suite" in args:
        i = args.index("--suite"); name = args[i + 1]; del args[i:i + 2]
        keep, label = (lambda s, n=name: s == n), f"suite == {name}"
    elif "--not-suite" in args:
        i = args.index("--not-suite"); name = args[i + 1]; del args[i:i + 2]
        keep, label = (lambda s, n=name: s != n), f"suite != {name}"
    if len(args) != 3:
        print(__doc__, file=sys.stderr); return 2
    names = ["4bit", "8bit", "f16"]
    data = {}
    print(f"prompt filter: {label}")
    for w, p in zip(names, args):
        s, inv, tot = load(p, keep)
        data[w] = s
        print(f"input {w}: {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}  rows {tot}, envValid=false {inv}")

    def delta(w, x, p):
        return data[w][(ARMS[x], p)] - data[w][(BASE, p)]

    def prompts(ws, xs):
        ps = {p for (_, p) in data[ws[0]]}
        return sorted(p for p in ps if all((a, p) in data[w] for w in ws for a in [BASE] + [ARMS[x] for x in xs]))

    res = {}
    print("\npaired logical-step deltas vs none, per artefact:")
    for w in names:
        for x in ARMS:
            ps = prompts([w], [x]); d = [delta(w, x, p) for p in ps]
            res[("delta", w, x)] = ci(d)
            print(f"  {w:5s} {x:6s} {fmt(res[('delta', w, x)], len(d))}")

    print("\nweight interaction: delta_w1 − delta_w2, per arm:")
    for w1, w2 in (("4bit", "f16"), ("4bit", "8bit"), ("8bit", "f16")):
        for x in ARMS:
            ps = prompts([w1, w2], [x]); d = [delta(w1, x, p) - delta(w2, x, p) for p in ps]
            res[("diff", w1, w2, x)] = ci(d)
            print(f"  {w1}−{w2:5s} {x:6s} {fmt(res[('diff', w1, w2, x)], len(d))}")

    print("\nKey specificity: (delta_w1 − delta_w2)[k4v8] − (delta_w1 − delta_w2)[k8v4]:")
    for w1, w2 in (("4bit", "f16"), ("4bit", "8bit")):
        ps = prompts([w1, w2], ["k4v8", "k8v4"])
        d = [(delta(w1, "k4v8", p) - delta(w2, "k4v8", p)) - (delta(w1, "k8v4", p) - delta(w2, "k8v4", p)) for p in ps]
        res[("key", w1, w2)] = ci(d)
        print(f"  {w1}−{w2:5s} {fmt(res[('key', w1, w2)], len(d))}")

    pos = lambda t: t[1] > 0
    spans = lambda t: t[1] <= 0 <= t[2]
    print("\npre-registered predictions (E-006; meaningful on --suite gsm8k_fresh200 only):")
    p1 = pos(res[("delta", "4bit", "k4v8")]) and pos(res[("delta", "4bit", "fp4")]) and spans(res[("delta", "4bit", "k8v4")])
    print(f"  (P1) 4-bit replication: k4v8 & fp4 CI > 0, k8v4 CI spans 0 ........ {'HOLDS' if p1 else 'FAILS'}")
    p2 = pos(res[("diff", "4bit", "f16", "k4v8")])
    print(f"  (P2) weight interaction: 4bit−f16 for k4v8 CI > 0 ................. {'HOLDS' if p2 else 'FAILS'}")
    p3 = pos(res[("key", "4bit", "f16")])
    print(f"  (P3) Key specificity: 4bit−f16 k4v8 minus k8v4 CI > 0 ............. {'HOLDS' if p3 else 'FAILS'}")
    p4 = spans(res[("delta", "8bit", "k4v8")]) and pos(res[("diff", "4bit", "8bit", "k4v8")])
    print(f"  (P4) dose: 8-bit k4v8 CI spans 0 and 4bit−8bit k4v8 CI > 0 ........ {'HOLDS' if p4 else 'FAILS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
