#!/usr/bin/env python3
"""e014_analysis.py — E-014: batched-prefill chunk sweep (prefill speed only).
Pre-registered 2026-10-20 with paper/experiments/E-014.md.

    python3 paper/scripts/e014_analysis.py <rows chunk=1> <rows chunk=c2> ... (any number of files)

Each file is one bench process at one --batched-prefill setting (row field batchedPrefillBlocks is the
engine echo; files are grouped by it). Warmup excluded; envValid=false excluded and counted. Per chunk:
mean prefill seconds, mean prefill forwards, and the per-prompt geometric-mean prefill ratio vs chunk 1
with a bootstrap 95 % CI (10,000 resamples, own random.Random(20261018)). Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys

B, SEED = 10_000, 20261018


def ci(xs):
    rng = random.Random(SEED)
    m = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(B))
    return statistics.fmean(xs), m[int(0.025 * B)], m[int(0.975 * B) - 1]


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr); return 2
    by: dict[int, dict] = {}
    for p in sys.argv[1:]:
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
        inval = 0
        for r in map(json.loads, open(p)):
            if r.get("warmupIncluded") or int(r.get("run", 0)) < 0:
                continue
            if not r.get("envValid"):
                inval += 1; continue
            by.setdefault(r["batchedPrefillBlocks"], {})[(r["suite"], r["promptId"])] = r
        print(f"  envValid=false {inval}")
    base = by[1]
    print("\nchunk  n   mean prefill s  mean prefill fwd  geo-mean ratio vs chunk 1 [95 % CI]")
    res = {}
    for c in sorted(by):
        ks = [k for k in by[c] if k in base]
        lr = [math.log(by[c][k]["prefillSeconds"] / base[k]["prefillSeconds"]) for k in ks]
        t = ci(lr); res[c] = math.exp(t[0])
        print(f"  {c:4d} {len(ks):3d}   {statistics.fmean(by[c][k]['prefillSeconds'] for k in ks):8.3f}        "
              f"{statistics.fmean(by[c][k]['prefillForwards'] for k in ks):7.2f}        "
              f"{math.exp(t[0]):.3f} [{math.exp(t[1]):.3f}, {math.exp(t[2]):.3f}]")
    best = min(res, key=res.get)
    print(f"\nfastest chunk: {best} (ratio {res[best]:.3f})")
    print("pre-registered prediction (E-014):")
    big = max(res)
    print(f"  (C1) the largest chunk ({big}) is within 5 % of the fastest chunk's ratio ...... "
          f"{'HOLDS' if res[big] <= 1.05 * res[best] else 'FAILS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
