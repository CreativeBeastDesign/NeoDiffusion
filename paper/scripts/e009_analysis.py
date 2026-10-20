#!/usr/bin/env python3
"""e009_analysis.py — E-009: Key/Value erosion asymmetry on a second model (LLaDA2.1-flash, 4-bit).
Pre-registered 2026-10-19 with paper/experiments/E-009.md.

    python3 paper/scripts/e009_analysis.py <rows.jsonl> [--suite gsm8k_flash200]

Per arm: paired per-prompt logical-step delta vs none; plus the direct Key-vs-Value contrast
steps[k4v8] − steps[k8v4] per prompt, and kivi4 − k4v8. Warmup excluded; envValid=false excluded
and counted. Bootstrap 10,000 resamples, 95 % percentile CI (index 250 / 9749), own
random.Random(20261018) per CI. Also per-arm totals and ms/forward. Stdlib only.
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


def ci(xs):
    rng = random.Random(SEED)
    m = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(B))
    return statistics.fmean(xs), m[int(0.025 * B)], m[int(0.975 * B) - 1]


def fmt(t, n):
    return f"n={n:3d} mean {t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main() -> int:
    a = sys.argv[1:]
    suite = None
    if "--suite" in a:
        i = a.index("--suite"); suite = a[i + 1]; del a[i:i + 2]
    if len(a) != 1:
        print(__doc__, file=sys.stderr); return 2
    p = a[0]
    print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}; suite filter: {suite or 'none'}")
    steps, agg, tot, inv = {}, {}, 0, 0
    for line in open(p):
        r = json.loads(line)
        if r.get("warmupIncluded") or int(r.get("run", 0)) < 0 or (suite and r.get("suite") != suite):
            continue
        tot += 1
        if not r.get("envValid"):
            inv += 1; continue
        steps[(r["arm"], r["promptId"])] = r["logicalSteps"]
        g = agg.setdefault(r["arm"], [0, 0, 0.0, 0])
        g[0] += r["logicalSteps"]; g[1] += r["forwardsEvaluated"]; g[2] += r["forwardSeconds"]; g[3] += r["tokensGenerated"]
    tc = json.loads(open(p).readline()).get("toolchain", {})
    print(f"rows {tot}, envValid=false {inv}; mlxCore {tc.get('mlxCoreVersion')} mlx-swift {tc.get('mlxSwiftPackage')} git {str(tc.get('gitCommit'))[:10]} dirty {tc.get('gitDirtyFileCount')}")
    print("\nper-arm aggregates: steps fwd tok ms/fwd")
    for arm in [BASE] + list(ARMS.values()):
        if arm in agg:
            g = agg[arm]; print(f"  {arm:24s} {g[0]:6d} {g[1]:6d} {g[3]:6d} {1000 * g[2] / max(g[1], 1):6.1f}")
    ps = sorted({q for (_, q) in steps})
    res = {}
    print("\npaired logical-step deltas vs none:")
    for x, arm in ARMS.items():
        qs = [q for q in ps if (arm, q) in steps and (BASE, q) in steps]
        v = [steps[(arm, q)] - steps[(BASE, q)] for q in qs]
        if not v:  # arm absent from this run (fix 2026-10-20: E-010/E-011 have no fp4 arm)
            continue
        res[x] = ci(v); print(f"  {x:6s} {fmt(res[x], len(v))}")
    def pair(x1, x2):
        qs = [q for q in ps if (ARMS[x1], q) in steps and (ARMS[x2], q) in steps]
        v = [steps[(ARMS[x1], q)] - steps[(ARMS[x2], q)] for q in qs]
        return ci(v), len(v)
    res["kv"], n1 = pair("k4v8", "k8v4"); print(f"\nk4v8 − k8v4: {fmt(res['kv'], n1)}")
    res["kk"], n2 = pair("kivi4", "k4v8"); print(f"kivi4 − k4v8: {fmt(res['kk'], n2)}")
    pos = lambda t: t[1] > 0
    spans = lambda t: t[1] <= 0 <= t[2]
    print("\npre-registered predictions (E-009; meaningful with --suite gsm8k_flash200):")
    print(f"  (1) k4v8 delta CI > 0 ................................ {'HOLDS' if pos(res['k4v8']) else 'FAILS'}")
    print(f"  (2) k8v4 delta CI spans 0 ............................ {'HOLDS' if spans(res['k8v4']) else 'FAILS'}")
    print(f"  (3) k4v8 − k8v4 CI > 0 (Key/Value asymmetry) ......... {'HOLDS' if pos(res['kv']) else 'FAILS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
