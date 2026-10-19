#!/usr/bin/env python3
"""group_deltas.py — KV-mode step deltas per condition (τ or context length) and between conditions.
Pre-registered 2026-10-19 with paper/experiments/E-007.md (τ sweep) and E-008.md (context length).

    python3 paper/scripts/group_deltas.py --by tau  --suite gsm8k_tau200 <rows τ=0.5> <rows τ=0.7> <rows τ=0.9>
    python3 paper/scripts/group_deltas.py --by ctx  <rows.jsonl>

--by tau : condition = row field `thresholdMask` (the engine's effective echo), over the given files;
           --suite restricts to one suite (the primary fresh prompts); omit for all.
--by ctx : condition = row field `suite` (gsm8k_ctx0 / _ctx1000 / _ctx2000 / _ctx3500).
Keys are (condition, promptId); a prompt enters a contrast only if every needed arm is valid in
both conditions. Warmup rows excluded; envValid=false rows excluded and counted.
Bootstrap: 10,000 resamples, 95 % percentile CI (index 250 / 9749), own random.Random(20261018)
per CI. Prints pre-registered verdict lines for the lowest/highest condition. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys
from collections import Counter

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


def ctx_order(c):
    return int(c.replace("gsm8k_ctx", "")) if c.startswith("gsm8k_ctx") else 10 ** 9


def main() -> int:
    a = sys.argv[1:]
    if "--by" not in a:
        print(__doc__, file=sys.stderr); return 2
    i = a.index("--by"); by = a[i + 1]; del a[i:i + 2]
    suite = None
    if "--suite" in a:
        i = a.index("--suite"); suite = a[i + 1]; del a[i:i + 2]
    steps, rows_meta = {}, {}
    inval = Counter(); total = Counter()
    for p in a:
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
        for line in open(p):
            r = json.loads(line)
            if r.get("warmupIncluded") or int(r.get("run", 0)) < 0:
                continue
            if suite and r.get("suite") != suite:
                continue
            if by == "ctx" and not str(r.get("suite", "")).startswith("gsm8k_ctx"):
                continue
            c = f"{r['thresholdMask']:.2f}" if by == "tau" else r["suite"]
            total[c] += 1
            if not r.get("envValid"):
                inval[c] += 1; continue
            steps[(c, r["arm"], r["promptId"])] = r["logicalSteps"]
            m = rows_meta.setdefault((c, r["arm"]), [0, 0.0, 0, 0])
            m[0] += r["forwardsEvaluated"]; m[1] += r["forwardSeconds"]; m[2] += r["tokensGenerated"]; m[3] += r["logicalSteps"]
            m.append(r["totalSeconds"])
    conds = sorted({c for c, _, _ in steps}, key=(float if by == "tau" else ctx_order))
    print(f"grouping: {by}; suite filter: {suite or 'none'}; conditions: {conds}")
    for c in conds:
        print(f"  {c}: rows {total[c]}, envValid=false {inval[c]}")

    def prompts(cs, arms):
        ps = {p for (_, _, p) in steps}
        return sorted(p for p in ps if all((c, x, p) in steps for c in cs for x in [BASE] + arms))

    def d(c, x, p):
        return steps[(c, ARMS[x], p)] - steps[(c, BASE, p)]

    print("\nper-condition aggregates (valid rows): steps, forwards, tokens, ms/forward, total s per generation")
    for c in conds:
        for x in [BASE] + list(ARMS.values()):
            m = rows_meta.get((c, x))
            if m:
                tot = m[4:]
                print(f"  {c:14s} {x:24s} steps {m[3]:6d} fwd {m[0]:6d} tok {m[2]:6d} ms/fwd {1000 * m[1] / max(m[0], 1):6.1f} s/gen {statistics.fmean(tot):6.2f}")

    res = {}
    print("\npaired logical-step deltas vs none, per condition:")
    for c in conds:
        for x in ARMS:
            ps = prompts([c], [ARMS[x]]); v = [d(c, x, p) for p in ps]
            res[("d", c, x)] = ci(v)
            print(f"  {c:14s} {x:6s} {fmt(res[('d', c, x)], len(v))}")

    lo, hi = conds[0], conds[-1]
    print(f"\ncontrast: delta[{hi}] − delta[{lo}], per arm:")
    for x in ARMS:
        ps = prompts([lo, hi], [ARMS[x]]); v = [d(hi, x, p) - d(lo, x, p) for p in ps]
        res[("c", x)] = ci(v)
        print(f"  {x:6s} {fmt(res[('c', x)], len(v))}")
    ps = prompts([hi], [ARMS["kivi4"], ARMS["k4v8"]])
    v = [steps[(hi, ARMS["kivi4"], p)] - steps[(hi, ARMS["k4v8"], p)] for p in ps]
    res["kk"] = ci(v)
    print(f"\nkivi4 − k4v8 at {hi}: {fmt(res['kk'], len(v))}")

    pos = lambda t: t[1] > 0
    neg = lambda t: t[2] < 0
    spans = lambda t: t[1] <= 0 <= t[2]
    mid = [c for c in conds if c == "0.70"] or [hi]
    print("\npre-registered predictions:")
    print(f"  (1) k4v8 delta grows: delta[{hi}] − delta[{lo}] CI > 0 ................ {'HOLDS' if pos(res[('c', 'k4v8')]) else 'FAILS'}")
    ref = mid[0] if by == "tau" else hi
    print(f"  (2) k4v8 delta CI > 0 at {ref} ...................................... {'HOLDS' if pos(res[('d', ref, 'k4v8')]) else 'FAILS'}")
    ok3 = all(spans(res[("d", c, "k8v4")]) for c in conds)
    print(f"  (3) k8v4 delta CI spans 0 at every condition .......................... {'HOLDS' if ok3 else 'FAILS'}")
    if by == "ctx":
        print(f"  (4) kivi4 − k4v8 at {hi} CI < 0 ....................................... {'HOLDS' if neg(res['kk']) else 'FAILS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
