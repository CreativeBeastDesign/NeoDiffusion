#!/usr/bin/env python3
"""e010_mechanism.py — E-010: where do Key-quantization's extra steps come from? Per-step traces.
Pre-registered 2026-10-20 with paper/experiments/E-010.md.

    python3 paper/scripts/e010_mechanism.py <traces.jsonl> <rows.jsonl> [--tau 0.7]

Trace records (bench --dump-traces): one per denoising step, fields arm, run, promptId, block, step,
conf[32] (top-1 probability per active-block position), masked[32], gamma[32], delta[32].
Warmup (run < 0 / arm "warmup") skipped; generations whose bench row is envValid=false are dropped.

Per generation (arm, prompt):
  FIRST STEP — the first trace record (first denoising step of the first generated block). Every arm
    has consumed byte-identical tokens up to here; only the cached prompt K/V differ (quantized or not).
      above1  = # masked positions with conf > tau            meanc1 = mean conf over masked positions
  TRAJECTORY —
      U = # steps with ≥ 1 masked position (unmasking steps)
      F = # unmasking steps whose max masked conf ≤ tau (fallback steps: Γ filled by the top-1 rule)
      P = # steps with no masked position (post / edit-only steps)
      G = # Γ commits (tokens unmasked)
Paired per prompt, arm − none, bootstrap 95 % CI (10,000 resamples, own random.Random(20261018) per
CI, index 250 / 9749 of sorted means). Fallback share for an arm: sum(ΔF) / sum(ΔU) over prompts,
CI by resampling prompts (resamples with sum(ΔU) ≤ 0 counted and dropped). Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random
import statistics
import sys

BASE = "cache-clean-commit-k1"
ARMS = {"k8v4": "cache-k8v4-clean-k1", "k4v8": "cache-k4v8-clean-k1", "kivi4": "cache-kivi4-clean-k1",
        "fp4": "cache-fp4-clean-k1"}
B, SEED = 10_000, 20261018


def ci(xs):
    rng = random.Random(SEED)
    m = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(B))
    return statistics.fmean(xs), m[int(0.025 * B)], m[int(0.975 * B) - 1]


def share_ci(df, du):
    rng = random.Random(SEED); n = len(df); out = []; dropped = 0
    for _ in range(B):
        idx = [rng.randrange(n) for _ in range(n)]
        su = sum(du[i] for i in idx)
        if su <= 0:
            dropped += 1; continue
        out.append(sum(df[i] for i in idx) / su)
    out.sort()
    point = sum(df) / sum(du) if sum(du) > 0 else float("nan")
    if not out:
        return point, float("nan"), float("nan"), dropped
    return point, out[int(0.025 * len(out))], out[int(0.975 * len(out)) - 1], dropped


def fmt(t, n):
    return f"n={n:3d} mean {t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main() -> int:
    a = sys.argv[1:]
    tau = 0.7
    if "--tau" in a:
        i = a.index("--tau"); tau = float(a[i + 1]); del a[i:i + 2]
    if len(a) != 2:
        print(__doc__, file=sys.stderr); return 2
    tp, rp = a
    for p in (tp, rp):
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    valid, inval = set(), 0
    for line in open(rp):
        r = json.loads(line)
        if r.get("warmupIncluded") or int(r.get("run", 0)) < 0:
            continue
        if r.get("envValid"):
            valid.add((r["arm"], r["promptId"]))
        else:
            inval += 1
    print(f"rows: {len(valid)} valid generations, envValid=false {inval}; tau {tau}")
    g: dict[tuple, dict] = {}
    for line in open(tp):
        t = json.loads(line)
        if t.get("arm") in ("", "warmup") or int(t.get("run", 0)) < 0:
            continue
        k = (t["arm"], t["promptId"])
        if k not in valid:
            continue
        mc = [c for c, m in zip(t["conf"], t["masked"]) if m]
        s = g.get(k)
        if s is None:
            s = g[k] = {"above1": sum(c > tau for c in mc), "meanc1": statistics.fmean(mc) if mc else float("nan"),
                        "U": 0, "F": 0, "P": 0, "G": 0}
        if mc:
            s["U"] += 1; s["F"] += max(mc) <= tau
        else:
            s["P"] += 1
        s["G"] += sum(t["gamma"])
    arms_present = sorted({x for x, _ in g})
    print(f"generations with traces: {len(g)}; arms: {arms_present}")
    print("\nper-arm totals: U (unmasking steps) F (fallback) P (post) G (Γ commits) G/U  F/U  mean above1  mean meanc1")
    for arm in [BASE] + list(ARMS.values()):
        v = [s for (x, _), s in g.items() if x == arm]
        if not v:
            continue
        U = sum(s["U"] for s in v); F = sum(s["F"] for s in v); P = sum(s["P"] for s in v); G = sum(s["G"] for s in v)
        print(f"  {arm:24s} n={len(v):3d} U {U:6d} F {F:6d} P {P:5d} G {G:6d} G/U {G / max(U, 1):.3f} F/U {F / max(U, 1):.3f} "
              f"above1 {statistics.fmean(s['above1'] for s in v):.3f} meanc1 {statistics.fmean(s['meanc1'] for s in v):.4f}")
    res = {}
    print("\npaired per-prompt differences, arm − none:")
    for short, arm in ARMS.items():
        ps = sorted(p for (x, p) in g if x == arm and (BASE, p) in g)
        if not ps:
            continue
        for metric in ("above1", "meanc1", "F", "U", "P"):
            d = [g[(arm, p)][metric] - g[(BASE, p)][metric] for p in ps]
            res[(short, metric)] = ci(d)
            print(f"  {short:6s} {metric:7s} {fmt(res[(short, metric)], len(d))}")
        df = [g[(arm, p)]["F"] - g[(BASE, p)]["F"] for p in ps]
        du = [g[(arm, p)]["U"] - g[(BASE, p)]["U"] for p in ps]
        sh = share_ci(df, du); res[(short, "share")] = sh
        print(f"  {short:6s} share   sum(ΔF)/sum(ΔU) {sh[0]:.3f} [{sh[1]:.3f}, {sh[2]:.3f}]  (resamples dropped, sum ΔU ≤ 0: {sh[3]})")
    neg = lambda t: t[2] < 0
    pos = lambda t: t[1] > 0
    spans = lambda t: t[1] <= 0 <= t[2]
    print("\npre-registered predictions (E-010):")
    print(f"  (M1) first step, matched input: k4v8 above1 − none CI < 0 ................ {'HOLDS' if neg(res[('k4v8', 'above1')]) else 'FAILS'}")
    m2 = spans(res[("k8v4", "above1")]) and spans(res[("kivi4", "above1")])
    print(f"  (M2) first step: k8v4 and kivi4 above1 − none CI spans 0 .................. {'HOLDS' if m2 else 'FAILS'}")
    print(f"  (M3) trajectory: k4v8 fallback steps F − none CI > 0 ...................... {'HOLDS' if pos(res[('k4v8', 'F')]) else 'FAILS'}")
    m4 = res[("k4v8", "share")][1] > 0.5
    print(f"  (M4) k4v8 fallback share of extra unmasking steps, CI lower bound > 0.5 ..... {'HOLDS' if m4 else 'FAILS'}")
    m5 = spans(res[("k8v4", "F")]) and spans(res[("kivi4", "F")])
    print(f"  (M5) k8v4 and kivi4 fallback steps F − none CI spans 0 .................... {'HOLDS' if m5 else 'FAILS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
