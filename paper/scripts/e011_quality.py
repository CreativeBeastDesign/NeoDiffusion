#!/usr/bin/env python3
"""e011_quality.py — E-011: paired GSM8K accuracy at long context (10-shot, ~1.7k prompt tokens), gen 768.
Pre-registered 2026-10-20 with paper/experiments/E-011.md.

    python3 paper/scripts/e011_quality.py <rows.jsonl> <text.jsonl> <genLength> [--suite gsm8k_qual2000]

Answer extraction: identical to scripts/gsm8k_accuracy.py (imported). Truncation: a generation is
truncated if it reached genLength tokens with no EOS block. Generations whose bench row is
envValid=false are dropped. Per arm vs none (paired by question): accuracy, lost/gained, exact
two-sided McNemar p (binomial on discordant pairs, p = 0.5), and the paired accuracy difference with
a bootstrap 95 % CI (10,000 resamples, own random.Random(20261018)). Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_accuracy import extract  # noqa: E402  (same rule as E-008)

BASE = "cache-clean-commit-k1"
ARMS = {"k8v4": "cache-k8v4-clean-k1", "k4v8": "cache-k4v8-clean-k1", "kivi4": "cache-kivi4-clean-k1",
        "fp4": "cache-fp4-clean-k1"}
B, SEED = 10_000, 20261018


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def main() -> int:
    a = sys.argv[1:]
    suite = "gsm8k_qual2000"
    if "--suite" in a:
        i = a.index("--suite"); suite = a[i + 1]; del a[i:i + 2]
    if len(a) != 3:
        print(__doc__, file=sys.stderr); return 2
    rp, tp, gl = a[0], a[1], int(a[2])
    for p in (rp, tp):
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    valid, inval = set(), 0
    for line in open(rp):
        r = json.loads(line)
        if r.get("warmupIncluded") or int(r.get("run", 0)) < 0 or r.get("suite") != suite:
            continue
        if r.get("envValid"):
            valid.add((r["arm"], r["promptId"]))
        else:
            inval += 1
    sd = Path(__file__).resolve().parents[2] / "Tools/diffusion-bench/PromptSuites"
    gold = {q["id"]: q["answer"] for q in json.load(open(sd / f"{suite}.json"))["prompts"]}
    ok, trunc = {}, {}
    for line in open(tp):
        r = json.loads(line)
        if int(r.get("run", 0)) < 0 or r.get("suite") != suite or (r["arm"], r["promptId"]) not in valid:
            continue
        ok[(r["arm"], r["promptId"])] = extract(r["text"]) == gold[r["promptId"]]
        trunc[(r["arm"], r["promptId"])] = int(r["tokens"]) >= gl and r.get("eosBlock") == "none"
    print(f"suite {suite}; gen {gl}; valid generations {len(valid)}; envValid=false {inval}")
    arms = [BASE] + list(ARMS.values())
    print("\ntruncation (reached gen limit with no EOS):")
    for arm in arms:
        v = [t for (x, _), t in trunc.items() if x == arm]
        if v:
            flag = "  <-- > 10 %: accuracy uninterpretable" if arm == BASE and sum(v) / len(v) > 0.10 else ""
            print(f"  {arm:24s} {sum(v):3d}/{len(v):3d} ({sum(v) / len(v):.1%}){flag}")
    base_trunc = [t for (x, _), t in trunc.items() if x == BASE]
    gate = bool(base_trunc) and sum(base_trunc) / len(base_trunc) <= 0.10
    print("\naccuracy and paired comparison vs none:")
    res = {}
    bacc = {p: v for (x, p), v in ok.items() if x == BASE}
    print(f"  none   {sum(bacc.values()):3d}/{len(bacc):3d}")
    for short, arm in ARMS.items():
        ps = sorted(p for (x, p) in ok if x == arm and p in bacc)
        if not ps:
            continue
        lost = sum(bacc[p] and not ok[(arm, p)] for p in ps); gained = sum(ok[(arm, p)] and not bacc[p] for p in ps)
        d = [int(ok[(arm, p)]) - int(bacc[p]) for p in ps]
        rng = random.Random(SEED)
        m = sorted(statistics.fmean(rng.choices(d, k=len(d))) for _ in range(B))
        pv = mcnemar_exact(lost, gained)
        res[short] = (lost, gained, pv)
        print(f"  {short:6s} {sum(ok[(arm, p)] for p in ps):3d}/{len(ps):3d}  lost {lost:2d} gained {gained:2d}  McNemar p {pv:.4f}  "
              f"Δacc {statistics.fmean(d):+.3f} [{m[int(0.025 * B)]:+.3f}, {m[int(0.975 * B) - 1]:+.3f}]")
    loss = lambda s: s in res and res[s][2] < 0.05 and res[s][0] > res[s][1]
    print("\npre-registered predictions (E-011):")
    print(f"  (G)  truncation gate: ≤ 10 % of uncompressed-cache generations truncated ............ {'PASSES' if gate else 'FAILS → accuracy uninterpretable'}")
    print(f"  (Q1) no KV mode shows a detectable accuracy loss (McNemar p < 0.05, lost > gained) ... {'HOLDS' if gate and not any(loss(s) for s in res) else ('FAILS' if gate else 'n/a')}")
    for s in res:
        print(f"       {s:6s} detectable loss: {'YES' if loss(s) else 'no'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
