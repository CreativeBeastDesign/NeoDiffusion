#!/usr/bin/env python3
"""e012_analysis.py — E-012: batched prefill (GenerationParams.batchedPrefillBlocks) vs per-block prefill.
Pre-registered 2026-10-20 with paper/experiments/E-012.md.

    python3 paper/scripts/e012_analysis.py <rows_off> <text_off> <rows_c16> <text_c16> <rows_c128> <text_c128> <genLength>

Each pair is one bench process with one prefill setting (row field `batchedPrefillBlocks` is the
engine's effective echo and is checked). Conditions are the context suites gsm8k_bp0 / gsm8k_bp2000 /
gsm8k_bp3000. Warmup excluded; envValid=false rows excluded and counted. Per suite and batched
setting vs off (paired by question):
  prefill seconds: geometric-mean ratio exp(mean log(batched/off)) with bootstrap 95 % CI;
  seconds per generation: same;
  trajectory: # questions with byte-identical generated text; paired logical-step delta (CI);
  accuracy: scripts/gsm8k_accuracy.py rule; lost/gained; exact McNemar p (scripts/e011_quality.py).
Pooled over the three suites: accuracy + McNemar per batched setting; truncation of `off`.
Questions gsm8k-1200/1201 (seen in the pre-registration smoke test) are excluded.
Bootstrap 10,000 resamples, 95 % percentile CI, own random.Random(20261018) per CI. Stdlib only.
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
from gsm8k_accuracy import extract  # noqa: E402
from e011_quality import mcnemar_exact  # noqa: E402

SUITES = ["gsm8k_bp0", "gsm8k_bp2000", "gsm8k_bp3000"]
# Run in the pre-registration smoke test (off vs chunk 128, all three suites) — excluded everywhere.
SMOKE_SEEN = {"gsm8k-1200", "gsm8k-1201"}
B, SEED = 10_000, 20261018


def ci(xs):
    rng = random.Random(SEED)
    m = sorted(statistics.fmean(rng.choices(xs, k=len(xs))) for _ in range(B))
    return statistics.fmean(xs), m[int(0.025 * B)], m[int(0.975 * B) - 1]


def load(rp, tp, gl, gold):
    rows, inval, chunks = {}, 0, set()
    for r in map(json.loads, open(rp)):
        if r.get("warmupIncluded") or int(r.get("run", 0)) < 0:
            continue
        if r["promptId"] in SMOKE_SEEN:
            continue
        if not r.get("envValid"):
            inval += 1; continue
        chunks.add(r.get("batchedPrefillBlocks"))
        rows[(r["suite"], r["promptId"])] = r
    out = {}
    for t in map(json.loads, open(tp)):
        k = (t["suite"], t["promptId"])
        if int(t.get("run", 0)) < 0 or k not in rows:
            continue
        r = rows[k]
        out[k] = dict(pre=r["prefillSeconds"], tot=r["totalSeconds"], steps=r["logicalSteps"], text=t["text"],
                      ok=extract(t["text"]) == gold[k], tr=int(t["tokens"]) >= gl and t.get("eosBlock") == "none")
    return out, inval, chunks


def main() -> int:
    a = sys.argv[1:]
    if len(a) != 7:
        print(__doc__, file=sys.stderr); return 2
    gl = int(a[6])
    sd = Path(__file__).resolve().parents[2] / "Tools/diffusion-bench/PromptSuites"
    gold = {(s, q["id"]): q["answer"] for s in SUITES for q in json.load(open(sd / f"{s}.json"))["prompts"]}
    for p in a[:6]:
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    data = {}
    for name, (rp, tp) in zip(["off", "c16", "c128"], [(a[0], a[1]), (a[2], a[3]), (a[4], a[5])]):
        d, inval, chunks = load(rp, tp, gl, gold)
        data[name] = d
        print(f"{name}: {len(d)} valid generations, envValid=false {inval}, effective batchedPrefillBlocks {sorted(chunks)}")
    off = data["off"]
    res = {}
    for s in SUITES:
        ks = sorted(k for k in off if k[0] == s)
        print(f"\n== {s} (n={len(ks)}): off prefill mean {statistics.fmean(off[k]['pre'] for k in ks):.3f} s, "
              f"s/gen {statistics.fmean(off[k]['tot'] for k in ks):.3f}, truncated {sum(off[k]['tr'] for k in ks)}")
        for name in ["c16", "c128"]:
            d = data[name]
            ps = [k for k in ks if k in d]
            lr = [math.log(d[k]["pre"] / off[k]["pre"]) for k in ps if d[k]["pre"] > 0 and off[k]["pre"] > 0]
            lt = [math.log(d[k]["tot"] / off[k]["tot"]) for k in ps]
            pr, tr_ = ci(lr) if lr else (float("nan"),) * 3, ci(lt)
            dst = ci([d[k]["steps"] - off[k]["steps"] for k in ps])
            same = sum(d[k]["text"] == off[k]["text"] for k in ps)
            lost = sum(off[k]["ok"] and not d[k]["ok"] for k in ps); gained = sum(d[k]["ok"] and not off[k]["ok"] for k in ps)
            res[(s, name)] = dict(pre=pr, tot=tr_, steps=dst)
            print(f"  {name:4s} prefill ratio {math.exp(pr[0]):.3f} [{math.exp(pr[1]):.3f}, {math.exp(pr[2]):.3f}] (n={len(lr)})  "
                  f"s/gen ratio {math.exp(tr_[0]):.3f} [{math.exp(tr_[1]):.3f}, {math.exp(tr_[2]):.3f}]")
            print(f"       text identical {same}/{len(ps)}; Δsteps {dst[0]:+.3f} [{dst[1]:+.3f}, {dst[2]:+.3f}]; "
                  f"acc {sum(d[k]['ok'] for k in ps)} vs off {sum(off[k]['ok'] for k in ps)} (lost {lost}, gained {gained})")
    print("\n== pooled over suites")
    offtr = sum(v["tr"] for v in off.values()); n_off = len(off)
    gate = n_off > 0 and offtr / n_off <= 0.10
    print(f"  off truncated {offtr}/{n_off} ({offtr / max(n_off, 1):.1%}); accuracy off {sum(v['ok'] for v in off.values())}/{n_off}")
    loss = {}
    for name in ["c16", "c128"]:
        d = data[name]; ps = [k for k in off if k in d]
        lost = sum(off[k]["ok"] and not d[k]["ok"] for k in ps); gained = sum(d[k]["ok"] and not off[k]["ok"] for k in ps)
        pv = mcnemar_exact(lost, gained)
        loss[name] = pv < 0.05 and lost > gained
        print(f"  {name:4s} accuracy {sum(d[k]['ok'] for k in ps)}/{len(ps)}  lost {lost} gained {gained}  McNemar p {pv:.4f}")
    upper = lambda t: math.exp(t[2])
    spans = lambda t: t[1] <= 0 <= t[2]
    print("\npre-registered predictions (E-012):")
    print(f"  (S1) c128 prefill ratio at gsm8k_bp3000, CI upper < 0.5 .................. {'HOLDS' if upper(res[('gsm8k_bp3000', 'c128')]['pre']) < 0.5 else 'FAILS'}")
    print(f"  (S2) c16 prefill ratio at gsm8k_bp3000, CI upper < 1 .................... {'HOLDS' if upper(res[('gsm8k_bp3000', 'c16')]['pre']) < 1 else 'FAILS'}")
    print(f"  (S3) c128 s/gen ratio at gsm8k_bp3000, CI upper < 1 ..................... {'HOLDS' if upper(res[('gsm8k_bp3000', 'c128')]['tot']) < 1 else 'FAILS'}")
    t1 = all(spans(res[(s, n)]["steps"]) for s in SUITES for n in ["c16", "c128"])
    print(f"  (T1) no systematic step change: Δsteps CI spans 0 for every suite × setting . {'HOLDS' if t1 else 'FAILS'}")
    print(f"  (G)  truncation gate (off, pooled) ≤ 10 % .................................. {'PASSES' if gate else 'FAILS → Q1 not evaluated'}")
    q1 = gate and not any(loss.values())
    print(f"  (Q1) no detectable accuracy loss (pooled McNemar p < 0.05 with lost > gained) . {'HOLDS' if q1 else ('FAILS' if gate else 'n/a')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
