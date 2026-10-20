#!/usr/bin/env python3
"""e011_loss_breakdown.py — POST-HOC (written 2026-10-20 after E-011's result was seen; exploratory only).

    python3 paper/scripts/e011_loss_breakdown.py <rows.jsonl> <text.jsonl> <genLength> [--suite gsm8k_qual2000]

For each arm vs none, splits the questions none got right and the arm got wrong ("lost") by whether
the arm's generation was truncated (reached genLength with no EOS), and recomputes paired accuracy
on the questions where NEITHER generation was truncated. Also: mean generated tokens per arm, and
the 4-gram repetition fraction (1 − distinct/total word 4-grams) of lost vs. other generations.
Same answer-extraction rule as scripts/gsm8k_accuracy.py. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_accuracy import extract  # noqa: E402

BASE = "cache-clean-commit-k1"


def rep4(t: str) -> float:
    w = t.split(); g = [" ".join(w[i:i + 4]) for i in range(len(w) - 3)]
    return 1 - len(set(g)) / len(g) if g else 0.0


def main() -> int:
    a = sys.argv[1:]
    suite = "gsm8k_qual2000"
    if "--suite" in a:
        i = a.index("--suite"); suite = a[i + 1]; del a[i:i + 2]
    rp, tp, gl = a[0], a[1], int(a[2])
    for p in (rp, tp):
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    valid = {(r["arm"], r["promptId"]) for r in map(json.loads, open(rp))
             if not r.get("warmupIncluded") and int(r.get("run", 0)) >= 0 and r.get("suite") == suite and r.get("envValid")}
    sd = Path(__file__).resolve().parents[2] / "Tools/diffusion-bench/PromptSuites"
    gold = {q["id"]: q["answer"] for q in json.load(open(sd / f"{suite}.json"))["prompts"]}
    g = {}
    for r in map(json.loads, open(tp)):
        k = (r["arm"], r["promptId"])
        if int(r.get("run", 0)) < 0 or r.get("suite") != suite or k not in valid:
            continue
        g[k] = dict(ok=extract(r["text"]) == gold[r["promptId"]], tr=int(r["tokens"]) >= gl and r.get("eosBlock") == "none",
                    tok=int(r["tokens"]), rep=rep4(r["text"]))
    arms = sorted({x for x, _ in g} - {BASE})
    print(f"mean generated tokens: " + ", ".join(f"{x.replace('cache-', '').replace('-clean-k1', '')} {statistics.fmean(v['tok'] for (y, _), v in g.items() if y == x):.1f}" for x in [BASE] + arms))
    for arm in arms:
        ps = sorted(p for (x, p) in g if x == arm and (BASE, p) in g)
        lost = [p for p in ps if g[(BASE, p)]["ok"] and not g[(arm, p)]["ok"]]
        lt = sum(g[(arm, p)]["tr"] for p in lost)
        both = [p for p in ps if not g[(arm, p)]["tr"] and not g[(BASE, p)]["tr"]]
        l2 = sum(g[(BASE, p)]["ok"] and not g[(arm, p)]["ok"] for p in both)
        g2 = sum(g[(arm, p)]["ok"] and not g[(BASE, p)]["ok"] for p in both)
        others = [p for p in ps if p not in lost]
        print(f"\n{arm}: lost {len(lost)} (arm truncated in {lt}, not truncated in {len(lost) - lt})")
        print(f"  questions with neither generation truncated: {len(both)}; arm {sum(g[(arm, p)]['ok'] for p in both)} vs none {sum(g[(BASE, p)]['ok'] for p in both)}; lost {l2} gained {g2}")
        if lost:
            print(f"  4-gram repetition, arm text: lost {statistics.fmean(g[(arm, p)]['rep'] for p in lost):.3f} vs other {statistics.fmean(g[(arm, p)]['rep'] for p in others):.3f}; "
                  f"none's text on the same lost questions {statistics.fmean(g[(BASE, p)]['rep'] for p in lost):.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
