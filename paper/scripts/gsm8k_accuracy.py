#!/usr/bin/env python3
"""gsm8k_accuracy.py — GSM8K answer accuracy per suite and arm from a bench --dump-text file (exploratory).
Registered 2026-10-19 with paper/experiments/E-008.md.

    python3 paper/scripts/gsm8k_accuracy.py <text.jsonl> [--suites-dir Tools/diffusion-bench/PromptSuites]

Gold answers come from the suite JSON named by each row's `suite` field. Extraction rule (fixed
before the run): cut the generation at the first "\\nQuestion:" (few-shot continuation), then take
the integer after the last "####" if present, else the last integer in the cut text; commas removed.
Warmup rows (run < 0) skipped. Prints accuracy per (suite, arm) and paired lost/gained vs none.
Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE = "cache-clean-commit-k1"


def extract(text: str) -> str | None:
    t = text.split("\nQuestion:")[0].replace(",", "")
    if "####" in t:
        m = re.search(r"-?\d+", t.rsplit("####", 1)[1])
        if m:
            return m.group(0)
    nums = re.findall(r"-?\d+", t)
    return nums[-1] if nums else None


def main() -> int:
    a = sys.argv[1:]
    sd = Path(__file__).resolve().parents[2] / "Tools/diffusion-bench/PromptSuites"
    if "--suites-dir" in a:
        i = a.index("--suites-dir"); sd = Path(a[i + 1]); del a[i:i + 2]
    if len(a) != 1:
        print(__doc__, file=sys.stderr); return 2
    p = Path(a[0])
    print(f"input {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")
    gold: dict[str, dict[str, str]] = {}
    ok: dict[tuple, dict[str, bool]] = defaultdict(dict)
    for line in p.open():
        r = json.loads(line)
        if int(r.get("run", 0)) < 0:
            continue
        s = r["suite"]
        if s not in gold:
            gold[s] = {q["id"]: q.get("answer", "") for q in json.load(open(sd / f"{s}.json"))["prompts"]}
        g = gold[s].get(r["promptId"], "")
        if not g:
            continue
        ok[(s, r["arm"])][r["promptId"]] = extract(r["text"]) == g
    for s in sorted(gold):
        base = ok.get((s, BASE), {})
        for (s2, arm), v in sorted(ok.items()):
            if s2 != s:
                continue
            line = f"  {s:16s} {arm:24s} {sum(v.values()):3d}/{len(v):3d}"
            if arm != BASE and base:
                ks = set(v) & set(base)
                lost = sum(base[k] and not v[k] for k in ks); gained = sum(v[k] and not base[k] for k in ks)
                line += f"  vs none lost {lost} gained {gained}"
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
