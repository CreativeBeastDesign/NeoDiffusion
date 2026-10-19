#!/usr/bin/env python3
"""rerun_identity.py — compare two bench runs of the same arms/prompts row by row.

    python3 paper/scripts/rerun_identity.py <rowsA.jsonl> <rowsB.jsonl> [<textA.jsonl> <textB.jsonl>]

Prints, per pair: input sha256s, envValid counts per file, the number of (arm, suite, promptId)
keys present in both, and how many differ in logicalSteps / forwardsEvaluated / tokensGenerated
(rows) and in generated text (text dumps). Warmup rows are skipped. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter


def sha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load(p: str) -> dict:
    d = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("warmupIncluded"):
            continue
        d[(r["arm"], r.get("suite"), r["promptId"])] = r
    return d


def main() -> int:
    if len(sys.argv) not in (3, 5):
        print(__doc__, file=sys.stderr)
        return 2
    ra, rb = sys.argv[1], sys.argv[2]
    for p in sys.argv[1:]:
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {sha(p)}")
    a, b = load(ra), load(rb)
    for name, d in (("A", a), ("B", b)):
        print(f"rows {name}: {len(d)} non-warmup; envValid {dict(Counter(r.get('envValid') for r in d.values()))}")
    keys = sorted(set(a) & set(b))
    print(f"keys in both: {len(keys)} (only A {len(set(a) - set(b))}, only B {len(set(b) - set(a))})")
    for field in ("logicalSteps", "forwardsEvaluated", "tokensGenerated"):
        n = sum(a[k].get(field) != b[k].get(field) for k in keys)
        print(f"  {field:18s} mismatches: {n}/{len(keys)}")
    if len(sys.argv) == 5:
        ta, tb = load(sys.argv[3]), load(sys.argv[4])
        tk = sorted(set(ta) & set(tb))
        n = sum(ta[k].get("text") != tb[k].get("text") for k in tk)
        print(f"  {'text':18s} mismatches: {n}/{len(tk)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
