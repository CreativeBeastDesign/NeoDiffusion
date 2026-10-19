#!/usr/bin/env python3
"""weight_interaction.py — difference of paired step deltas between two weight artefacts (same prompts, arms).

    python3 paper/scripts/weight_interaction.py <rows_artefactA.jsonl> <rows_artefactB.jsonl>

Per arm X and prompt p: d = (steps_A[X,p] − steps_A[none,p]) − (steps_B[X,p] − steps_B[none,p]).
Prints mean d with a bootstrap 95 % CI (10,000 resamples, seed 20261018, as e001_analysis.py).
Valid (envValid) non-warmup rows only; prompts missing in either file are dropped. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys

NONE = "cache-clean-commit-k1"


def load(p: str) -> dict:
    d = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("warmupIncluded") or not r.get("envValid"):
            continue
        d[(r["arm"], r["promptId"])] = r["logicalSteps"]
    return d


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    for p in sys.argv[1:]:
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    a, b = load(sys.argv[1]), load(sys.argv[2])
    arms = sorted({x for x, _ in a} - {NONE})
    for arm in arms:
        ps = sorted(p for x, p in a if x == arm and (NONE, p) in a and (arm, p) in b and (NONE, p) in b)
        d = [(a[(arm, p)] - a[(NONE, p)]) - (b[(arm, p)] - b[(NONE, p)]) for p in ps]
        rng = random.Random(20261018)
        bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(10000))
        print(f"  {arm:24s} n={len(d)} mean {sum(d) / len(d):+.3f} [{bs[250]:+.3f}, {bs[9749]:+.3f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
