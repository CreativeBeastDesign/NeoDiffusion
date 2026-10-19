#!/usr/bin/env python3
"""genlen_identity.py — same arms/prompts at two generation lengths: are short completions identical?

    python3 paper/scripts/genlen_identity.py <text_short.jsonl> <genLenShort> <text_long.jsonl>

For every (arm, promptId) that ended with an EOS block and fewer than genLenShort tokens in the short
run, report whether the long run produced the same text. Per arm and overall. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict


def load(p: str) -> dict:
    d = {}
    for line in open(p):
        r = json.loads(line)
        if int(r.get("run", 0)) < 0:
            continue
        d[(r["arm"], r["promptId"])] = r
    return d


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    ps, gl, pl = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    for p in (ps, pl):
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    s, l = load(ps), load(pl)
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for k, r in s.items():
        if r.get("eosBlock") == "none" or int(r["tokens"]) >= gl or k not in l:
            continue
        agg[k[0]][0] += 1
        agg[k[0]][1] += r["text"] == l[k]["text"]
    for arm, (n, same) in sorted(agg.items()):
        print(f"{arm:24s} finished<{gl}: {n:3d}  identical at long length: {same:3d}")
    n, same = sum(v[0] for v in agg.values()), sum(v[1] for v in agg.values())
    print(f"{'all':24s} finished<{gl}: {n:3d}  identical at long length: {same:3d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
