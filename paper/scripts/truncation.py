#!/usr/bin/env python3
"""Truncation check for E-002 (and any run with --dump-text). Pre-registered 2026-10-19.
    python3 paper/scripts/truncation.py <text.jsonl> <genLength> [--suite gsm8k]
Per arm: generations that reached genLength tokens, and how many of those have no EOS block."""
import json, sys
from collections import defaultdict
path, gl = sys.argv[1], int(sys.argv[2])
suite = sys.argv[sys.argv.index("--suite") + 1] if "--suite" in sys.argv else ""
agg = defaultdict(lambda: [0, 0, 0])
for l in open(path):
    t = json.loads(l)
    if int(t.get("run", 0)) < 0 or not t["promptId"].startswith(suite):
        continue
    a = agg[t["arm"]]; a[0] += 1
    if int(t["tokens"]) >= gl:
        a[1] += 1
        a[2] += t.get("eosBlock") == "none"
for arm, (n, full, noeos) in sorted(agg.items()):
    flag = "  <-- > 10 %: accuracy uninterpretable" if n and noeos / n > 0.10 else ""
    print(f"{arm:24s} n={n:3d} hit-limit={full:3d} hit-limit-no-EOS={noeos:3d} ({noeos / max(n, 1):.1%}){flag}")
