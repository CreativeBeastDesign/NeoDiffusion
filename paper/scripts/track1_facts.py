#!/usr/bin/env python3
"""Derive the Track-1 facts (F-001..F-004) from the ingested R-005 data. Stdlib only.
    python3 paper/scripts/track1_facts.py
Rows: run >= 0, warmupIncluded false. R-005 has 96/96 non-warmup rows envValid (see R-005)."""
import hashlib, json
from collections import defaultdict
from pathlib import Path
F = Path(__file__).resolve().parents[1] / 'facts/data/2026-10-17-cache-campaign/track1_kv_starvation_k1.jsonl'
print('input', F.name, 'sha256', hashlib.sha256(F.read_bytes()).hexdigest())
rows = [json.loads(l) for l in F.open()]
rows = [r for r in rows if r.get('run', 0) >= 0 and not r.get('warmupIncluded')]
by = defaultdict(list)
for r in rows: by[r['arm']].append(r)
base = {r['promptId']: r for r in by['cache-clean-commit-k1']}
print(f"{'arm':26s} {'kvQuant':8s} n valid steps unmask one_tr one_tr/unmask fwd tok sec aggTPS ms/fwd sum_steps_delta_vs_fp16 prompts_more_steps prompts_fewer_steps")
for arm, rs in by.items():
    st = sum(r['logicalSteps'] for r in rs); um = sum(r['unmaskingSteps'] for r in rs); one = sum(r['starvationSteps'] for r in rs)
    fw = sum(r['forwardsEvaluated'] for r in rs); tok = sum(r['tokensGenerated'] for r in rs); sec = sum(r['totalSeconds'] for r in rs)
    d = [r['logicalSteps'] - base[r['promptId']]['logicalSteps'] for r in rs]
    print(f"{arm:26s} {rs[0]['kvQuantization']:8s} {len(rs)} {sum(r['envValid'] for r in rs)} {st} {um} {one} {one/um:.3f} {fw} {tok} {sec:.2f} {tok/sec:.3f} {1000*sec/fw:.1f} {sum(d):+d} {sum(x>0 for x in d)} {sum(x<0 for x in d)}")
