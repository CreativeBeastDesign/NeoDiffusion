#!/usr/bin/env python3
"""Summarise /tmp/neodiffusion-compute-gate.log per label prefix (text before the first '-').

    python3 Tools/analysis/compute_gate_report.py [log] [--prefix impl-x]

For each completed gated command: label, waited seconds (from request to start, includes
low-memory retries), ran seconds, rc. Totals per prefix (e.g. impl-x, impl-y, main)."""
import re, sys
from collections import defaultdict
args = [a for a in sys.argv[1:] if not a.startswith('--')]
path = args[0] if args else '/tmp/neodiffusion-compute-gate.log'
want = sys.argv[sys.argv.index('--prefix') + 1] if '--prefix' in sys.argv else None
starts, rows = {}, []
lowmem = defaultdict(int); waits_lock = defaultdict(int)
for ln in open(path):
    m = re.search(r'pid=(\d+) label=(\S+) (.*)', ln)
    if not m: continue
    pid, label, rest = m.groups()
    pre = label.rsplit('-', 1)[0] if label.startswith('impl-') else label.split('-')[0]
    pre = '-'.join(label.split('-')[:2]) if label.startswith('impl-') else label.split('-')[0]
    if rest.startswith('start'):
        starts[pid] = (label, int(re.search(r'waited=(\d+)s', rest).group(1)))
    elif rest.startswith('end') and pid in starts:
        lab, waited = starts.pop(pid)
        rows.append((pre, lab, waited, int(re.search(r'ran=(\d+)s', rest).group(1)), int(re.search(r'rc=(-?\d+)', rest).group(1))))
    elif rest.startswith('low-memory'): lowmem[pre] += 1
    elif rest.startswith('wait-lock'): waits_lock[pre] += 1
tot = defaultdict(lambda: [0, 0, 0, 0])
for pre, lab, w, r, rc in rows:
    if want and pre != want: continue
    print(f'{lab:40s} waited {w:6d}s  ran {r:6d}s  rc {rc}')
    t = tot[pre]; t[0] += 1; t[1] += w; t[2] += r; t[3] += rc != 0
print('---- per prefix: n, total waited s, total ran s, nonzero rc, lock-wait polls, low-mem retries')
for pre, (n, w, r, bad) in sorted(tot.items()):
    print(f'{pre:12s} n={n:3d} waited={w:6d}s ran={r:6d}s nonzero_rc={bad} lockpolls={waits_lock[pre]} lowmem={lowmem[pre]}')
for pid, (lab, w) in starts.items():
    print(f'RUNNING: {lab} (pid {pid})')
