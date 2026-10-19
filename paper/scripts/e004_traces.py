#!/usr/bin/env python3
"""E-004 trace analysis — committed before the run (paper/experiments/E-004.md). Stdlib only.
    python3 paper/scripts/e004_traces.py <traces.jsonl> <text.jsonl> <rows.jsonl> [--tau 0.7]
H2: fraction of masked-position top-1 confidences above tau, by step-in-block 0..3, per arm.
H3: at the first generated-token position where an arm's final text tokens differ from none's,
compare the confidence each arm committed that position with (the trace step where gamma=1 or the
last edit). Reported: per arm, divergent prompts and how many committed with conf >= none's."""
import json, sys
from collections import defaultdict
BASE = "cache-clean-commit-k1"
traces_p, text_p, rows_p = sys.argv[1], sys.argv[2], sys.argv[3]
plen = {(r["arm"], r["promptId"]): r["promptTokens"] for r in map(json.loads, open(rows_p)) if r.get("run", 0) >= 0}
tau = float(sys.argv[sys.argv.index("--tau") + 1]) if "--tau" in sys.argv else 0.7
above = defaultdict(lambda: defaultdict(lambda: [0, 0]))      # arm -> step -> [above, total]
# EXPLORATORY, added 2026-10-19 after registration but before the run (question from André:
# "did the threshold ever fail so the top-1 fallback had to unmask?"): a step is a FALLBACK step if
# masked positions exist and none has confidence > tau, so Γ is filled by the top-1 fallback.
fallback = defaultdict(lambda: [0, 0])                          # arm -> [fallback steps, unmask-eligible steps]
commit_conf = defaultdict(dict)                                 # (arm, prompt) -> {(block,pos): conf}
for l in open(traces_p):
    t = json.loads(l)
    arm = t.get("arm", "")
    if arm in ("", "warmup") or t.get("run", 0) < 0:
        continue
    key = (arm, t["promptId"])
    mconf = [c for c, m in zip(t["conf"], t["masked"]) if m]
    if mconf:
        f = fallback[arm]; f[1] += 1; f[0] += max(mconf) <= tau
    for i, (c, m) in enumerate(zip(t["conf"], t["masked"])):
        if m and t["step"] <= 3:
            a = above[arm][t["step"]]; a[0] += c > tau; a[1] += 1
        if t["gamma"][i] or t["delta"][i]:
            commit_conf[key][(t["block"], i)] = c
print(f"H2 — fraction of masked-position confidences > {tau}, by step-in-block:")
arms = sorted(above)
print("  " + f"{'arm':24s}" + "".join(f"  step{s}" for s in range(4)))
for arm in arms:
    print("  " + f"{arm:24s}" + "".join(f"  {above[arm][s][0] / max(above[arm][s][1], 1):.3f}" for s in range(4)))
print("\nFallback steps (masked positions present, none above tau → top-1 forced) — exploratory:")
for arm in arms:
    f = fallback[arm]
    print(f"  {arm:24s} {f[0]:5d} / {f[1]:5d} steps = {f[0] / max(f[1], 1):.3f}")
toks = {}
for l in open(text_p):
    t = json.loads(l)
    if int(t.get("run", 0)) >= 0:
        toks[(t["arm"], t["promptId"])] = [int(x) for x in t["tokenIds"].split(",") if x]
print("\nH3 — first divergence vs none (position counted from first generated token):")
for arm in arms:
    if arm == BASE:
        continue
    div = notlower = known = 0
    for (a, pid), seq in toks.items():
        if a != arm or (BASE, pid) not in toks:
            continue
        ref = toks[(BASE, pid)]
        j = next((k for k in range(min(len(seq), len(ref))) if seq[k] != ref[k]), None)
        if j is None:
            continue
        div += 1
        # absolute position of generated token j = promptTokens + j; trace block index is the
        # absolute block (prefill blocks included) and positions are block-local.
        absj = plen.get((arm, pid), 0) + j
        bj, ij = absj // 32, absj % 32
        ca = [commit_conf.get((arm, pid), {}).get((bj, ij))]
        cb = [commit_conf.get((BASE, pid), {}).get((bj, ij))]
        ca, cb = [x for x in ca if x is not None], [x for x in cb if x is not None]
        if ca and cb:
            known += 1
            notlower += max(ca) >= max(cb)
    print(f"  {arm:24s} divergent prompts {div:3d}; conf known {known:3d}; arm conf >= none's: {notlower}")
print("\nNote: assumes trace `block` is the absolute block index (prefill blocks included) — verify on"
      " the first run (H3 'conf known' should be close to 'divergent prompts'); exploratory either way.")
