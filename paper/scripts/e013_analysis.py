#!/usr/bin/env python3
"""e013_analysis.py — E-013: 8-bit vs 4-bit weights as the Studio serving format.
Pre-registered 2026-10-20 with paper/experiments/E-013.md.

    python3 paper/scripts/e013_analysis.py <rows 4bit r1> <text 4bit r1> <rows 8bit r1> <text 8bit r1> \
                                           <rows 4bit r2> <text 4bit r2> <rows 8bit r2> <text 8bit r2> <genLength>

Process order was 4bit r1 → 8bit r1 → 4bit r2 → 8bit r2 (ABAB, to spread host drift). Uncompressed KV
cache, per-block prefill. Warmup excluded; envValid=false rows excluded and counted. Per suite:
  determinism: # questions whose text is byte-identical between r1 and r2 of the same artefact;
  throughput: tokens/s = sum tokensGenerated / sum totalSeconds over both repeats; the 8-bit/4-bit
    ratio with a bootstrap 95 % CI resampling prompts (same prompt set, ratio of sums);
  steps per generated token: same construction (8-bit/4-bit);
  ms/forward, peak memory (descriptive);
  accuracy (gsm8k_tr* suites, r1 texts): scripts/gsm8k_accuracy.py rule; lost/gained 8-bit vs 4-bit,
    exact McNemar p (scripts/e011_quality.py).
Bootstrap 10,000 resamples, 95 % percentile CI, own random.Random(20261018) per CI. Stdlib only.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gsm8k_accuracy import extract  # noqa: E402
from e011_quality import mcnemar_exact  # noqa: E402

B, SEED = 10_000, 20261018
GSM = ["gsm8k_tr0", "gsm8k_tr2000"]


def load(rp, tp, gl):
    rows, inval = {}, 0
    for r in map(json.loads, open(rp)):
        if r.get("warmupIncluded") or int(r.get("run", 0)) < 0:
            continue
        if not r.get("envValid"):
            inval += 1; continue
        rows[(r["suite"], r["promptId"])] = r
    text = {}
    for t in map(json.loads, open(tp)):
        k = (t["suite"], t["promptId"])
        if int(t.get("run", 0)) >= 0 and k in rows:
            text[k] = t
    return rows, text, inval


def ratio_ci(keys, num_a, den_a, num_b, den_b):
    """CI of (sum num_b / sum den_b) / (sum num_a / sum den_a) resampling keys."""
    rng = random.Random(SEED)
    def r(ks):
        a = sum(num_a[k] for k in ks) / max(sum(den_a[k] for k in ks), 1e-12)
        b = sum(num_b[k] for k in ks) / max(sum(den_b[k] for k in ks), 1e-12)
        return b / a
    pt = r(keys)
    bs = sorted(r(rng.choices(keys, k=len(keys))) for _ in range(B))
    return pt, bs[int(0.025 * B)], bs[int(0.975 * B) - 1]


def main() -> int:
    a = sys.argv[1:]
    if len(a) != 9:
        print(__doc__, file=sys.stderr); return 2
    gl = int(a[8])
    for p in a[:8]:
        print(f"input {p.rsplit('/', 1)[-1]} sha256 {hashlib.sha256(open(p, 'rb').read()).hexdigest()}")
    runs = {}
    for name, (rp, tp) in zip(["4r1", "8r1", "4r2", "8r2"], [(a[0], a[1]), (a[2], a[3]), (a[4], a[5]), (a[6], a[7])]):
        rows, text, inval = load(rp, tp, gl)
        runs[name] = (rows, text)
        print(f"{name}: {len(rows)} valid rows, envValid=false {inval}")
    sd = Path(__file__).resolve().parents[2] / "Tools/diffusion-bench/PromptSuites"
    gold = {(s, q["id"]): q["answer"] for s in GSM for q in json.load(open(sd / f"{s}.json"))["prompts"]}
    suites = sorted({s for (s, _) in runs["4r1"][0]})
    res = {}
    pooled_lost = pooled_gained = 0
    for s in suites:
        ks = sorted(k for k in runs["4r1"][0] if k[0] == s and all(k in runs[n][0] for n in runs))
        print(f"\n== {s} (n={len(ks)})")
        for w in "48":
            same = sum(runs[f"{w}r1"][1][k]["text"] == runs[f"{w}r2"][1][k]["text"] for k in ks)
            print(f"  {w}-bit determinism r1 vs r2: {same}/{len(ks)} texts identical")
        def agg(w, field):
            return {k: runs[f"{w}r1"][0][k][field] + runs[f"{w}r2"][0][k][field] for k in ks}
        tok4, sec4, tok8, sec8 = agg("4", "tokensGenerated"), agg("4", "totalSeconds"), agg("8", "tokensGenerated"), agg("8", "totalSeconds")
        st4, st8 = agg("4", "logicalSteps"), agg("8", "logicalSteps")
        tps = ratio_ci(ks, tok4, sec4, tok8, sec8)
        spt = ratio_ci(ks, st4, tok4, st8, tok8)
        res[(s, "tps")], res[(s, "spt")] = tps, spt
        for w, tok, sec, st in (("4", tok4, sec4, st4), ("8", tok8, sec8, st8)):
            fw = sum(runs[f"{w}r1"][0][k]["forwardsEvaluated"] + runs[f"{w}r2"][0][k]["forwardsEvaluated"] for k in ks)
            fs = sum(runs[f"{w}r1"][0][k]["forwardSeconds"] + runs[f"{w}r2"][0][k]["forwardSeconds"] for k in ks)
            pk = max(runs[f"{w}r1"][0][k]["peakMemoryGB"] for k in ks)
            print(f"  {w}-bit: tok/s {sum(tok.values()) / sum(sec.values()):.2f}  steps/token {sum(st.values()) / sum(tok.values()):.4f}  "
                  f"ms/fwd {1000 * fs / fw:.1f}  peak {pk:.2f} GB")
        print(f"  8/4 tok/s ratio {tps[0]:.3f} [{tps[1]:.3f}, {tps[2]:.3f}]   8/4 steps-per-token ratio {spt[0]:.3f} [{spt[1]:.3f}, {spt[2]:.3f}]")
        if s in GSM:
            o4 = {k: extract(runs["4r1"][1][k]["text"]) == gold[k] for k in ks}
            o8 = {k: extract(runs["8r1"][1][k]["text"]) == gold[k] for k in ks}
            lost = sum(o4[k] and not o8[k] for k in ks); gained = sum(o8[k] and not o4[k] for k in ks)
            pooled_lost += lost; pooled_gained += gained
            tr4 = sum(int(runs["4r1"][1][k]["tokens"]) >= gl and runs["4r1"][1][k].get("eosBlock") == "none" for k in ks)
            print(f"  accuracy 4-bit {sum(o4.values())}/{len(ks)}  8-bit {sum(o8.values())}/{len(ks)}  "
                  f"(8 vs 4: lost {lost}, gained {gained}, McNemar p {mcnemar_exact(lost, gained):.4f}); 4-bit truncated {tr4}")
    pv = mcnemar_exact(pooled_lost, pooled_gained)
    print(f"\npooled gsm8k_tr*: 8 vs 4 lost {pooled_lost} gained {pooled_gained}, McNemar p {pv:.4f}")
    print("\npre-registered predictions (E-013):")
    w1 = all(res[(s, "tps")][1] > 1 for s in GSM)
    print(f"  (W1) 8-bit tok/s ratio vs 4-bit CI > 1 on gsm8k_tr0 AND gsm8k_tr2000 ......... {'HOLDS' if w1 else 'FAILS'}")
    w2 = all(res[(s, "spt")][2] < 1 for s in GSM)
    print(f"  (W2) 8-bit steps-per-token ratio CI < 1 on both ................................. {'HOLDS' if w2 else 'FAILS'}")
    w3 = not (pv < 0.05 and pooled_lost > pooled_gained)
    print(f"  (W3) no detectable 8-bit accuracy loss vs 4-bit (pooled McNemar) ................ {'HOLDS' if w3 else 'FAILS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
