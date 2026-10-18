#!/usr/bin/env python3
"""E-001 analysis — written and committed BEFORE the run (pre-registration: paper/experiments/E-001.md).

    python3 paper/scripts/e001_analysis.py <rows.jsonl> <text.jsonl> [--suites-dir Tools/diffusion-bench/PromptSuites]

Stdlib only, deterministic (bootstrap seed fixed). Prints a plain-text report; nothing is written.

Rules applied exactly as registered:
- exclude the warmup row (run < 0 or warmupIncluded); exclude envValid=false rows and LIST them.
  Note: the envValid rule was amended to v2 on 2026-10-18 (after E-001's registration, before its run;
  .agents/AGENTS.md). Rows carry both `envValid` (v2, used here) and `envValidV1`; both counts are printed.
- primary metric: logical steps per prompt, paired by (run, suite, promptId) against arm `cache-clean-commit-k1`.
- predictions (E-001.md): (1) k4v8 and fp4 paired step delta > 0 with bootstrap 95 % CI excluding 0, and
  each larger than k8v4's; (2) k8v4's CI includes 0; (3) kivi4's delta below fp4's with the CI of the
  paired difference (kivi4 − fp4) excluding 0.
- secondary: forwards, tokens per unmasking step, gsm8k accuracy per arm (last integer in the output vs the
  suite's `answer`, commas stripped), aggregate TPS (descriptive only).
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

BASE = "cache-clean-commit-k1"
ARMS = {"none": BASE, "k8v4": "cache-k8v4-clean-k1", "k4v8": "cache-k4v8-clean-k1",
        "fp4": "cache-fp4-clean-k1", "kivi4": "cache-kivi4-clean-k1"}
B_RESAMPLES = 10_000
SEED = 20261018


def boot_ci(xs: list[float], rng: random.Random) -> tuple[float, float, float]:
    n = len(xs)
    means = sorted(statistics.fmean(rng.choices(xs, k=n)) for _ in range(B_RESAMPLES))
    return statistics.fmean(xs), means[int(0.025 * B_RESAMPLES)], means[int(0.975 * B_RESAMPLES) - 1]


def last_int(text: str) -> str | None:
    nums = re.findall(r"-?\d+", text.replace(",", ""))
    return nums[-1] if nums else None


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__); return 2
    rows_path, text_path = Path(sys.argv[1]), Path(sys.argv[2])
    suites_dir = Path(sys.argv[sys.argv.index("--suites-dir") + 1]) if "--suites-dir" in sys.argv \
        else Path(__file__).resolve().parents[2] / "Tools/diffusion-bench/PromptSuites"
    for p in (rows_path, text_path):
        print(f"input {p.name} sha256 {hashlib.sha256(p.read_bytes()).hexdigest()}")

    rows = [json.loads(l) for l in rows_path.open()]
    warm = [r for r in rows if r.get("run", 0) < 0 or r.get("warmupIncluded")]
    rows = [r for r in rows if not (r.get("run", 0) < 0 or r.get("warmupIncluded"))]
    tc = rows[0].get("toolchain", {})
    print(f"host {sorted({r['host'] for r in rows})} mlxCore {tc.get('mlxCoreVersion')} mlx-swift {tc.get('mlxSwiftPackage')} "
          f"git {str(tc.get('gitCommit'))[:10]} dirty {tc.get('gitDirtyFileCount')} schedule {sorted({r.get('schedule') for r in rows})} "
          f"K {sorted({r.get('speculationK') for r in rows})} genLength {sorted({r.get('genLength') for r in rows})}")
    invalid = [r for r in rows if not r.get("envValid")]
    print(f"rows: {len(rows)} non-warmup (+{len(warm)} warmup excluded); envValid(v2) false: {len(invalid)}; "
          f"envValidV1 false: {sum(1 for r in rows if not r.get('envValidV1', r.get('envValid')))}")
    for r in invalid:
        e = r.get("env", {})
        print(f"  EXCLUDED {r['arm']} {r['suite']}/{r['promptId']} run {r['run']}: free {e.get('freeMemoryMBBefore', -1):.0f} "
              f"avail {e.get('availableMemoryMBBefore') or -1:.0f} swap +{e.get('swapUsedMBAfter', 0) - e.get('swapUsedMBBefore', 0):.0f} thermal {e.get('thermalBefore')}")
    rows = [r for r in rows if r.get("envValid")]
    missing = set(ARMS.values()) - {r["arm"] for r in rows}
    if missing:
        print(f"WARNING: arms with no valid rows: {sorted(missing)}")

    by_key: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        by_key[(r["run"], r["suite"], r["promptId"])][r["arm"]] = r

    print("\nper-arm aggregates (valid rows):")
    print(f"  {'arm':22s} {'n':>4s} {'steps':>7s} {'fwd':>7s} {'tok':>7s} {'tok/unmask':>10s} {'1-tr frac':>9s} {'aggTPS':>7s} {'ms/fwd':>7s}")
    for short, arm in ARMS.items():
        rs = [r for r in rows if r["arm"] == arm]
        if not rs: continue
        st = sum(r["logicalSteps"] for r in rs); fw = sum(r["forwardsEvaluated"] for r in rs)
        tok = sum(r["tokensGenerated"] for r in rs); sec = sum(r["totalSeconds"] for r in rs)
        um = sum(r.get("unmaskingSteps", 0) for r in rs); one = sum(r.get("starvationSteps", 0) for r in rs)
        tr = sum(r.get("transfersTotal", 0) for r in rs)
        print(f"  {short:22s} {len(rs):4d} {st:7d} {fw:7d} {tok:7d} {tr / max(um, 1):10.3f} {one / max(um, 1):9.3f} {tok / sec:7.3f} {1000 * sec / fw:7.1f}")

    rng = random.Random(SEED)
    print(f"\npaired logical-step deltas vs none (mean, bootstrap 95% CI, {B_RESAMPLES} resamples, seed {SEED}):")
    deltas: dict[str, list[float]] = {}
    ci: dict[str, tuple[float, float, float]] = {}
    for short, arm in ARMS.items():
        if short == "none": continue
        d = [v[arm]["logicalSteps"] - v[BASE]["logicalSteps"] for v in by_key.values() if arm in v and BASE in v]
        if not d: continue
        deltas[short] = d
        ci[short] = boot_ci(d, rng)
        m, lo, hi = ci[short]
        print(f"  {short:6s} n={len(d):3d} mean {m:+.3f} [{lo:+.3f}, {hi:+.3f}]  more/fewer/equal {sum(x > 0 for x in d)}/{sum(x < 0 for x in d)}/{sum(x == 0 for x in d)}")

    # kivi4 − fp4 paired
    kf = [v[ARMS["kivi4"]]["logicalSteps"] - v[ARMS["fp4"]]["logicalSteps"]
          for v in by_key.values() if ARMS["kivi4"] in v and ARMS["fp4"] in v]
    kf_ci = boot_ci(kf, rng) if kf else None
    if kf_ci:
        print(f"  kivi4−fp4 n={len(kf)} mean {kf_ci[0]:+.3f} [{kf_ci[1]:+.3f}, {kf_ci[2]:+.3f}]")

    print("\npre-registered predictions:")
    def verdict(ok: bool | None) -> str:
        return "n/a (missing data)" if ok is None else ("HOLDS" if ok else "FAILS")
    p1 = None
    if all(k in ci for k in ("k4v8", "fp4", "k8v4")):
        p1 = (ci["k4v8"][1] > 0 and ci["fp4"][1] > 0
              and ci["k4v8"][0] > ci["k8v4"][0] and ci["fp4"][0] > ci["k8v4"][0])
    p2 = (ci["k8v4"][1] <= 0 <= ci["k8v4"][2]) if "k8v4" in ci else None
    p3 = (kf_ci[2] < 0) if kf_ci else None
    print(f"  (1) k4v8 & fp4 delta > 0 (CI excl. 0) and > k8v4's ............ {verdict(p1)}   → I-001 {'not refuted' if p1 else 'REFUTED' if p1 is False else '?'}")
    print(f"  (2) k8v4 CI includes 0 ............................................ {verdict(p2)}")
    print(f"  (3) kivi4 below fp4 (paired CI excl. 0) ........................... {verdict(p3)}   → I-002 {'not refuted' if p3 else 'REFUTED' if p3 is False else '?'}")

    # gsm8k accuracy
    gs = suites_dir / "gsm8k_100.json"
    answers = {p["id"]: str(p["answer"]).replace(",", "").strip() for p in json.loads(gs.read_text())["prompts"]}
    texts: dict[tuple, str] = {}
    for l in text_path.open():
        t = json.loads(l)
        texts[(t["arm"], int(t["run"]), t["promptId"])] = t.get("text", "")
    print("\ngsm8k accuracy (valid rows; last integer vs answer):")
    correct: dict[str, dict] = {}
    for short, arm in ARMS.items():
        keys = [(r["run"], r["promptId"]) for r in rows if r["arm"] == arm and r["suite"] == "gsm8k_100"]
        res = {k: last_int(texts.get((arm, k[0], k[1]), "")) == answers.get(k[1]) for k in keys}
        correct[short] = res
        if keys:
            print(f"  {short:6s} {sum(res.values())}/{len(res)} = {sum(res.values()) / len(res):.3f}")
    if correct.get("none"):
        for short in ARMS:
            if short == "none" or not correct.get(short): continue
            common = set(correct[short]) & set(correct["none"])
            b = sum(1 for k in common if correct["none"][k] and not correct[short][k])
            c = sum(1 for k in common if not correct["none"][k] and correct[short][k])
            print(f"  vs none: {short:6s} lost {b}, gained {c} (paired discordant, n={len(common)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
