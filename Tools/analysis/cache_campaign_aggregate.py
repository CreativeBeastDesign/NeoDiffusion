#!/usr/bin/env python3
"""Aggregate the 2026-10 cache-optimisation campaign JSONL files (stdlib only, CPU only).

Usage:  python3 Tools/analysis/cache_campaign_aggregate.py [files...]
Default inputs (missing files are skipped):
  scratch/cache_campaign1_k4.jsonl scratch/cache_campaign1b_k4.jsonl
  scratch/cache_campaign2_k1.jsonl scratch/track1_kv_starvation_k1.jsonl
  scratch/cache_k8v4_clean_k1.jsonl

Per file: sha256; per arm (rows with run<0 or warmupIncluded excluded):
  n, envValid count, sum tokensGenerated, sum totalSeconds,
  aggregate TPS (= sum tokens / sum seconds), mean-of-row tps,
  sum forwardsEvaluated, ms/forward (= 1000*sum seconds / sum forwards),
  sum logicalSteps, distinct host / processId / toolchain,
  and arm execution order (blocked = each arm one contiguous run of rows).
Deltas vs the first arm (or --baseline ARM) are printed for both TPS definitions.
Also printed: the same table restricted to envValid rows.
"""
import hashlib, json, os, sys

DEFAULTS = ["scratch/cache_campaign1_k4.jsonl", "scratch/cache_campaign1b_k4.jsonl",
            "scratch/cache_campaign2_k1.jsonl", "scratch/track1_kv_starvation_k1.jsonl",
            "scratch/cache_k8v4_clean_k1.jsonl"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                print("  WARN: unparsable line skipped")
    return rows


def tc(r):
    t = r.get("toolchain") or {}
    if isinstance(t, dict):
        return "mlxSwift=%s mlxCore=%s swift=%s" % (t.get("mlxSwiftPackage"), t.get("mlxCoreVersion"),
                                                  t.get("swiftVersion", t.get("swift")))
    return str(t)


def order_kind(rows):
    seq = [r.get("arm") for r in rows]
    runs = []
    for a in seq:
        if not runs or runs[-1] != a:
            runs.append(a)
    blocked = len(runs) == len(set(runs))
    return ("BLOCKED (sequential, one contiguous run per arm)" if blocked
            else "INTERLEAVED (%d arm-runs for %d arms)" % (len(runs), len(set(runs)))), runs


def table(rows, label):
    arms = []
    by = {}
    for r in rows:
        a = r.get("arm")
        if a not in by:
            by[a] = []
            arms.append(a)
        by[a].append(r)
    print("  -- %s --" % label)
    print("  %-26s %3s %4s %7s %9s %8s %8s %7s %8s %7s" % (
        "arm", "n", "envV", "tokens", "seconds", "aggTPS", "meanTPS", "fwds", "ms/fwd", "logStp"))
    stats = {}
    for a in arms:
        rs = by[a]
        tok = sum(r.get("tokensGenerated", 0) for r in rs)
        sec = sum(r.get("totalSeconds", 0.0) for r in rs)
        fw = sum(r.get("forwardsEvaluated", 0) for r in rs)
        ls = sum(r.get("logicalSteps", 0) for r in rs)
        ev = sum(1 for r in rs if r.get("envValid"))
        agg = tok / sec if sec else float("nan")
        mean = sum(r.get("tps", 0.0) for r in rs) / len(rs)
        msf = 1000 * sec / fw if fw else float("nan")
        stats[a] = (agg, mean, msf, fw)
        print("  %-26s %3d %4d %7d %9.2f %8.3f %8.3f %7d %8.2f %7d" % (a, len(rs), ev, tok, sec, agg, mean, fw, msf, ls))
    return arms, stats


def deltas(arms, stats, base):
    if base not in stats:
        return
    b = stats[base]
    print("  deltas vs %s:  arm: aggTPS%% | meanTPS%% | ms/fwd%% | forwards%%" % base)
    for a in arms:
        s = stats[a]
        print("    %-26s %+7.1f%% | %+7.1f%% | %+6.1f%% | %+6.1f%%" % (
            a, 100 * (s[0] / b[0] - 1), 100 * (s[1] / b[1] - 1), 100 * (s[2] / b[2] - 1), 100 * (s[3] / b[3] - 1)))


def main(argv):
    base = None
    if "--baseline" in argv:
        i = argv.index("--baseline"); base = argv[i + 1]; del argv[i:i + 2]
    files = argv or DEFAULTS
    for p in files:
        print("=" * 100)
        print(p)
        if not os.path.exists(p):
            print("  (missing, skipped)")
            continue
        print("  sha256 %s" % sha256(p))
        allrows = load(p)
        rows = [r for r in allrows if not (r.get("run", 0) < 0 or r.get("warmupIncluded"))]
        print("  rows total %d, kept %d (excluded run<0 or warmupIncluded: %d)" % (len(allrows), len(rows), len(allrows) - len(rows)))
        print("  hosts: %s" % sorted({str(r.get("host")) for r in rows}))
        print("  processIds: %s" % sorted({str(r.get("processId")) for r in rows}))
        print("  toolchains: %s" % sorted({tc(r) for r in rows}))
        kind, runs = order_kind(rows)
        print("  arm execution order: %s" % kind)
        print("    %s" % " -> ".join(str(x) for x in runs))
        arms, stats = table(rows, "all kept rows")
        deltas(arms, stats, base or arms[0])
        vrows = [r for r in rows if r.get("envValid")]
        if vrows and len(vrows) != len(rows):
            arms2, stats2 = table(vrows, "envValid rows only")
            deltas(arms2, stats2, base if base in stats2 else arms2[0])


if __name__ == "__main__":
    main(sys.argv[1:])
