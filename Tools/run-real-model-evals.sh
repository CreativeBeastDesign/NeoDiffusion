#!/bin/zsh
# run-real-model-evals.sh — real-model (4-bit, ~9.6 GB peak) checks that cannot run next to the Claude
# desktop app on the 16 GB dev M1. Run from a plain terminal with the Claude app QUIT:
#
#     cd ~/Documents/Swift/NeoDiffusion && caffeinate -i Tools/run-real-model-evals.sh
#
# Steps (each through Tools/compute-gate.sh, one at a time; model steps need MODEL_MIN_AVAIL_MB and are
# labelled invalid if swap grows > 256 MB):
#   1. Exp-5B hidden real-model test (B1 bit-exactness + turn-2 speed, B2 k8v4 exactness) for impl-x, then impl-y.
#   2. B-1 pre-registered bench: K=1 vs K=4 on the current default engine (Plans/b1-speculation-k-m1-prereg.md).
# Expected wall-clock: ~25–35 min. Everything is logged under scratch/2026-10-18/real/; a summary is
# printed at the end and written to scratch/2026-10-18/real/SUMMARY.txt. Safe to re-run (appends).
# Written 2026-10-18 by the Claude session that ran Exp-5B; see Plans/exp5b-opus-vs-fable-logbook.md.

set -u
REPO=${0:A:h:h}
OUT=$REPO/scratch/2026-10-18/real
mkdir -p $OUT
SUM=$OUT/SUMMARY.txt
GATE=$REPO/Tools/compute-gate.sh
# Memory floors (2026-10-18 fix): the first version applied 10.5 GB to every step, which this host never
# reports even with the Claude app quit (vm_stat free+inactive+speculative excludes reclaimable file cache),
# so the gate waited forever. Builds need no floor beyond the default; model steps use MODEL_MIN_AVAIL_MB,
# and every model step is additionally judged by swap growth (envValid rule: <= 256 MB).
MODEL_MIN_AVAIL_MB=${MODEL_MIN_AVAIL_MB:-6500}
export MIN_AVAIL_MB=2000
export RETRY_SECONDS=${RETRY_SECONDS:-30}
export NEODIFFUSION_LOOP_FIXTURE_DIR=$REPO/scratch/loop_fixtures
HIDDEN=$REPO/Plans/exp5b-artifacts/HiddenSessionCacheEvalTests.v3.swift.txt

say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $SUM; }

swap_mb() { sysctl -n vm.swapusage | awk '{gsub(/M/,"",$6); print int($6)}'; }
avail_mb() { vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3}
  END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s);printf "%d",(f+i+s)*ps/1048576}'; }

say "=== run-real-model-evals start | host $(sysctl -n hw.model) | git $(git -C $REPO rev-parse --short HEAD) | avail $(avail_mb) MB | $(sysctl -n vm.swapusage)"
if pgrep -xq Claude; then
  say "WARNING: the Claude desktop app is running — quit it first (model steps need ${MODEL_MIN_AVAIL_MB} MB available and no swap growth)."
fi
[[ -f $HIDDEN ]] || { say "missing $HIDDEN"; exit 1; }
[[ "$(shasum -a 256 $HIDDEN | cut -c1-16)" == "5978cdf3cc77d360" ]] || { say "hidden test hash mismatch — refusing"; exit 1; }

# ---- 1. Exp-5B real-model evaluation ------------------------------------------------------------
for pair in "impl-x:agent-ae8403ea4fa3150dc" "impl-y:agent-ac87f8dedec5e3cb7"; do
  L=${pair%%:*}; W=$REPO/.claude/worktrees/${pair#*:}
  [[ -d $W ]] || { say "$L: worktree $W missing — skipped"; continue; }
  cd $W
  [ -e scratch ] || ln -s $REPO/scratch scratch
  cp $HIDDEN Tests/DiffusionGenerationTests/HiddenSessionCacheEvalTests.swift
  say "$L: build (commit $(git rev-parse --short HEAD))"
  $GATE real-$L-build swift build --build-tests > $OUT/$L-build.log 2>&1 || say "$L build FAILED (see $OUT/$L-build.log)"
  $GATE real-$L-seed Tools/seed-metallib.sh >> $OUT/$L-build.log 2>&1
  say "$L: real-model hidden test (avail $(avail_mb) MB)"
  SW0=$(swap_mb)
  MIN_AVAIL_MB=$MODEL_MIN_AVAIL_MB NEODIFFUSION_SESSION_EVAL_REAL=1 NEODIFFUSION_SESSION_EVAL_LABEL=$L NEODIFFUSION_SESSION_EVAL_OUT=$OUT/exp5b_session_perf.jsonl \
    $GATE real-$L-eval swift test --skip-build --filter HiddenSessionCacheRealEvalTests > $OUT/$L-real.log 2>&1
  RC=$?
  SW1=$(swap_mb); VALID=$([ $((SW1 - SW0)) -le 256 ] && echo valid || echo "INVALID(swap +$((SW1 - SW0)) MB — paging, timings unusable)")
  say "$L: real-model rc=$RC | $(grep -E "Executed [0-9]+ test" $OUT/$L-real.log | tail -1 | xargs) | swap ${SW0}->${SW1} MB | env $VALID"
  grep -E "\[exp5b\]" $OUT/$L-real.log | tee -a $SUM
  rm -f Tests/DiffusionGenerationTests/HiddenSessionCacheEvalTests.swift
done

# ---- 2. B-1: K=1 vs K=4 on the M1 (pre-registered) ----------------------------------------------
cd $REPO
say "B-1: release build of diffusion-bench (commit $(git rev-parse --short HEAD), dirty=$(git status --porcelain --untracked-files=no | wc -l | xargs))"
$GATE real-b1-build swift build -c release --product diffusion-bench > $OUT/b1-build.log 2>&1 || say "B-1 build FAILED"
$GATE real-b1-seed Tools/seed-metallib.sh >> $OUT/b1-build.log 2>&1
say "B-1: bench (avail $(avail_mb) MB)"
MIN_AVAIL_MB=$MODEL_MIN_AVAIL_MB $GATE real-b1-bench ./.build/release/diffusion-bench llada --arms spk-1-clean,spk-4-clean --interleave \
  --suites chat,reasoning,code --runs 2 --gen-length 128 --warmup-once \
  --json $OUT/b1_spk_k1_vs_k4.jsonl --text-out $OUT/b1_spk_k1_vs_k4_text.jsonl > $OUT/b1-bench.log 2>&1
say "B-1: bench rc=$?"
python3 - $OUT/b1_spk_k1_vs_k4.jsonl <<'EOF' | tee -a $SUM
import json, sys, statistics
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1])]
rows = [r for r in rows if r.get('run', 0) >= 0 and not r.get('warmupIncluded')]
valid = [r for r in rows if r.get('envValid')]
print(f"B-1 rows: {len(rows)} non-warmup, {len(valid)} envValid; schedule={sorted(set(r.get('schedule') for r in rows))}")
agg = defaultdict(lambda: [0, 0.0, 0])
for r in valid:
    a = agg[r['arm']]; a[0] += r['tokensGenerated']; a[1] += r['totalSeconds']; a[2] += r['forwardsEvaluated']
for arm, (tok, sec, fw) in sorted(agg.items()):
    print(f"  {arm:14s} aggTPS={tok/sec:.3f} tokens={tok} sec={sec:.1f} forwards={fw} ms/fwd={1000*sec/fw:.1f}")
pair = defaultdict(dict)
for r in valid: pair[(r['run'], r['promptId'])][r['arm']] = r
ratios = [p['spk-1-clean']['tps'] / p['spk-4-clean']['tps'] for p in pair.values() if len(p) == 2]
same = sum(1 for p in pair.values() if len(p) == 2 and p['spk-1-clean']['textPrefix'] == p['spk-4-clean']['textPrefix'])
if ratios:
    print(f"  paired K1/K4 TPS ratio: median {statistics.median(ratios):.3f} min {min(ratios):.3f} max {max(ratios):.3f} (n={len(ratios)}); identical text prefixes {same}/{len(ratios)}")
EOF
say "=== done. Logs: $OUT"
