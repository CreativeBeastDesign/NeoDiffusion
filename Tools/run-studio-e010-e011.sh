#!/bin/zsh
# run-studio-e010-e011.sh — E-010 (mechanism: per-step traces at ~1.7k-token context, gen 128) then
# E-011 (paired GSM8K quality at ~1.7k-token context, gen 768). 4 arms, served 4-bit artefact.
# Pre-registrations: paper/experiments/E-01{0,1}.md. Run inside screen:
#
#     screen -dmS mq zsh -c 'export PATH=$HOME/.venvs/neodiffusion-mlx/bin:$PATH; caffeinate -ims Tools/run-studio-e010-e011.sh'
#
# PHASES="e010 e011" by default. Rough wall-clock `[Inferred from the 2026-10-20 smoke test]`: e010 ~45 min,
# e011 ~80 min. Outputs: scratch/<date>/studio-mq/.

set -u
REPO=${0:A:h:h}
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/$(date +%Y-%m-%d)/studio-mq
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30
PHASES=${PHASES:-"e010 e011"}
M4=${M4:-$REPO/models/llada2-1-mini-4bit}
TOK=${TOK:-$REPO/models/llada2-1-mini}
ARMS=cache-clean-commit-k1,cache-k8v4-clean-k1,cache-k4v8-clean-k1,cache-kivi4-clean-k1
BENCH=$REPO/.build/release/diffusion-bench
say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
has() { [[ " $PHASES " == *" $1 "* ]]; }
freemb() { print $(( $(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}') * 16384 / 1048576 )); }

cd $REPO
say "=== E-010/E-011 start | host $(sysctl -n hw.model) | git $(git rev-parse --short HEAD) | phases: $PHASES"
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
[[ $DIRTY == 0 ]] || { say "REFUSING: $DIRTY tracked files modified (P-5)."; exit 1; }
for e in E-010 E-011; do
  grep -q '^registered_commit:' paper/experiments/$e.md || { say "REFUSING: $e is not frozen"; exit 1; }
  say "registered: paper/experiments/$e.md @ $(git log -1 --format=%h -- paper/experiments/$e.md)"
done
MIN_AVAIL_MB=4000 $GATE mq-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 \
  || { say "BUILD FAILED"; exit 1; }

if has e010; then
  ROWS=$OUT/e010-rows-$STAMP.jsonl TEXT=$OUT/e010-text-$STAMP.jsonl TR=$OUT/e010-traces-$STAMP.jsonl
  say "e010: suite gsm8k_mech2000 gen 128 traces | free $(freemb) MB"
  MIN_AVAIL_MB=20000 $GATE mq-e010 $BENCH llada --model $M4 --tokenizer $TOK --arms $ARMS \
    --speculation-k 1 --interleave --suites gsm8k_mech2000 --runs 1 --gen-length 128 --warmup-once \
    --json $ROWS --dump-text $TEXT --dump-traces $TR > $OUT/e010-bench-$STAMP.log 2>&1
  say "e010: rc=$? rows $(wc -l < $ROWS | xargs) traces $(wc -l < $TR | xargs)"
  { echo "### E-010 PRIMARY"; python3 paper/scripts/e010_mechanism.py $TR $ROWS
    echo; echo "### E-010 SECONDARY (row-level step deltas; verdict lines below are E-009's, not E-010's)"
    python3 paper/scripts/e009_analysis.py $ROWS --suite gsm8k_mech2000
  } > $OUT/e010-REPORT-$STAMP.txt 2>&1
  grep -E "envValid=false|\(M[1-5]\)" $OUT/e010-REPORT-$STAMP.txt | tee -a $LOG
fi
if has e011; then
  ROWS=$OUT/e011-rows-$STAMP.jsonl TEXT=$OUT/e011-text-$STAMP.jsonl
  say "e011: suite gsm8k_qual2000 gen 768 | free $(freemb) MB"
  MIN_AVAIL_MB=20000 $GATE mq-e011 $BENCH llada --model $M4 --tokenizer $TOK --arms $ARMS \
    --speculation-k 1 --interleave --suites gsm8k_qual2000 --runs 1 --gen-length 768 --warmup-once \
    --json $ROWS --dump-text $TEXT > $OUT/e011-bench-$STAMP.log 2>&1
  say "e011: rc=$? rows $(wc -l < $ROWS | xargs) text $(wc -l < $TEXT | xargs)"
  { echo "### E-011 PRIMARY"; python3 paper/scripts/e011_quality.py $ROWS $TEXT 768
    echo; echo "### E-011 SECONDARY (row-level step deltas; verdict lines below are E-009's, not E-011's)"
    python3 paper/scripts/e009_analysis.py $ROWS --suite gsm8k_qual2000
  } > $OUT/e011-REPORT-$STAMP.txt 2>&1
  grep -E "envValid=false|\((G|Q1)\)|detectable loss" $OUT/e011-REPORT-$STAMP.txt | tee -a $LOG
fi
say "=== done. Reports in $OUT"
