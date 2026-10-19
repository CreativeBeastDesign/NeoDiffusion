#!/bin/zsh
# run-studio-e005.sh — E-005: E-002 re-run at gen 768 (GSM8K quality, 4-bit, 5 KV arms) on the Mac Studio.
# Pre-registration: paper/experiments/E-005.md. Same bench invocation as run-studio-batch.sh's e002 phase,
# only the generation length differs. Run inside screen/tmux:
#
#     screen -dmS e005 zsh -c 'export PATH=$HOME/.venvs/neodiffusion-mlx/bin:$PATH; caffeinate -ims Tools/run-studio-e005.sh'
#
# Outputs: scratch/<date>/studio-e005/{e005-rows,e005-text,e005-REPORT,run}-<stamp>.*

set -u
REPO=${0:A:h:h}
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/$(date +%Y-%m-%d)/studio-e005
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30
M4=${M4:-$REPO/models/llada2-1-mini-4bit}
TOK=${TOK:-$REPO/models/llada2-1-mini}
GL=768
ARMS=cache-clean-commit-k1,cache-k8v4-clean-k1,cache-k4v8-clean-k1,cache-fp4-clean-k1,cache-kivi4-clean-k1
BENCH=$REPO/.build/release/diffusion-bench

say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }

cd $REPO
say "=== E-005 start | host $(sysctl -n hw.model) | git $(git rev-parse --short HEAD) | gen $GL"
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
[[ $DIRTY == 0 ]] || { say "REFUSING: $DIRTY tracked files modified — runs must come from a committed tree (P-5)."; exit 1; }
grep -q '^registered_commit:' paper/experiments/E-005.md || { say "REFUSING: E-005 is not frozen (no registered_commit)"; exit 1; }
say "registered: paper/experiments/E-005.md @ $(git log -1 --format=%h -- paper/experiments/E-005.md)"
say "memory: free $(( $(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}') * 16384 / 1048576 )) MB (envValid floor on this host: 12288 MB)"

say "build: release diffusion-bench"
MIN_AVAIL_MB=4000 $GATE e005-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 \
  || { say "BUILD FAILED — $OUT/build-$STAMP.log"; exit 1; }

ROWS=$OUT/e005-rows-$STAMP.jsonl TEXT=$OUT/e005-text-$STAMP.jsonl
say "e005: model ${M4:t} suites gsm8k_100 gen $GL"
MIN_AVAIL_MB=20000 $GATE e005 $BENCH llada --model $M4 --tokenizer $TOK --arms $ARMS \
  --speculation-k 1 --interleave --suites gsm8k_100 --runs 1 --gen-length $GL --warmup-once \
  --json $ROWS --dump-text $TEXT > $OUT/e005-bench-$STAMP.log 2>&1
say "e005: rc=$? rows $(wc -l < $ROWS 2>/dev/null | xargs) text $(wc -l < $TEXT 2>/dev/null | xargs)"
{ python3 paper/scripts/truncation.py $TEXT $GL --suite gsm8k
  python3 paper/scripts/e001_analysis.py $ROWS $TEXT; } > $OUT/e005-REPORT-$STAMP.txt 2>&1
grep -E "hit-limit|rows:|/100 =|lost .* gained" $OUT/e005-REPORT-$STAMP.txt | tee -a $LOG
say "=== done. Report: $OUT/e005-REPORT-$STAMP.txt"
