#!/bin/zsh
# run-studio-e012.sh — E-012: batched prefill off / 16 / 128 blocks per forward, uncompressed KV,
# 300 GSM8K questions at 0 / ~1.7k / ~2.9k context, gen 768. Pre-registration: paper/experiments/E-012.md.
#
#     screen -dmS e012 zsh -c 'export PATH=$HOME/.venvs/neodiffusion-mlx/bin:$PATH; caffeinate -ims Tools/run-studio-e012.sh'
#
# Outputs: scratch/<date>/studio-e012/.
set -u
REPO=${0:A:h:h}
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/$(date +%Y-%m-%d)/studio-e012
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30
M4=${M4:-$REPO/models/llada2-1-mini-4bit}
TOK=${TOK:-$REPO/models/llada2-1-mini}
BENCH=$REPO/.build/release/diffusion-bench
SUITES=gsm8k_bp0,gsm8k_bp2000,gsm8k_bp3000
say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
freemb() { print $(( $(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}') * 16384 / 1048576 )); }

cd $REPO
say "=== E-012 start | host $(sysctl -n hw.model) | git $(git rev-parse --short HEAD)"
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
[[ $DIRTY == 0 ]] || { say "REFUSING: $DIRTY tracked files modified (P-5)."; exit 1; }
grep -q '^registered_commit:' paper/experiments/E-012.md || { say "REFUSING: E-012 is not frozen"; exit 1; }
say "registered: paper/experiments/E-012.md @ $(git log -1 --format=%h -- paper/experiments/E-012.md)"
MIN_AVAIL_MB=4000 $GATE e012-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 \
  || { say "BUILD FAILED"; exit 1; }
for C in 1 16 128; do
  ROWS=$OUT/e012-c$C-rows-$STAMP.jsonl TEXT=$OUT/e012-c$C-text-$STAMP.jsonl
  say "e012-c$C: --batched-prefill $C | free $(freemb) MB"
  MIN_AVAIL_MB=20000 $GATE e012-c$C $BENCH llada --model $M4 --tokenizer $TOK --arms cache-clean-commit-k1 \
    --speculation-k 1 --interleave --suites $SUITES --runs 1 --gen-length 768 --warmup-once \
    --batched-prefill $C --json $ROWS --dump-text $TEXT > $OUT/e012-c$C-bench-$STAMP.log 2>&1
  say "e012-c$C: rc=$? rows $(wc -l < $ROWS | xargs) text $(wc -l < $TEXT | xargs)"
done
python3 paper/scripts/e012_analysis.py \
  $OUT/e012-c1-rows-$STAMP.jsonl $OUT/e012-c1-text-$STAMP.jsonl \
  $OUT/e012-c16-rows-$STAMP.jsonl $OUT/e012-c16-text-$STAMP.jsonl \
  $OUT/e012-c128-rows-$STAMP.jsonl $OUT/e012-c128-text-$STAMP.jsonl 768 > $OUT/e012-REPORT-$STAMP.txt 2>&1
grep -E "envValid=false|\((S[1-3]|T1|G|Q1)\)" $OUT/e012-REPORT-$STAMP.txt | tee -a $LOG
say "=== done. Report: $OUT/e012-REPORT-$STAMP.txt"
