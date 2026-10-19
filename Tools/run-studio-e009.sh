#!/bin/zsh
# run-studio-e009.sh — E-009: KV-arm erosion on LLaDA2.1-flash (4-bit), 5 arms, 212 prompts, gen 128.
# Pre-registration: paper/experiments/E-009.md. Starts with a one-prompt load/compatibility check.
#
#     screen -dmS e009 zsh -c 'export PATH=$HOME/.venvs/neodiffusion-mlx/bin:$PATH; caffeinate -ims Tools/run-studio-e009.sh'
#
# AFTER=<file> CUTOFF_HOUR=5: wait until <file> exists, then run only if the local hour is < CUTOFF_HOUR.
# Outputs: scratch/<date>/studio-e009/.

set -u
REPO=${0:A:h:h}
cd $REPO
if [[ -n ${AFTER:-} ]]; then
  until [[ -f $AFTER ]]; do sleep 60; done
  H=$(date +%H); H=${H#0}
  if (( H >= ${CUTOFF_HOUR:-5} && H < 18 )); then
    mkdir -p scratch/$(date +%Y-%m-%d)/studio-e009
    print -r -- "$(date '+%F %T') E-009 NOT started: predecessor finished at hour $H (cutoff ${CUTOFF_HOUR:-5})" | tee scratch/$(date +%Y-%m-%d)/studio-e009/NOT-STARTED.txt
    exit 0
  fi
fi
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/$(date +%Y-%m-%d)/studio-e009
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30
MF=${MF:-$REPO/models/llada2.1-flash-4bit}
ARMS=cache-clean-commit-k1,cache-k8v4-clean-k1,cache-k4v8-clean-k1,cache-fp4-clean-k1,cache-kivi4-clean-k1
BENCH=$REPO/.build/release/diffusion-bench
say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
freemb() { print $(( $(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}') * 16384 / 1048576 )); }

say "=== E-009 start | host $(sysctl -n hw.model) | git $(git rev-parse --short HEAD) | free $(freemb) MB"
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
[[ $DIRTY == 0 ]] || { say "REFUSING: $DIRTY tracked files modified (P-5)."; exit 1; }
grep -q '^registered_commit:' paper/experiments/E-009.md || { say "REFUSING: E-009 is not frozen"; exit 1; }
say "registered: paper/experiments/E-009.md @ $(git log -1 --format=%h -- paper/experiments/E-009.md)"
[[ -f $MF/model.safetensors ]] || { say "missing $MF"; exit 1; }

MIN_AVAIL_MB=4000 $GATE e009-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 \
  || { say "BUILD FAILED"; exit 1; }

say "compat check: one prompt, uncompressed cache"
MIN_AVAIL_MB=70000 $GATE e009-compat $BENCH llada --model $MF --tokenizer $MF --arms cache-clean-commit-k1 \
  --speculation-k 1 --prompt "What is 17 * 23? Answer with the number only." --runs 1 --gen-length 32 \
  --json $OUT/compat-rows-$STAMP.jsonl --dump-text $OUT/compat-text-$STAMP.jsonl > $OUT/compat-$STAMP.log 2>&1
say "compat: rc=$? text: $(python3 -c "import json,sys;print([json.loads(l)['text'] for l in open(sys.argv[1])])" $OUT/compat-text-$STAMP.jsonl 2>&1 | head -c 300)"
grep -q 391 $OUT/compat-text-$STAMP.jsonl 2>/dev/null || { say "ABORT: compat check failed (no '391') — E-009 not run; see $OUT/compat-$STAMP.log"; exit 1; }

ROWS=$OUT/e009-rows-$STAMP.jsonl TEXT=$OUT/e009-text-$STAMP.jsonl
say "e009: suites gsm8k_flash200,chat,reasoning,code | free $(freemb) MB"
MIN_AVAIL_MB=70000 $GATE e009 $BENCH llada --model $MF --tokenizer $MF --arms $ARMS \
  --speculation-k 1 --interleave --suites gsm8k_flash200,chat,reasoning,code --runs 1 --gen-length 128 --warmup-once \
  --json $ROWS --dump-text $TEXT > $OUT/e009-bench-$STAMP.log 2>&1
say "e009: rc=$? rows $(wc -l < $ROWS 2>/dev/null | xargs) text $(wc -l < $TEXT 2>/dev/null | xargs)"
{ echo "### E-009 PRIMARY (gsm8k_flash200)"; python3 paper/scripts/e009_analysis.py $ROWS --suite gsm8k_flash200
  echo; echo "### E-009 SECONDARY (all 212)"; python3 paper/scripts/e009_analysis.py $ROWS
} > $OUT/e009-REPORT-$STAMP.txt 2>&1
grep -E "envValid=false|\([1-3]\)" $OUT/e009-REPORT-$STAMP.txt | tee -a $LOG
say "=== done. Report: $OUT/e009-REPORT-$STAMP.txt"
