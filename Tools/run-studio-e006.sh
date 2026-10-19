#!/bin/zsh
# run-studio-e006.sh — E-006: weight-precision control (4-bit / 8-bit / F16), 5 KV arms, 312 prompts each.
# Pre-registration: paper/experiments/E-006.md. One bench process per artefact, sequential, fixed order.
# Needs models/llada2-1-mini-8bit (convert_weights_streaming.py --bits 8 --group-size 64). Run inside screen:
#
#     screen -dmS e006 zsh -c 'export PATH=$HOME/.venvs/neodiffusion-mlx/bin:$PATH; caffeinate -ims Tools/run-studio-e006.sh'
#
# Outputs: scratch/<date>/studio-e006/{e006-<w>-rows,e006-<w>-text,e006-REPORT,run}-<stamp>.*

set -u
REPO=${0:A:h:h}
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/$(date +%Y-%m-%d)/studio-e006
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30
TOK=${TOK:-$REPO/models/llada2-1-mini}
ARMS=cache-clean-commit-k1,cache-k8v4-clean-k1,cache-k4v8-clean-k1,cache-fp4-clean-k1,cache-kivi4-clean-k1
SUITES=gsm8k_100,chat,reasoning,code,gsm8k_fresh200
BENCH=$REPO/.build/release/diffusion-bench
typeset -A MODEL=(4bit $REPO/models/llada2-1-mini-4bit 8bit $REPO/models/llada2-1-mini-8bit f16 $REPO/models/llada2-1-mini-f16)

say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }

cd $REPO
say "=== E-006 start | host $(sysctl -n hw.model) | git $(git rev-parse --short HEAD)"
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
[[ $DIRTY == 0 ]] || { say "REFUSING: $DIRTY tracked files modified — runs must come from a committed tree (P-5)."; exit 1; }
grep -q '^registered_commit:' paper/experiments/E-006.md || { say "REFUSING: E-006 is not frozen (no registered_commit)"; exit 1; }
for w in 4bit 8bit f16; do
  [[ -f $MODEL[$w]/model.safetensors ]] || { say "REFUSING: missing $MODEL[$w]/model.safetensors"; exit 1; }
done
say "registered: paper/experiments/E-006.md @ $(git log -1 --format=%h -- paper/experiments/E-006.md); e006_analysis.py @ $(git log -1 --format=%h -- paper/scripts/e006_analysis.py)"

say "build: release diffusion-bench"
MIN_AVAIL_MB=4000 $GATE e006-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 \
  || { say "BUILD FAILED — $OUT/build-$STAMP.log"; exit 1; }

for w in 4bit 8bit f16; do
  ROWS=$OUT/e006-$w-rows-$STAMP.jsonl TEXT=$OUT/e006-$w-text-$STAMP.jsonl
  say "e006-$w: model ${MODEL[$w]:t} free $(( $(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}') * 16384 / 1048576 )) MB"
  MIN_AVAIL_MB=20000 $GATE e006-$w $BENCH llada --model $MODEL[$w] --tokenizer $TOK --arms $ARMS \
    --speculation-k 1 --interleave --suites $SUITES --runs 1 --gen-length 128 --warmup-once \
    --json $ROWS --dump-text $TEXT > $OUT/e006-$w-bench-$STAMP.log 2>&1
  say "e006-$w: rc=$? rows $(wc -l < $ROWS 2>/dev/null | xargs) text $(wc -l < $TEXT 2>/dev/null | xargs)"
done

R=($OUT/e006-{4bit,8bit,f16}-rows-$STAMP.jsonl)
{ echo "### PRIMARY (fresh prompts)"; python3 paper/scripts/e006_analysis.py $R --suite gsm8k_fresh200
  echo; echo "### S2a original 112 (not blind)"; python3 paper/scripts/e006_analysis.py $R --not-suite gsm8k_fresh200
  echo; echo "### S2b all 312 (not blind)"; python3 paper/scripts/e006_analysis.py $R
} > $OUT/e006-REPORT-$STAMP.txt 2>&1
grep -E "^###|envValid=false|\(P[1-4]\)" $OUT/e006-REPORT-$STAMP.txt | tee -a $LOG
say "=== done. Report: $OUT/e006-REPORT-$STAMP.txt"
