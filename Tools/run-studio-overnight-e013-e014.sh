#!/bin/zsh
# run-studio-overnight-e013-e014.sh — E-014 (batched-prefill chunk sweep, ~20 min) then E-013
# (8-bit vs 4-bit weights, ABAB, ~80 min). Pre-registrations: paper/experiments/E-01{3,4}.md.
#
#     screen -dmS night2 zsh -c 'export PATH=$HOME/.venvs/neodiffusion-mlx/bin:$PATH; caffeinate -ims Tools/run-studio-overnight-e013-e014.sh'
set -u
REPO=${0:A:h:h}
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/$(date +%Y-%m-%d)/studio-night2
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30
TOK=${TOK:-$REPO/models/llada2-1-mini}
BENCH=$REPO/.build/release/diffusion-bench
say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
freemb() { print $(( $(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}') * 16384 / 1048576 )); }
cd $REPO
say "=== overnight-2 start | host $(sysctl -n hw.model) | git $(git rev-parse --short HEAD)"
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
[[ $DIRTY == 0 ]] || { say "REFUSING: $DIRTY tracked files modified (P-5)."; exit 1; }
for e in E-013 E-014; do
  grep -q '^registered_commit:' paper/experiments/$e.md || { say "REFUSING: $e is not frozen"; exit 1; }
  say "registered: paper/experiments/$e.md @ $(git log -1 --format=%h -- paper/experiments/$e.md)"
done
for m in llada2-1-mini-4bit llada2-1-mini-8bit; do [[ -f models/$m/model.safetensors ]] || { say "missing models/$m"; exit 1; }; done
MIN_AVAIL_MB=4000 $GATE n2-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 || { say "BUILD FAILED"; exit 1; }

# E-014
for C in 1 4 8 16 32 64 128; do
  R=$OUT/e014-c$C-rows-$STAMP.jsonl
  say "e014-c$C | free $(freemb) MB"
  MIN_AVAIL_MB=20000 $GATE n2-e014-c$C $BENCH llada --model models/llada2-1-mini-4bit --tokenizer $TOK --arms cache-clean-commit-k1 \
    --speculation-k 1 --suites gsm8k_bp3000 --runs 1 --gen-length 32 --warmup-once --batched-prefill $C \
    --json $R > $OUT/e014-c$C-bench-$STAMP.log 2>&1
  say "e014-c$C: rc=$? rows $(wc -l < $R | xargs)"
done
python3 paper/scripts/e014_analysis.py $OUT/e014-c{1,4,8,16,32,64,128}-rows-$STAMP.jsonl > $OUT/e014-REPORT-$STAMP.txt 2>&1
grep -E "envValid=false|fastest|\(C1\)" $OUT/e014-REPORT-$STAMP.txt | tee -a $LOG

# E-013 (ABAB)
i=0
for W in 4 8 4 8; do
  i=$((i + 1)); RUN=r$(( (i + 1) / 2 ))
  R=$OUT/e013-${W}bit-$RUN-rows-$STAMP.jsonl T=$OUT/e013-${W}bit-$RUN-text-$STAMP.jsonl
  say "e013-${W}bit-$RUN | free $(freemb) MB"
  MIN_AVAIL_MB=30000 $GATE n2-e013-$W-$RUN $BENCH llada --model models/llada2-1-mini-${W}bit --tokenizer $TOK --arms cache-clean-commit-k1 \
    --speculation-k 1 --interleave --suites gsm8k_tr0,gsm8k_tr2000,chat,reasoning,code --runs 1 --gen-length 768 --warmup-once \
    --json $R --dump-text $T > $OUT/e013-${W}bit-$RUN-bench-$STAMP.log 2>&1
  say "e013-${W}bit-$RUN: rc=$? rows $(wc -l < $R | xargs)"
done
python3 paper/scripts/e013_analysis.py \
  $OUT/e013-4bit-r1-rows-$STAMP.jsonl $OUT/e013-4bit-r1-text-$STAMP.jsonl $OUT/e013-8bit-r1-rows-$STAMP.jsonl $OUT/e013-8bit-r1-text-$STAMP.jsonl \
  $OUT/e013-4bit-r2-rows-$STAMP.jsonl $OUT/e013-4bit-r2-text-$STAMP.jsonl $OUT/e013-8bit-r2-rows-$STAMP.jsonl $OUT/e013-8bit-r2-text-$STAMP.jsonl 768 \
  > $OUT/e013-REPORT-$STAMP.txt 2>&1
grep -E "envValid=false|\(W[1-3]\)" $OUT/e013-REPORT-$STAMP.txt | tee -a $LOG
say "=== done. Reports in $OUT"
