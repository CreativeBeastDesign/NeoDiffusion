#!/bin/zsh
# run-studio-overnight-e007-e008.sh — E-007 (τ_mask sweep 0.5/0.7/0.9) then E-008 (context length 0/1k/2k/3.5k),
# 5 KV arms, served 4-bit artefact, on the Mac Studio. Pre-registrations: paper/experiments/E-00{7,8}.md.
# Run inside screen:
#
#     screen -dmS night zsh -c 'export PATH=$HOME/.venvs/neodiffusion-mlx/bin:$PATH; caffeinate -ims Tools/run-studio-overnight-e007-e008.sh'
#
# PHASES="e007 e008" by default. Rough wall-clock `[Inferred from the 2026-10-19 smoke test + E-006]`:
# e007 ~1.5 h, e008 ~2 h. Outputs: scratch/<date>/studio-night/.

set -u
REPO=${0:A:h:h}
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/$(date +%Y-%m-%d)/studio-night
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30
PHASES=${PHASES:-"e007 e008"}
M4=${M4:-$REPO/models/llada2-1-mini-4bit}
TOK=${TOK:-$REPO/models/llada2-1-mini}
ARMS=cache-clean-commit-k1,cache-k8v4-clean-k1,cache-k4v8-clean-k1,cache-fp4-clean-k1,cache-kivi4-clean-k1
BENCH=$REPO/.build/release/diffusion-bench

say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
has() { [[ " $PHASES " == *" $1 "* ]]; }
freemb() { print $(( $(vm_stat | awk '/Pages free/ {gsub(/\./,"",$3); print $3}') * 16384 / 1048576 )); }

cd $REPO
say "=== overnight start | host $(sysctl -n hw.model) | git $(git rev-parse --short HEAD) | phases: $PHASES"
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
[[ $DIRTY == 0 ]] || { say "REFUSING: $DIRTY tracked files modified — runs must come from a committed tree (P-5)."; exit 1; }
for e in E-007 E-008; do
  grep -q '^registered_commit:' paper/experiments/$e.md || { say "REFUSING: $e is not frozen (no registered_commit)"; exit 1; }
  say "registered: paper/experiments/$e.md @ $(git log -1 --format=%h -- paper/experiments/$e.md)"
done
say "registered: paper/scripts/group_deltas.py @ $(git log -1 --format=%h -- paper/scripts/group_deltas.py)"
[[ -f $M4/model.safetensors ]] || { say "missing $M4"; exit 1; }

say "build: release diffusion-bench"
MIN_AVAIL_MB=4000 $GATE night-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 \
  || { say "BUILD FAILED — $OUT/build-$STAMP.log"; exit 1; }

# bench <label> <suites> [extra flags...]
bench() {
  local L=$1 S=$2; shift 2
  local ROWS=$OUT/$L-rows-$STAMP.jsonl TEXT=$OUT/$L-text-$STAMP.jsonl
  say "$L: suites $S $* | free $(freemb) MB"
  MIN_AVAIL_MB=20000 $GATE night-$L $BENCH llada --model $M4 --tokenizer $TOK --arms $ARMS \
    --speculation-k 1 --interleave --suites $S --runs 1 --gen-length 128 --warmup-once \
    --json $ROWS --dump-text $TEXT "$@" > $OUT/$L-bench-$STAMP.log 2>&1
  say "$L: rc=$? rows $(wc -l < $ROWS 2>/dev/null | xargs) text $(wc -l < $TEXT 2>/dev/null | xargs)"
}

if has e007; then
  for T in 0.5 0.7 0.9; do bench e007-tau$T gsm8k_tau200,chat,reasoning,code --threshold-mask $T; done
  R=($OUT/e007-tau{0.5,0.7,0.9}-rows-$STAMP.jsonl)
  { echo "### E-007 PRIMARY (gsm8k_tau200)"; python3 paper/scripts/group_deltas.py --by tau --suite gsm8k_tau200 $R
    echo; echo "### E-007 SECONDARY (all 212)"; python3 paper/scripts/group_deltas.py --by tau $R
  } > $OUT/e007-REPORT-$STAMP.txt 2>&1
  { echo "## E-007"; grep -E "envValid=false|\([1-4]\)" $OUT/e007-REPORT-$STAMP.txt; } | tee -a $LOG
fi

if has e008; then
  bench e008 gsm8k_ctx0,gsm8k_ctx1000,gsm8k_ctx2000,gsm8k_ctx3500
  ROWS=$OUT/e008-rows-$STAMP.jsonl TEXT=$OUT/e008-text-$STAMP.jsonl
  { echo "### E-008 PRIMARY"; python3 paper/scripts/group_deltas.py --by ctx $ROWS
    echo; echo "### E-008 exploratory accuracy"; python3 paper/scripts/gsm8k_accuracy.py $TEXT
  } > $OUT/e008-REPORT-$STAMP.txt 2>&1
  { echo "## E-008"; grep -E "envValid=false|\([1-4]\)" $OUT/e008-REPORT-$STAMP.txt; } | tee -a $LOG
fi
say "=== done. Reports in $OUT"
