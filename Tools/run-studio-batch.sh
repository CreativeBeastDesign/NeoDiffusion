#!/bin/zsh
# run-studio-batch.sh — E-002 (GSM8K quality), E-003 (unquantized-weights control), E-004 (confidence
# traces) in ONE sequential run on the Mac Studio M2 Ultra. Pre-registrations: paper/experiments/E-00{2,3,4}.md.
# Works over SSH (no GUI needed). Run inside tmux/screen so a dropped connection does not kill it:
#
#     cd ~/Documents/Swift/NeoDiffusion && git pull && tmux new -s batch 'caffeinate -ims Tools/run-studio-batch.sh'
#
# Phases (each skippable: PHASES="convert e002 e003a e003b e004" by default; set e.g. PHASES="e002"):
#   convert  build models/llada2-1-mini-f16 from the BF16 source if missing (~33 GB, needs python3 + mlx)
#   e002     GSM8K-100, gen 384, 5 KV arms, 4-bit                       — quality
#   e003a    E-001 replica (112 prompts, gen 128), 5 arms, 4-bit        — same-host baseline for H5
#   e003b    same, unquantized F16 artefact                              — weights control
#   e004     gsm8k_20+chat+reasoning+code, gen 128, 5 arms, --dump-traces — mechanism
# Rough wall-clock on the Ultra `[Speculative — the Studio has not run these arms]`: e002 1–2 h,
# e003a ~30 min, e003b ~45 min, e004 ~20 min, convert 15–30 min. Outputs: scratch/<date>/studio-batch/.
# Every phase writes its own JSONL + REPORT; a final SUMMARY.txt collects the verdict lines.

set -u
REPO=${0:A:h:h}
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/$(date +%Y-%m-%d)/studio-batch
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
SUM=$OUT/SUMMARY-$STAMP.txt
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30
PHASES=${PHASES:-"convert e002 e003a e003b e004"}
M4=${M4:-$REPO/models/llada2-1-mini-4bit}
MF16=${MF16:-$REPO/models/llada2-1-mini-f16}
BF16SRC=${BF16SRC:-$REPO/models/llada2-1-mini}
TOK=${TOK:-$REPO/models/llada2-1-mini}
ARMS=cache-clean-commit-k1,cache-k8v4-clean-k1,cache-k4v8-clean-k1,cache-fp4-clean-k1,cache-kivi4-clean-k1
BENCH=$REPO/.build/release/diffusion-bench

say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
has() { [[ " $PHASES " == *" $1 "* ]]; }

cd $REPO
say "=== studio batch start | host $(sysctl -n hw.model) $(( $(sysctl -n hw.memsize) / 1073741824 )) GB | git $(git rev-parse --short HEAD) | phases: $PHASES"
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
[[ $DIRTY == 0 ]] || { say "REFUSING: $DIRTY tracked files modified — runs must come from a committed tree (P-5)."; exit 1; }
for f in paper/experiments/E-002.md paper/experiments/E-003.md paper/experiments/E-004.md \
         paper/scripts/e001_analysis.py paper/scripts/truncation.py paper/scripts/e004_traces.py; do
  say "registered: $f @ $(git log -1 --format=%h -- $f)"
done
[[ -f $M4/model.safetensors ]] || { say "missing 4-bit artefact $M4"; exit 1; }

say "build: release diffusion-bench"
MIN_AVAIL_MB=4000 $GATE batch-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 \
  || { say "BUILD FAILED — $OUT/build-$STAMP.log"; exit 1; }
MIN_AVAIL_MB=4000 $GATE batch-seed Tools/seed-metallib.sh >> $OUT/build-$STAMP.log 2>&1 \
  || say "WARNING: seed-metallib failed (needs an Xcode DerivedData build of the workspace) — the bench may crash at GPU init"

if has convert && [[ ! -f $MF16/model.safetensors ]] && (has e003b); then
  say "convert: $BF16SRC -> $MF16 (unquantized F16, ~33 GB)"
  MIN_AVAIL_MB=40000 $GATE batch-convert python3 Tools/convert_weights_streaming.py --src-dir $BF16SRC --dest-dir $MF16 --no-quantize \
    > $OUT/convert-$STAMP.log 2>&1 || { say "CONVERT FAILED — $OUT/convert-$STAMP.log (skipping e003b)"; PHASES=${PHASES/e003b/}; }
  cp $BF16SRC/tokenizer* $MF16/ 2>/dev/null || true
fi

# run_bench <phase> <model> <suites> <genLength> [extra flags...]
run_bench() {
  local P=$1 MODEL=$2 SUITES=$3 GL=$4; shift 4
  local ROWS=$OUT/$P-rows-$STAMP.jsonl TEXT=$OUT/$P-text-$STAMP.jsonl
  say "$P: model ${MODEL:t} suites $SUITES gen $GL $*"
  MIN_AVAIL_MB=20000 $GATE batch-$P $BENCH llada --model $MODEL --tokenizer $TOK --arms $ARMS \
    --speculation-k 1 --interleave --suites $SUITES --runs 1 --gen-length $GL --warmup-once \
    --json $ROWS --dump-text $TEXT "$@" > $OUT/$P-bench-$STAMP.log 2>&1
  say "$P: rc=$? rows $(wc -l < $ROWS 2>/dev/null | xargs) text $(wc -l < $TEXT 2>/dev/null | xargs)"
}

if has e002; then
  run_bench e002 $M4 gsm8k_100 384
  { python3 paper/scripts/truncation.py $OUT/e002-text-$STAMP.jsonl 384 --suite gsm8k
    python3 paper/scripts/e001_analysis.py $OUT/e002-rows-$STAMP.jsonl $OUT/e002-text-$STAMP.jsonl; } > $OUT/e002-REPORT-$STAMP.txt 2>&1
  { echo "## E-002"; grep -E "hit-limit|/100 =|lost .* gained" $OUT/e002-REPORT-$STAMP.txt; } | tee -a $SUM
fi
for P in e003a e003b; do
  has $P || continue
  MODEL=$M4; [[ $P == e003b ]] && MODEL=$MF16
  [[ -f $MODEL/model.safetensors ]] || { say "$P: missing $MODEL — skipped"; continue; }
  run_bench $P $MODEL gsm8k_100,chat,reasoning,code 128
  python3 paper/scripts/e001_analysis.py $OUT/$P-rows-$STAMP.jsonl $OUT/$P-text-$STAMP.jsonl > $OUT/$P-REPORT-$STAMP.txt 2>&1
  { echo "## ${P:u} (${MODEL:t})"; grep -E "rows:|step-cap|mean .*\[|\(1\)|\(2\)|\(3\)" $OUT/$P-REPORT-$STAMP.txt; } | tee -a $SUM
done
if has e004; then
  run_bench e004 $M4 gsm8k_20,chat,reasoning,code 128 --dump-traces $OUT/e004-traces-$STAMP.jsonl
  python3 paper/scripts/e004_traces.py $OUT/e004-traces-$STAMP.jsonl $OUT/e004-text-$STAMP.jsonl $OUT/e004-rows-$STAMP.jsonl \
    > $OUT/e004-REPORT-$STAMP.txt 2>&1
  { echo "## E-004"; cat $OUT/e004-REPORT-$STAMP.txt; } | tee -a $SUM
fi
say "=== done. Summary: $SUM (full reports alongside)"
