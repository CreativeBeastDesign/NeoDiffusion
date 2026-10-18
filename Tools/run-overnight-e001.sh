#!/bin/zsh
# run-overnight-e001.sh — the pre-registered E-001 run (paper/experiments/E-001.md, frozen at 7fb7369):
# 2×2 KV-cache quantization (+ kivi4) on gsm8k_100 + chat/reasoning/code, K=1, clean-commit on,
# prompt-major interleaved arms, one process. Run from a plain terminal with the Claude app QUIT:
#
#     cd ~/Documents/Swift/NeoDiffusion && caffeinate -ims Tools/run-overnight-e001.sh
#
# Expected wall-clock ≈ 2–3 h (5 arms × 112 prompts at gen 128 ≈ 560 generations at ~10–15 s, plus
# build and model load) `[Inferred: ~0.185 s/forward on this host, ~30–60 forwards per generation]`.
# Everything goes to scratch/2026-10-19/e001/; the summary (incl. the pre-registered verdicts from
# paper/scripts/e001_analysis.py) is printed at the end and saved as REPORT.txt. Safe to Ctrl-C;
# rows already written stay in the JSONL (re-running appends to new files with a timestamp suffix).

set -u
REPO=${0:A:h:h}
STAMP=$(date +%Y%m%d-%H%M)
OUT=$REPO/scratch/2026-10-19/e001
mkdir -p $OUT
LOG=$OUT/run-$STAMP.log
ROWS=$OUT/e001_rows-$STAMP.jsonl
TEXT=$OUT/e001_text-$STAMP.jsonl
REPORT=$OUT/REPORT-$STAMP.txt
GATE=$REPO/Tools/compute-gate.sh
export RETRY_SECONDS=30

say() { print -r -- "$(date '+%H:%M:%S') $*" | tee -a $LOG; }
avail_mb() { vm_stat | awk '/page size of/{ps=$8} /Pages free/{f=$3} /Pages inactive/{i=$3} /Pages speculative/{s=$3}
  END{gsub(/\./,"",f);gsub(/\./,"",i);gsub(/\./,"",s);printf "%d",(f+i+s)*ps/1048576}'; }

cd $REPO
say "=== E-001 overnight start | host $(sysctl -n hw.model) | git $(git rev-parse --short HEAD) | avail $(avail_mb) MB | $(sysctl -n vm.swapusage)"
if pgrep -xq Claude; then say "WARNING: the Claude desktop app is running — quit it for clean timings (steps/accuracy are unaffected)."; fi
DIRTY=$(git status --porcelain --untracked-files=no | wc -l | xargs)
if [[ $DIRTY != 0 ]]; then
  say "REFUSING: tracked files are modified ($DIRTY). E-001 must run from a committed tree (pre-registration rule P-5)."
  git status --short --untracked-files=no | tee -a $LOG
  exit 1
fi
# The analysis script must be the committed one (pre-registration says: committed before the run).
git log -1 --format=%H -- paper/scripts/e001_analysis.py | xargs -I{} say "analysis script committed at {}"

say "build: release diffusion-bench"
MIN_AVAIL_MB=2000 $GATE e001-build swift build -c release --product diffusion-bench > $OUT/build-$STAMP.log 2>&1 \
  || { say "BUILD FAILED — see $OUT/build-$STAMP.log"; exit 1; }
MIN_AVAIL_MB=2000 $GATE e001-seed Tools/seed-metallib.sh >> $OUT/build-$STAMP.log 2>&1

say "bench: 5 arms × gsm8k_100+chat+reasoning+code, gen 128, K=1, interleaved (avail $(avail_mb) MB)"
MIN_AVAIL_MB=6500 $GATE e001-bench ./.build/release/diffusion-bench llada \
  --arms cache-clean-commit-k1,cache-k8v4-clean-k1,cache-k4v8-clean-k1,cache-fp4-clean-k1,cache-kivi4-clean-k1 \
  --speculation-k 1 --interleave --suites gsm8k_100,chat,reasoning,code --runs 1 --gen-length 128 \
  --warmup-once --json $ROWS --dump-text $TEXT > $OUT/bench-$STAMP.log 2>&1
RC=$?
say "bench rc=$RC | rows $(wc -l < $ROWS 2>/dev/null | xargs) | text $(wc -l < $TEXT 2>/dev/null | xargs) | $(sysctl -n vm.swapusage)"
[[ $RC == 0 ]] || say "bench FAILED — see $OUT/bench-$STAMP.log (partial rows are still analysed below)"

say "analysis (paper/scripts/e001_analysis.py)"
python3 paper/scripts/e001_analysis.py $ROWS $TEXT > $REPORT 2>&1
cat $REPORT | tee -a $LOG
say "=== done. Report: $REPORT"
