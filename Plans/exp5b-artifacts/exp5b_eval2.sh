#!/bin/zsh
# Exp-5B evaluation driver v2: per-test invocations with timeouts (v1 hung on impl-x A1).
set -u
REPO=/Users/andrebarlocher/Documents/Swift/NeoDiffusion
SP=/private/tmp/claude-502/-Users-andrebarlocher-Documents-Swift-NeoDiffusion/bebb36db-34d5-42c3-a37f-db0b953b6bbd/scratchpad/exp5b-hidden
OUT=$REPO/scratch/2026-10-18
GATE=$REPO/Tools/compute-gate.sh
SUM=$OUT/exp5b-eval-summary.log
export NEODIFFUSION_LOOP_FIXTURE_DIR=$REPO/scratch/loop_fixtures
# run_to <seconds> <logfile> <cmd...>: run under the gate, kill the test process tree on timeout.
run_to() {
  local T=$1 LOG=$2; shift 2
  "$@" > $LOG 2>&1 &
  local P=$!; local S=0
  while kill -0 $P 2>/dev/null; do
    sleep 5; S=$((S+5))
    if (( S >= T )); then pkill -P $P; pkill -f "xctest.*HiddenSessionCache"; sleep 2; kill $P 2>/dev/null; echo "TIMEOUT ${T}s" >> $LOG; return 124; fi
  done
  wait $P
}
for pair in "impl-y:agent-ac87f8dedec5e3cb7" "impl-x:agent-ae8403ea4fa3150dc"; do
  L=${pair%%:*}; W=$REPO/.claude/worktrees/${pair#*:}
  cd $W || exit 1
  echo "== v2 $L $(git rev-parse HEAD) $(date -u)" >> $SUM
  [ -e scratch ] || ln -s $REPO/scratch scratch
  cp $SP/HiddenSessionCacheEvalTests.swift Tests/DiffusionGenerationTests/
  $GATE eval-$L-build swift build --build-tests > $OUT/exp5b-eval-$L-build.log 2>&1; echo "$L build rc=$?" >> $SUM
  $GATE eval-$L-seed Tools/seed-metallib.sh >> $OUT/exp5b-eval-$L-build.log 2>&1
  for t in testA1_TwoTurnExactAndReuses testA2_DivergenceAndUnrelated testA3_IncompatibleParamsDoNotReuse testA4_QuantizedSessionExact testA5_ResetAndForwardAccounting; do
    S0=$(date +%s)
    run_to 600 $OUT/exp5b-eval-$L-$t.log $GATE eval-$L-$t swift test --skip-build --filter "HiddenSessionCacheEvalTests/$t"
    RC=$?; echo "$L $t rc=$RC secs=$(( $(date +%s) - S0 ))" >> $SUM
  done
  S0=$(date +%s)
  NEODIFFUSION_SESSION_EVAL_REAL=1 NEODIFFUSION_SESSION_EVAL_LABEL=$L NEODIFFUSION_SESSION_EVAL_OUT=$OUT/exp5b_session_perf.jsonl \
    run_to 1200 $OUT/exp5b-eval-$L-real.log $GATE eval-$L-real swift test --skip-build --filter HiddenSessionCacheRealEvalTests
  RC=$?; echo "$L real rc=$RC secs=$(( $(date +%s) - S0 ))" >> $SUM
  rm -f Tests/DiffusionGenerationTests/HiddenSessionCacheEvalTests.swift
done
echo "DONE v2 $(date -u)" >> $SUM
