#!/bin/zsh
# Exp-5B evaluation driver (orchestrator). Sequential; every heavy step via compute-gate.
set -u
REPO=/Users/andrebarlocher/Documents/Swift/NeoDiffusion
SP=/private/tmp/claude-502/-Users-andrebarlocher-Documents-Swift-NeoDiffusion/bebb36db-34d5-42c3-a37f-db0b953b6bbd/scratchpad/exp5b-hidden
OUT=$REPO/scratch/2026-10-18
GATE=$REPO/Tools/compute-gate.sh
export NEODIFFUSION_LOOP_FIXTURE_DIR=$REPO/scratch/loop_fixtures
for pair in "impl-x:agent-ae8403ea4fa3150dc" "impl-y:agent-ac87f8dedec5e3cb7"; do
  L=${pair%%:*}; W=$REPO/.claude/worktrees/${pair#*:}
  cd $W || exit 1
  echo "== $L $(git rev-parse HEAD) $(date -u)" >> $OUT/exp5b-eval-summary.log
  [ -e scratch ] || ln -s $REPO/scratch scratch
  cp $SP/HiddenSessionCacheEvalTests.swift Tests/DiffusionGenerationTests/
  $GATE eval-$L-build swift build --build-tests > $OUT/exp5b-eval-$L-build.log 2>&1; echo "$L build rc=$?" >> $OUT/exp5b-eval-summary.log
  $GATE eval-$L-seed Tools/seed-metallib.sh >> $OUT/exp5b-eval-$L-build.log 2>&1
  $GATE eval-$L-hidden swift test --skip-build --filter HiddenSessionCacheEvalTests > $OUT/exp5b-eval-$L-hidden.log 2>&1; echo "$L hidden rc=$?" >> $OUT/exp5b-eval-summary.log
  NEODIFFUSION_SESSION_EVAL_REAL=1 NEODIFFUSION_SESSION_EVAL_LABEL=$L NEODIFFUSION_SESSION_EVAL_OUT=$OUT/exp5b_session_perf.jsonl \
    $GATE eval-$L-real swift test --skip-build --filter HiddenSessionCacheRealEvalTests > $OUT/exp5b-eval-$L-real.log 2>&1; echo "$L real rc=$?" >> $OUT/exp5b-eval-summary.log
  $GATE eval-$L-full swift test --skip-build > $OUT/exp5b-eval-$L-full.log 2>&1; echo "$L full rc=$?" >> $OUT/exp5b-eval-summary.log
  rm Tests/DiffusionGenerationTests/HiddenSessionCacheEvalTests.swift
done
echo "DONE $(date -u)" >> $OUT/exp5b-eval-summary.log
