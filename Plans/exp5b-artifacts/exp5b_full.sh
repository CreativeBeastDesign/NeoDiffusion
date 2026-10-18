#!/bin/zsh
set -u
REPO=/Users/andrebarlocher/Documents/Swift/NeoDiffusion; OUT=$REPO/scratch/2026-10-18; GATE=$REPO/Tools/compute-gate.sh; SUM=$OUT/exp5b-eval-summary.log
export NEODIFFUSION_LOOP_FIXTURE_DIR=$REPO/scratch/loop_fixtures
for pair in "impl-x:agent-ae8403ea4fa3150dc" "impl-y:agent-ac87f8dedec5e3cb7"; do
  L=${pair%%:*}; W=$REPO/.claude/worktrees/${pair#*:}; cd $W
  rm -f Tests/DiffusionGenerationTests/HiddenSessionCacheEvalTests.swift Tests/DiffusionGenerationTests/DiagA1Tests.swift
  [ -e scratch ] || ln -s $REPO/scratch scratch
  echo "== full $L $(git rev-parse HEAD) clean=$(git status --porcelain --untracked-files=no | wc -l) $(date -u)" >> $SUM
  $GATE full-$L-build swift build --build-tests > $OUT/exp5b-full-$L-build.log 2>&1
  $GATE full-$L-seed Tools/seed-metallib.sh >> $OUT/exp5b-full-$L-build.log 2>&1
  $GATE full-$L swift test --skip-build > $OUT/exp5b-full-$L.log 2>&1; echo "$L full rc=$?" >> $SUM
  grep -A1 "Test Suite 'All tests'" $OUT/exp5b-full-$L.log | grep Executed >> $SUM
done
echo "DONE full $(date -u)" >> $SUM
