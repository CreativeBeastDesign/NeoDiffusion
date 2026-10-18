#!/bin/zsh
set -u
REPO=/Users/andrebarlocher/Documents/Swift/NeoDiffusion
SP=/private/tmp/claude-502/-Users-andrebarlocher-Documents-Swift-NeoDiffusion/bebb36db-34d5-42c3-a37f-db0b953b6bbd/scratchpad/exp5b-hidden
OUT=$REPO/scratch/2026-10-18; GATE=$REPO/Tools/compute-gate.sh; SUM=$OUT/exp5b-eval-summary.log
export NEODIFFUSION_LOOP_FIXTURE_DIR=$REPO/scratch/loop_fixtures
run_to() { local T=$1 LOG=$2; shift 2; "$@" > $LOG 2>&1 & local P=$!; local S=0
  while kill -0 $P 2>/dev/null; do sleep 5; S=$((S+5)); if (( S >= T )); then pkill -f "xctest.*HiddenSessionCache"; sleep 2; kill $P 2>/dev/null; echo "TIMEOUT ${T}s" >> $LOG; return 124; fi; done; wait $P; }
for spec in "impl-y:agent-ac87f8dedec5e3cb7:testA1_TwoTurnExactAndReuses" "impl-x:agent-ae8403ea4fa3150dc:testA1_TwoTurnExactAndReuses testA2_DivergenceAndUnrelated testA3_IncompatibleParamsDoNotReuse testA4_QuantizedSessionExact testA5_ResetAndForwardAccounting"; do
  L=${spec%%:*}; rest=${spec#*:}; W=$REPO/.claude/worktrees/${rest%%:*}; tests=${rest#*:}
  cd $W || exit 1; echo "== v3 $L $(git rev-parse HEAD) $(date -u)" >> $SUM
  [ -e scratch ] || ln -s $REPO/scratch scratch
  cp $SP/HiddenSessionCacheEvalTests.swift Tests/DiffusionGenerationTests/
  $GATE eval3-$L-build swift build --build-tests > $OUT/exp5b-eval3-$L-build.log 2>&1; echo "$L build rc=$?" >> $SUM
  $GATE eval3-$L-seed Tools/seed-metallib.sh >> $OUT/exp5b-eval3-$L-build.log 2>&1
  for t in ${=tests}; do S0=$(date +%s)
    run_to 600 $OUT/exp5b-eval3-$L-$t.log $GATE eval3-$L-$t swift test --skip-build --filter "HiddenSessionCacheEvalTests/$t"
    RC=$?; echo "$L $t (v3) rc=$RC secs=$(( $(date +%s) - S0 ))" >> $SUM; done
done
echo "DONE v3 $(date -u)" >> $SUM
