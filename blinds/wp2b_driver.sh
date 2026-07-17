#!/bin/zsh
# WP-2b sweep driver (2026-07-11). Arms recorded in Plans/wp2b-logbook.md §3.
# Counters are deterministic at temp 0 -> --runs 1; wall-clock is host-scoped anyway (§0.1).
BIN=.build/release/diffusion-bench
JSON=scratch/wp2b_sweep.jsonl

run_arm() {
  local name=$1; shift
  echo "==== ARM $name ($(date +%H:%M:%S)) ===="
  $BIN llada --runs 1 --cooldown 0 --suites chat,reasoning --gen-length 128 \
    --arm "$name" --json "$JSON" "$@"
}

# 2b-2 dynamic tau sweep (Q mode, tau0 = 0.7)
run_arm base-wp2b   --arm-mode q --dump-text scratch/wp2b_text_base.jsonl
run_arm dyntau-a03  --arm-mode q --dyn-tau-alpha 0.3
run_arm dyntau-a06  --arm-mode q --dyn-tau-alpha 0.6
run_arm dyntau-a09  --arm-mode q --dyn-tau-alpha 0.9
# S mode (tau0 = 0.5)
run_arm base-wp2b-s  --arm-mode s
run_arm dyntau-s-a03 --arm-mode s --dyn-tau-alpha 0.3
run_arm dyntau-s-a06 --arm-mode s --dyn-tau-alpha 0.6
# paper tau0 = 0.9 probe (threshold-calibration consolidation, Optimisations.md)
run_arm t09-static  --arm-mode q --threshold-mask 0.9
run_arm t09-dyn-a06 --arm-mode q --threshold-mask 0.9 --dyn-tau-alpha 0.6
# 2b-3 EOS early exit (baseline arm base-wp2b has eosEarlyStop on by default)
run_arm eosexit --arm-mode q --eos-early-exit --dump-text scratch/wp2b_text_eosexit.jsonl
# nBuf=2 composability arms (matrix cells 1b x 2b-2 and 1b x 2b-3)
run_arm nbuf2-base       --arm-mode q --n-buf 2 --tau-add 0.5
run_arm nbuf2-dyntau-a06 --arm-mode q --n-buf 2 --tau-add 0.5 --dyn-tau-alpha 0.6
run_arm nbuf2-eosexit    --arm-mode q --n-buf 2 --tau-add 0.5 --eos-early-exit
echo "==== SWEEP DONE ($(date +%H:%M:%S)) ===="
