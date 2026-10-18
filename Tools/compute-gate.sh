#!/bin/zsh
# compute-gate.sh — serialize heavy compute (swift build / swift test / bench / generation) on the
# 16 GB dev M1 across concurrent agents and sessions.
#
# Usage:  Tools/compute-gate.sh <label> <command> [args...]
#   e.g.  Tools/compute-gate.sh opus-test swift test --filter SessionPrefixCacheTests
#
# Semantics (2026-10-18):
#   1. Acquire a machine-wide exclusive lock (atomic `mkdir` of $LOCK_DIR). If held by a live
#      process, wait; a lock whose owner PID is dead is stale and is removed.
#   2. While holding the lock, require available memory ≥ $MIN_AVAIL_MB (free + inactive +
#      speculative pages from vm_stat — memory the kernel can hand out without swapping).
#      If below, release the lock and retry after $RETRY_SECONDS (default 60).
#   3. Run the command, release the lock on exit (also on Ctrl-C / kill), propagate its exit code.
#   Every acquire/wait/release is appended to $GATE_LOG with timestamps, so total queueing time per
#   label can be audited afterwards (Tools/analysis/compute_gate_report.py).
#
# Env overrides: MIN_AVAIL_MB (default 5000), RETRY_SECONDS (60), LOCK_DIR, GATE_LOG.

set -u
LABEL="${1:?usage: compute-gate.sh <label> <command...>}"; shift
[[ $# -ge 1 ]] || { echo "compute-gate: no command given" >&2; exit 2; }

LOCK_DIR="${LOCK_DIR:-/tmp/neodiffusion-compute.lock}"
GATE_LOG="${GATE_LOG:-/tmp/neodiffusion-compute-gate.log}"
MIN_AVAIL_MB="${MIN_AVAIL_MB:-5000}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"

log() { print -r -- "$(date -u +%Y-%m-%dT%H:%M:%SZ) pid=$$ label=$LABEL $*" >> "$GATE_LOG"; }

avail_mb() {
  vm_stat | awk '
    /page size of/ { ps = $8 }
    /Pages free/ { f = $3 } /Pages inactive/ { i = $3 } /Pages speculative/ { s = $3 }
    END { gsub(/\./, "", f); gsub(/\./, "", i); gsub(/\./, "", s); printf "%d", (f + i + s) * ps / 1048576 }'
}

acquire() {
  local owner_pid=""   # declared once: zsh prints `name=value` when `local` re-declares a set variable
  while true; do
    if mkdir "$LOCK_DIR" 2>/dev/null; then
      print -r -- "$$ $LABEL $(date -u +%s)" > "$LOCK_DIR/owner"
      return 0
    fi
    owner_pid=$(awk '{print $1}' "$LOCK_DIR/owner" 2>/dev/null)
    if [[ -n "$owner_pid" ]] && ! kill -0 "$owner_pid" 2>/dev/null; then
      log "stale-lock owner=$owner_pid removed"
      rm -rf "$LOCK_DIR"
      continue
    fi
    log "wait-lock holder=$(cat "$LOCK_DIR/owner" 2>/dev/null)"
    sleep "$RETRY_SECONDS"
  done
}

release() { [[ -f "$LOCK_DIR/owner" ]] && [[ "$(awk '{print $1}' "$LOCK_DIR/owner")" == "$$" ]] && rm -rf "$LOCK_DIR"; }
trap 'release; log "released (signal)"; exit 130' INT TERM
trap 'release' EXIT

REQUESTED=$(date -u +%s)
while true; do
  acquire
  AV=$(avail_mb)
  if (( AV >= MIN_AVAIL_MB )); then break; fi
  log "low-memory avail=${AV}MB < ${MIN_AVAIL_MB}MB; retry in ${RETRY_SECONDS}s"
  release
  sleep "$RETRY_SECONDS"
done
START=$(date -u +%s)
log "start waited=$((START - REQUESTED))s avail=${AV}MB cmd=$*"
"$@"
RC=$?
END=$(date -u +%s)
log "end rc=$RC ran=$((END - START))s"
exit $RC
