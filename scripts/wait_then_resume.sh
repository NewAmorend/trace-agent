#!/usr/bin/env bash
# Wait for the Claude API rate-limit reset, run `swe batch --retry-agent-error`
# until every row is non-pending, then run `eval` on the done rows only.
#
# Idempotent and resumable. The batch runner itself is the source of truth for
# what still needs work — re-running this script after a kill is safe.
#
# Env knobs:
#   RUNS_DIR           default: data/swe/runs_v1
#   INSTANCES_FILE     default: $RUNS_DIR/instances.txt
#   EVAL_OUT           default: out/runs_v1_eval
#   WAKE_TZ            default: Asia/Shanghai
#   WAKE_HOUR          default: 19  (7pm)
#   MAX_RETRIES        default: 4
#   RETRY_SLEEP_SEC    default: 1800  (30 min between retries)
#   LOG                default: logs/resume_wait.log

set -uo pipefail

RUNS_DIR="${RUNS_DIR:-data/swe/runs_v1}"
INSTANCES_FILE="${INSTANCES_FILE:-${RUNS_DIR}/instances.txt}"
EVAL_OUT="${EVAL_OUT:-out/runs_v1_eval}"
WAKE_TZ="${WAKE_TZ:-Asia/Shanghai}"
WAKE_HOUR="${WAKE_HOUR:-19}"
MAX_RETRIES="${MAX_RETRIES:-4}"
RETRY_SLEEP_SEC="${RETRY_SLEEP_SEC:-1800}"
LOG="${LOG:-logs/resume_wait.log}"
TIMEOUT="${TIMEOUT:-360}"

mkdir -p "$(dirname "$LOG")"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$LOG"; }

# --- Compute wake epoch -----------------------------------------------------
WAKE_EPOCH=$(TZ="$WAKE_TZ" date -d "today ${WAKE_HOUR}:01" +%s)
NOW=$(date +%s)
if [ "$WAKE_EPOCH" -le "$NOW" ]; then
  WAKE_EPOCH=$((WAKE_EPOCH + 86400))
fi

log "wait_then_resume.sh starting (pid=$$)"
log "  RUNS_DIR=${RUNS_DIR}"
log "  INSTANCES_FILE=${INSTANCES_FILE}"
log "  WAKE_TZ=${WAKE_TZ} WAKE_HOUR=${WAKE_HOUR}"
log "  next wake: $(TZ="$WAKE_TZ" date -d "@${WAKE_EPOCH}" -Iseconds) ($(( (WAKE_EPOCH - NOW) / 60 )) min from now)"

# --- Wait -------------------------------------------------------------------
while [ "$(date +%s)" -lt "$WAKE_EPOCH" ]; do
  sleep 60
done
log "wake window reached; starting resume attempts"

# --- Retry loop -------------------------------------------------------------
attempt=0
while [ "$attempt" -lt "$MAX_RETRIES" ]; do
  attempt=$((attempt + 1))
  log "attempt ${attempt}/${MAX_RETRIES}: invoking swe batch --retry-agent-error"

  python main.py swe batch \
    --instances-file "$INSTANCES_FILE" \
    --runs-dir "$RUNS_DIR" \
    --agent claude \
    --timeout "$TIMEOUT" \
    --sleep-between 5 \
    --retry-agent-error \
    >> "$LOG" 2>&1

  REMAINING=$(python - <<PY
import json
m = json.load(open("${RUNS_DIR}/_batch.json"))
todo = [r for r in m if r.get("status") not in ("done",)]
print(len(todo))
PY
)
  log "attempt ${attempt} done. rows still non-done: ${REMAINING}"
  if [ "$REMAINING" -eq 0 ]; then
    break
  fi
  if [ "$attempt" -lt "$MAX_RETRIES" ]; then
    log "sleeping ${RETRY_SLEEP_SEC}s before next attempt..."
    sleep "$RETRY_SLEEP_SEC"
  fi
done

# --- Stage done trajectories for eval ---------------------------------------
log "staging done trajectories for eval"
EVAL_INPUT="${RUNS_DIR}_done"
rm -rf "$EVAL_INPUT"
mkdir -p "$EVAL_INPUT"

python - <<PY
import json, shutil
from pathlib import Path
m = json.load(open("${RUNS_DIR}/_batch.json"))
src = Path("${RUNS_DIR}")
dst = Path("${EVAL_INPUT}")
done = [r for r in m if r.get("status") == "done"]
print(f"done rows: {len(done)}")
for row in done:
    inst = row["instance_id"]
    traj = src / f"{inst}.jsonl"
    side = src / f"{inst}.jsonl.sidecar.json"
    if traj.exists():
        shutil.copy2(traj, dst / traj.name)
    if side.exists():
        shutil.copy2(side, dst / side.name)
PY

# --- Eval -------------------------------------------------------------------
log "running eval on ${EVAL_INPUT} -> ${EVAL_OUT}"
python main.py eval --input "$EVAL_INPUT" --output "$EVAL_OUT" >> "$LOG" 2>&1
log "eval complete; see ${EVAL_OUT}/batch_summary.md"

# --- Final summary ----------------------------------------------------------
log "===== final manifest summary ====="
python - <<PY 2>&1 | tee -a "$LOG"
import json
m = json.load(open("${RUNS_DIR}/_batch.json"))
from collections import Counter
print("status counts:")
for s, c in Counter(r.get("status") for r in m).most_common():
    print(f"  {s}: {c}")
print("verified counts:")
for v, c in Counter(r.get("verified_pass") for r in m).most_common():
    print(f"  {v}: {c}")
print(f"total: {len(m)}")
PY
log "wait_then_resume.sh finished"
