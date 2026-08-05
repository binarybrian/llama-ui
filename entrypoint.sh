#!/usr/bin/env bash
# llama-server entrypoint with MTP speculative-decoding auto-fallback.
#
# Tries MTP first (--spec-type draft-mtp). If the model lacks MTP layers or
# the process dies during startup, we detect that within a probe window and
# restart without the --spec-* flags.
#
# Flags mirror /spool/workspace/lmcp/q36-qwable-dau (Qwen3.6-27B DAU profile),
# adapted for the IQ2_M MTP model on the 4060 Ti 16GB.
set -uo pipefail

# -----------------------------------------------------------------------------
# Configuration (env-overridable so the same image works for other models)
# -----------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-/models/qwen36-dau/Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-IQ2_M.gguf}"
MMPROJ_PATH="${MMPROJ_PATH:-/models/qwen36-dau/mmproj-BF16.gguf}"
ALIAS="${ALIAS:-qwable-dau}"
PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"

# Probe window: how long to wait before deciding MTP startup failed.
# The 11GB IQ2_M model loading off NFS can take several minutes on a cold
# read; 600s is a safe upper bound. On TrueNAS with local SSD it'll be ~15s.
MTP_PROBE_SECONDS="${MTP_PROBE_SECONDS:-600}"

# -----------------------------------------------------------------------------
# Sanity checks
# -----------------------------------------------------------------------------
if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "ERROR: model not found at ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${MMPROJ_PATH}" ]]; then
  echo "WARN: mmproj not found at ${MMPROJ_PATH}; vision disabled" >&2
  MMPROJ_PATH=""
fi

# -----------------------------------------------------------------------------
# Build the argument list. argv[0] is the binary name; the rest are flags.
# -----------------------------------------------------------------------------
base_args=(
  llama-server
  -m "${MODEL_PATH}"
  --alias "${ALIAS}"
  --host "${HOST}" --port "${PORT}"
  -ngl all
  -fa on
  --ctx-size "${CTX_SIZE:-32768}"
  --kv-unified
  --cache-type-k q8_0
  --cache-type-v q5_0
  --parallel 1
  --jinja
  --reasoning-format auto
  --reasoning auto
  --reasoning-preserve
  --chat-template-kwargs '{"preserve_thinking":true}'
  --metrics
  --perf
  --log-timestamps
  --log-prefix
  --temp 0.6
  --top-p 0.95
  --top-k 20
  --min-p 0.00
  --repeat-penalty 1.0
  --presence-penalty 0.0
  --tools all
)

if [[ -n "${MMPROJ_PATH}" ]]; then
  base_args+=( --mmproj "${MMPROJ_PATH}" --image-min-tokens 1024 )
fi

mtp_args=(
  --spec-type draft-mtp
  --spec-draft-n-max 4
  --spec-draft-n-min 0
  --spec-draft-p-split 0.10
  --spec-draft-p-min 0.6
  --spec-draft-ngl auto
)

# -----------------------------------------------------------------------------
# Decide whether to try MTP (env override: TRY_MTP=0 to skip the probe entirely)
# -----------------------------------------------------------------------------
TRY_MTP="${TRY_MTP:-1}"

if [[ "${TRY_MTP}" != "1" ]]; then
  echo "TRY_MTP=0; starting without speculative decoding"
  exec "${base_args[@]}"
fi

# -----------------------------------------------------------------------------
# MTP probe: launch in background, wait for /health, fall back on death.
# Logs go to a tmp file so we can diagnose MTP failures and also stream to
# stdout so `docker logs` shows progress in real time.
# -----------------------------------------------------------------------------
echo "Starting llama-server with MTP speculative decoding (probe ${MTP_PROBE_SECONDS}s)..."
LOG=/tmp/llama-mtp-start.log
: > "${LOG}"

# tee to both the log file and stdout so docker logs shows progress
( "${base_args[@]}" "${mtp_args[@]}" ) 2>&1 | tee "${LOG}" &
pipe_pid=$!
# The actual llama-server PID is the tee'd process's child; find it.
# `tee` is PID ${pipe_pid}, llama-server is its stdin producer.
mtp_pid=$(pgrep -P "${pipe_pid}" -f llama-server | head -1 || echo "")

probe_ok=0
for _ in $(seq 1 "${MTP_PROBE_SECONDS}"); do
  # Check if the MTP process is still alive
  if [[ -n "${mtp_pid}" ]] && ! kill -0 "${mtp_pid}" 2>/dev/null; then
    probe_ok=0
    break
  fi
  # Once the HTTP server is listening, startup succeeded.
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    probe_ok=1
    break
  fi
  sleep 1
done

if [[ "${probe_ok}" == "1" ]]; then
  echo "MTP speculative decoding active. Server is listening on ${HOST}:${PORT}."
  # Hand the foreground to the running process; wait until it exits.
  wait "${mtp_pid}" 2>/dev/null
  exit $?
fi

# MTP failed — kill any straggler and restart without --spec-*.
echo "--------------------------------------------------------------------"
echo "MTP startup did not succeed within ${MTP_PROBE_SECONDS}s. Last log lines:"
tail -n 40 "${LOG}" >&2 || true
echo "--------------------------------------------------------------------"
echo "Restarting WITHOUT speculative decoding (plain decode)..."
[[ -n "${mtp_pid}" ]] && kill "${mtp_pid}" 2>/dev/null || true
kill "${pipe_pid}" 2>/dev/null || true
wait 2>/dev/null || true
sleep 2

# exec replaces the shell with llama-server as PID 1
exec "${base_args[@]}"