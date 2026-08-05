#!/usr/bin/env bash
# llama-server entrypoint with MTP speculative-decoding auto-fallback.
#
# Tries MTP first (--spec-type draft-mtp). If the model lacks MTP layers,
# llama-server dies during startup; we detect that within a probe window
# and restart without the --spec-* flags.
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
# The 11GB IQ2_M model loads off SSD in ~5-15s; 45s is a safe upper bound.
MTP_PROBE_SECONDS="${MTP_PROBE_SECONDS:-45}"

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
# Build the argument list (shared between MTP and non-MTP invocations)
# -----------------------------------------------------------------------------
base_args=(
  llama-server
  -m "${MODEL_PATH}"
  --alias "${ALIAS}"
  --host "${HOST}" --port "${PORT}"
  -ngl all
  -fa on
  --fit on
  --fit-target 256
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

run_server() {
  # Exec into llama-server in the foreground; this is the final process (PID 1).
  exec llama-server "$@"
}

if [[ "${TRY_MTP}" != "1" ]]; then
  echo "TRY_MTP=0; starting without speculative decoding"
  run_server "${base_args[@]}"
  exit 0  # unreachable; exec replaces us
fi

# -----------------------------------------------------------------------------
# MTP probe: launch in background, wait for /health, fall back on death.
# We write logs to a tmp file so we can diagnose MTP failures.
# -----------------------------------------------------------------------------
echo "Starting llama-server with MTP speculative decoding (probe ${MTP_PROBE_SECONDS}s)..."
LOG=/tmp/llama-mtp-start.log
: > "${LOG}"

( "${base_args[@]}" "${mtp_args[@]}" ) >"${LOG}" 2>&1 &
pid=$!

# Poll the process: if it dies within the probe window, MTP isn't supported.
probe_ok=0
for _ in $(seq 1 "${MTP_PROBE_SECONDS}"); do
  if ! kill -0 "${pid}" 2>/dev/null; then
    probe_ok=0
    break
  fi
  # Once the HTTP server is listening, the model loaded successfully.
  if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
    probe_ok=1
    break
  fi
  sleep 1
done

if [[ "${probe_ok}" == "1" ]]; then
  # MTP worked. Hand the process to the foreground so it becomes PID 1's child
  # and we stream its logs. `wait` blocks until it exits; its exit code
  # becomes our exit code.
  echo "MTP speculative decoding active. Tailing server logs..."
  tail -f "${LOG}" --pid="${pid}" 2>/dev/null &
  wait "${pid}"
  exit $?
fi

# MTP failed — kill any straggler and restart without --spec-*.
echo "--------------------------------------------------------------------"
echo "MTP startup did not succeed within ${MTP_PROBE_SECONDS}s. Last log lines:"
tail -n 40 "${LOG}" >&2 || true
echo "--------------------------------------------------------------------"
echo "Restarting WITHOUT speculative decoding (plain decode)..."
kill "${pid}" 2>/dev/null || true
wait "${pid}" 2>/dev/null || true
sleep 1

run_server "${base_args[@]}"