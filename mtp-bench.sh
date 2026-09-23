#!/usr/bin/env bash
# mtp-bench.sh — measure llama-server prompt/decode speed via /completion.
#
# Usage: ./mtp-bench.sh [base_url] [n_predict] [runs]
#   base_url   default http://localhost:8484
#   n_predict  default 1000
#   runs       default 3 (plus one unmeasured warmup)
#
# Compare two server configs (e.g. TRY_MTP=1 vs TRY_MTP=0) by running this
# against each. Each measured run uses a unique prompt prefix so the server's
# prompt cache cannot carry state between runs, and the prompt is rotated so
# no two runs share content.
#
# Note on the "saved" column: tokens_cached in the response is an OUTPUT
# metric — how many tokens of the finished context the server RETAINED for
# possible reuse by a later request. With cache_prompt=false it is never
# actually consumed, so it does not affect these numbers.
set -euo pipefail

URL="${1:-http://localhost:8484}"
N_PREDICT="${2:-1000}"
RUNS="${3:-3}"

PROMPTS=(
  "Write a well-commented Python script implementing a thread-safe LRU cache with a max size, get/put methods, hit/miss/eviction statistics, a __repr__, and a __main__ demo that populates the cache, prints stats, and asserts correctness."
  "Write a well-commented Python script implementing a rate limiter using the token bucket algorithm with configurable capacity and refill rate, a decorator API, burst handling, and a __main__ demo that exercises refill, exhaustion, and shutdown with assertions."
  "Write a well-commented Python script implementing an in-memory pub/sub event bus with wildcard topic matching, synchronous and async dispatch, subscriber lifecycle management, and a __main__ demo that publishes events and asserts delivery counts and ordering."
)

run_once() {
  local tag="$1" prompt="$2"
  # cache_prompt=false: the server's slot cache (slot-prompt-similarity
  # defaults to 0.10) would otherwise reuse context from previous runs,
  # so each run would decode on top of the prior run's tokens.
  curl -s "${URL}/completion" -d '{
    "prompt": "Benchmark '"${tag}"': '"${prompt}"'",
    "n_predict": '"${N_PREDICT}"',
    "temp": 1.0,
    "cache_prompt": false
  }'
}

echo "target: ${URL}   n_predict: ${N_PREDICT}   runs: ${RUNS} (+1 warmup)"

run_once "warmup" "${PROMPTS[0]}" > /dev/null

echo
printf "%-4s %-9s %-12s %-12s %-11s %-9s %-8s\n" "run" "tokens" "decode_t/s" "prompt_t/s" "draft_acc" "stop" "saved"
rates=()
for i in $(seq 1 "${RUNS}"); do
  prompt="${PROMPTS[$(( (i - 1) % ${#PROMPTS[@]} ))]}"
  resp="$(run_once "run${i}" "${prompt}")"
  printf "%-4s " "r${i}"
  printf '%s' "${resp}" | jq -r '
    [ (.tokens_predicted | tostring),
      (.timings.predicted_per_second | floor | tostring),
      (.timings.prompt_per_second | floor | tostring),
      (if (.timings.draft_n // 0) > 0
       then ((.timings.draft_n_accepted / .timings.draft_n * 100) | floor | tostring) + "%"
       else "n/a" end),
      (.stop_type // "n/a"),
      (.tokens_cached | tostring)
    ] | @tsv' | awk '{printf "%-9s %-12s %-12s %-11s %-9s %-8s\n", $1, $2, $3, $4, $5, $6}'
  rates+=("$(printf '%s' "${resp}" | jq -r '.timings.predicted_per_second')")
  got="$(printf '%s' "${resp}" | jq -r '.tokens_predicted')"
  if [[ "${got}" != "${N_PREDICT}" ]]; then
    echo "  note: stopped early (tokens_predicted=${got} < n_predict=${N_PREDICT}); compare decode rates with care"
  fi
done

echo
printf "median decode: %s t/s (n=%s)\n" \
  "$(printf '%s\n' "${rates[@]}" | sort -n | awk '{a[NR]=$1} END {printf "%.0f", a[int((NR+1)/2)]}')" "${RUNS}"
