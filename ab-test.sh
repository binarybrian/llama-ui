#!/usr/bin/env bash
# A/B quality comparison: local 5090 (Q6_K) vs TrueNAS 4060 Ti (IQ2_M)
# Sends identical prompts to both endpoints and saves results side-by-side.
set -uo pipefail

LOCAL="http://localhost:8484/v1/chat/completions"
REMOTE="http://192.168.2.1:30084/v1/chat/completions"
OUT="/tmp/llama-ab-test-$(date +%Y%m%d-%H%M%S).md"
TMPFILE="/tmp/ab-response-$$.json"

cat <<EOF > "$OUT"
# llama.cpp A/B Quality Comparison

**Date**: $(date)
**Local (5090, Q6_K)**: $LOCAL
**Remote (4060 Ti, IQ2_M)**: $REMOTE
**Temperature**: 0.3, **Max tokens**: unlimited

EOF

send_prompt() {
  local label="$1"
  local prompt="$2"
  local endpoint_name="$3"
  local endpoint_url="$4"

  echo "  → $endpoint_name" >&2

  local http_code
  http_code=$(curl -sS \
    --retry 3 \
    --retry-delay 5 \
    --retry-connrefused \
    --connect-timeout 10 \
    --max-time 600 \
    --keepalive-time 30 \
    --keepalive-cnt 3 \
    -o "$TMPFILE" \
    -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -d "{
      \"model\": \"qwable-dau\",
      \"messages\": [{\"role\": \"user\", \"content\": $(python3 -c "import json,sys; print(json.dumps(sys.argv[1]))" "$prompt")}],
      \"temperature\": 0.3,
      \"stream\": false
    }" \
    "$endpoint_url" 2>/tmp/ab-curl-stderr-$$.log)

  local curl_exit=$?

  if [[ "$http_code" != "200" ]] || [[ $curl_exit -ne 0 ]]; then
    local err_msg
    err_msg=$(cat /tmp/ab-curl-stderr-$$.log 2>/dev/null || echo "(no stderr)")
    echo "" >> "$OUT"
    echo "### $endpoint_name" >> "$OUT"
    echo "" >> "$OUT"
    echo '```' >> "$OUT"
    echo "ERROR: HTTP $http_code (curl exit $curl_exit) after 3 retries" >> "$OUT"
    echo "stderr: $err_msg" >> "$OUT"
    echo "response: $(head -c 500 "$TMPFILE" 2>/dev/null || echo '(empty)')" >> "$OUT"
    echo '```' >> "$OUT"
    echo "" >> "$OUT"
    rm -f "$TMPFILE" /tmp/ab-curl-stderr-$$.log
    return
  fi

  local response
  response=$(python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    msg = d['choices'][0]['message']
    content = msg.get('content', '') or ''
    reasoning = msg.get('reasoning_content', '') or ''
    u = d.get('usage', {})
    t = d.get('choices', [{}])[0].get('timings', {})

    if content:
        print(content)
    else:
        print('(no content — all tokens consumed by reasoning)')
    if reasoning:
        rlen = len(reasoning)
        print()
        print(f'[Reasoning: {rlen} chars, first 200:]')
        print(reasoning[:200])
        if rlen > 200:
            print('...')
    print()
    print(f'---')
    print(f'Tokens: {u.get(\"total_tokens\",\"?\")} (prompt={u.get(\"prompt_tokens\",\"?\")}, completion={u.get(\"completion_tokens\",\"?\")})')
    if t:
        try:
            print(f'Speed: {t.get(\"predicted_per_second\",0):.1f} t/s')
        except:
            pass
except Exception as e:
    print(f'ERROR parsing response: {e}')
" < "$TMPFILE" 2>&1)

  rm -f "$TMPFILE" /tmp/ab-curl-stderr-$$.log

  echo "" >> "$OUT"
  echo "### $endpoint_name" >> "$OUT"
  echo "" >> "$OUT"
  echo '```' >> "$OUT"
  echo "$response" >> "$OUT"
  echo '```' >> "$OUT"
  echo "" >> "$OUT"
}

run_test() {
  local num="$1"
  local title="$2"
  local prompt="$3"

  echo "" >&2
  echo "=== Test $num: $title ===" >&2

  echo "" >> "$OUT"
  echo "---" >> "$OUT"
  echo "" >> "$OUT"
  echo "## Test $num: $title" >> "$OUT"
  echo "" >> "$OUT"
  echo "**Prompt**:" >> "$OUT"
  echo "" >> "$OUT"
  echo "> $prompt" >> "$OUT"
  echo "" >> "$OUT"

  send_prompt "$num" "$prompt" "Local (5090, Q6_K)" "$LOCAL"
  send_prompt "$num" "$prompt" "Remote (4060 Ti, IQ2_M)" "$REMOTE"
}

echo "Running A/B quality tests..." >&2
echo "Output: $OUT" >&2

run_test 1 "Math Reasoning" \
"A farmer has 3 fields. Field A yields 2.5x more corn than Field B. Field C yields 40% less than Field A. If Field B yields 800 bushels, how many total bushels do all three fields produce? Show your work step by step."

run_test 2 "Code Generation" \
"Write a Python function that takes a list of dictionaries and returns a new list with only the entries where the \"status\" key equals \"active\" and the \"score\" key is greater than 50. Sort the result by score descending. Include type hints and a docstring."

run_test 3 "Multi-Constraint Instruction Following" \
"Write a 4-line poem about the ocean. Rules: (1) Each line must start with a different letter of the alphabet in order (A, B, C, D). (2) No word may be repeated. (3) The poem must mention a specific sea creature. (4) Line 3 must contain a color."

run_test 4 "Factual Recall" \
"Name the 8 planets in our solar system in order from the sun, and give one distinguishing fact about each. Format as a numbered list."

echo "" >&2
echo "Done! Results saved to: $OUT" >&2
echo "" >&2
cat "$OUT"