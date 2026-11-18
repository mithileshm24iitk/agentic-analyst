#!/usr/bin/env bash
# Smoke test against the deployed service. Needs SERVICE_URL and API_KEY in the environment.
#   1. GET /health is 200 and open
#   2. POST /ask without a key (and with a wrong key) is 401
#   3. POST /ask with the key: prints answer, route and latency
# Optional: Q="your question"
set -euo pipefail

: "${SERVICE_URL:?Set SERVICE_URL (printed by make deploy)}"
: "${API_KEY:?Set API_KEY (see docs/DEPLOY.md: gcloud secrets versions access)}"
URL="${SERVICE_URL%/}"
Q="${Q:-What is the maximum acceptable return rate for Jeans?}"
BODY="$(python3 -c 'import json, sys; print(json.dumps({"question": sys.argv[1]}))' "$Q")"
OUT="$(mktemp)"
trap 'rm -f "$OUT"' EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }

echo "1. GET /health"
code="$(curl -sS -o "$OUT" -w '%{http_code}' --max-time 60 "$URL/health")"  # 60s: allows a cold start
[[ "$code" == 200 ]] || fail "/health returned $code: $(cat "$OUT")"
echo "   200 $(cat "$OUT")"

echo "2. POST /ask without a key, then with a wrong key"
for header in "X-No-Key: 1" "X-API-Key: wrong-key"; do
  code="$(curl -sS -o "$OUT" -w '%{http_code}' --max-time 60 -X POST "$URL/ask" \
          -H "Content-Type: application/json" -H "$header" -d "$BODY")"
  [[ "$code" == 401 ]] || fail "expected 401 with '$header', got $code: $(cat "$OUT")"
done
echo "   401 both times"

echo "3. POST /ask with the key: $Q"
# The key header is read from a file descriptor, so it never shows up in `ps` output.
code="$(curl -sS -o "$OUT" -w '%{http_code}' --max-time 310 -X POST "$URL/ask" \
        -H "Content-Type: application/json" -H @<(printf 'X-API-Key: %s\n' "$API_KEY") -d "$BODY")"
[[ "$code" == 200 ]] || fail "/ask returned $code: $(cat "$OUT")"

python3 - "$OUT" <<'EOF'
import json, sys
body = json.load(open(sys.argv[1]))
m = body["metrics"]
cost = "unknown" if m["cost_usd"] is None else f"${m['cost_usd']:.4f}"
print(f"   route:   {body['route']} (agents: {', '.join(body['agents_called']) or 'none'})")
print(f"   latency: {m['latency_ms'] / 1000:.1f}s  llm_calls={m['llm_calls']}  cost={cost}")
print(f"   critic:  supported={body['critic']['supported']}  revisions={body['critic']['revisions']}")
print(f"   answer:  {body['answer']}")
EOF
echo "OK"
