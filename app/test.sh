#!/usr/bin/env bash
# test.sh — drive the FastAPI chat app with curl
# Run the server first: uvicorn main:app --reload
# Then: bash test.sh

BASE="http://localhost:8000"
SEP="─────────────────────────────────────────────"

echo ""
echo "1. Health check"
echo $SEP
curl -s "$BASE/health" | python3 -m json.tool

echo ""
echo "2. Start a new session (no session_id → auto-created)"
echo $SEP
RESPONSE=$(curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d '{"message": "My pod is stuck in CrashLoopBackOff. Where do I start debugging?"}')

echo $RESPONSE | python3 -m json.tool

# Extract session_id from the response
SESSION_ID=$(echo $RESPONSE | python3 -c "import sys,json; print(json.load(sys.stdin)['session_id'])")
echo ""
echo "Session ID: $SESSION_ID"

echo ""
echo "3. Continue the SAME conversation (passing session_id)"
echo $SEP
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"The logs show ImagePullBackOff. Now what?\", \"session_id\": \"$SESSION_ID\"}" \
  | python3 -m json.tool

echo ""
echo "4. One more turn — model remembers everything"
echo $SEP
curl -s -X POST "$BASE/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Show me the exact kubectl command to check image pull secrets\", \"session_id\": \"$SESSION_ID\"}" \
  | python3 -m json.tool

echo ""
echo "5. View full conversation history"
echo $SEP
curl -s "$BASE/history/$SESSION_ID" | python3 -m json.tool

echo ""
echo "6. Stream a response token-by-token"
echo $SEP
curl -s --no-buffer -X POST "$BASE/chat/stream" \
  -H "Content-Type: application/json" \
  -d '{"message": "Give me a one-liner to watch all pods across all namespaces"}'
echo ""

echo ""
echo "7. List all sessions"
echo $SEP
curl -s "$BASE/sessions" | python3 -m json.tool

echo ""
echo "8. Clear history (keep session)"
echo $SEP
curl -s -X DELETE "$BASE/history/$SESSION_ID" | python3 -m json.tool

echo ""
echo "Done."