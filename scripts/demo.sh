#!/bin/bash
set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

echo "=== Kubeflow Docs Agent - Demo ==="
echo "API: $API_URL"
echo ""

# health check
echo "1. Health Check"
echo "   GET /health"
curl -s "$API_URL/health" | python3 -m json.tool
echo ""

# docs query
echo "2. Docs Query: 'How do I install Kubeflow?'"
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I install Kubeflow?", "stream": false}' | python3 -m json.tool
echo ""

# greeting
echo "3. Greeting: 'Hello'"
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "Hello", "stream": false}' | python3 -m json.tool
echo ""

# out of scope
echo "4. Out of Scope: 'What is the weather?'"
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the weather?", "stream": false}' | python3 -m json.tool
echo ""

# issues-style query
echo "5. Issues Query: 'CrashLoopBackOff error in pipeline pod'"
curl -s -X POST "$API_URL/chat" \
  -H "Content-Type: application/json" \
  -d '{"query": "CrashLoopBackOff error in pipeline pod", "stream": false}' | python3 -m json.tool
echo ""

echo "=== Demo Complete ==="
