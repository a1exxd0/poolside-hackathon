curl -s http://127.0.0.1:8000/v1/models | python3 -m json.tool


curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "poolside/Laguna-XS.2-NVFP4-1Mctx",
    "messages": [{"role": "user", "content": "Say hi in 3 words."}],
    "max_tokens": 16
  }' | python3 -m json.tool
