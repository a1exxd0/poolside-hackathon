#!/usr/bin/env bash
# Launch the custom INT4-KIVI vLLM serve for the local YaRN-extended
# poolside/Laguna-XS.2-NVFP4 (config patched to factor=256 / 1M positions) and
# expose it publicly via a cloudflared quick tunnel (outbound 443 only; no
# inbound ports beyond ssh/22). Mirrors VLLM_SETUP.md's verified run recipe.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/.tools"
VENV="$ROOT/.venv-vllm"
PORT="${PORT:-8000}"
# Serve the patched NVFP4 snapshot from the HF cache by absolute path so vLLM
# loads our edited config.json verbatim (no hub re-check / no re-download that
# could revert the 1M rope patch). Override with MODEL=... if needed.
MODEL="${MODEL:-/home/alex/.cache/huggingface/hub/models--poolside--Laguna-XS.2-NVFP4/snapshots/d9c13a066a97f81abd091da69f17ec5cb72a93ca}"
# Honour the full 1M context the patched config advertises (256x YaRN; quality
# degrades past the model's native 256k -- accepted by design). B300 has 275GB,
# so raise GPU util to guarantee the KV cache can hold a 1M-token sequence.
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-1048576}"

# API key: reuse if present, else mint one. Required -- this is a public URL.
KEYFILE="$TOOLS/api_key.txt"
[ -s "$KEYFILE" ] || echo "sk-laguna-$(openssl rand -hex 20)" > "$KEYFILE"
KEY="$(cat "$KEYFILE")"

# cloudflared binary (gitignored; fetch if missing).
CF="$TOOLS/cloudflared"
if [ ! -x "$CF" ]; then
  echo "Downloading cloudflared..."
  curl -fsSL -o "$CF" \
    https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
  chmod +x "$CF"
fi

# --- vLLM ------------------------------------------------------------------
VLOG="$TOOLS/vllm_serve.log"; : > "$VLOG"
# Run from a non-vllm/ cwd or `import vllm` resolves to the submodule dir.
( cd /tmp && \
  CUDA_HOME=/usr/local/cuda-12.8 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  nohup "$VENV/bin/vllm" serve "$MODEL" \
    --served-model-name poolside/Laguna-XS.2-NVFP4 \
    --dtype bfloat16 \
    --kv-cache-dtype int4_kivi \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --host 127.0.0.1 --port "$PORT" \
    --enforce-eager \
    --api-key "$KEY" >> "$VLOG" 2>&1 & echo $! > "$TOOLS/vllm.pid" )
echo "vLLM pid $(cat "$TOOLS/vllm.pid") -- waiting for startup (model load)..."
for _ in $(seq 1 80); do
  grep -qiE "Application startup complete" "$VLOG" && break
  if grep -qiE "OutOfMemory|EngineCore failed|Engine core init" "$VLOG"; then
    echo "vLLM failed to start -- see $VLOG (likely GPU OOM: check nvidia-smi)"; exit 1
  fi
  sleep 3
done

# --- cloudflared quick tunnel ---------------------------------------------
TLOG="$TOOLS/cloudflared.log"; : > "$TLOG"
nohup "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >> "$TLOG" 2>&1 &
echo $! > "$TOOLS/cloudflared.pid"
URL=""
for _ in $(seq 1 30); do
  URL=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" "$TLOG" | head -1)
  [ -n "$URL" ] && break
  sleep 2
done
echo "$URL" > "$TOOLS/public_url.txt"

echo
echo "Public URL : ${URL:-<not found -- see $TLOG>}"
echo "API key    : $KEY"
echo "Model      : poolside/Laguna-XS.2-NVFP4 (1M ctx, YaRN factor=256)"
echo "Max ctx    : $MAX_LEN tokens"
echo "Stop with  : $TOOLS/stop.sh"
