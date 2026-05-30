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
# Model: resolve (and download weights if missing) the NVFP4 repo via
# `hf download`, which is idempotent and prints the local snapshot dir. The 1M
# rope patch is (re)applied to that dir below, so a fresh download shipping the
# native-256k config is fine. Override MODEL=<dir> to serve a local copy (e.g. a
# fine-tuned checkpoint), or MODEL_REPO=<id> for a different hub model.
MODEL_REPO="${MODEL_REPO:-poolside/Laguna-XS.2-NVFP4}"
if [ -z "${MODEL:-}" ]; then
  echo "Resolving $MODEL_REPO (downloading weights if missing)..."
  # `hf download` prints the snapshot dir on its last line, prefixed "path="
  # in current CLI versions -- strip it if present.
  MODEL="$("$VENV/bin/hf" download "$MODEL_REPO" | tail -1)"
  MODEL="${MODEL#path=}"
fi
[ -d "$MODEL" ] || { echo "ERROR: model dir not resolved (got '$MODEL')" >&2; exit 1; }
# Honour the full 1M context the patched config advertises (256x YaRN; quality
# degrades past the model's native 256k -- accepted by design). B300 has 275GB,
# so raise GPU util to guarantee the KV cache can hold a 1M-token sequence.
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_LEN="${MAX_LEN:-1048576}"
# YaRN anchor: config rope_parameters.full_attention.original_max_position_embeddings.
# The extension invariant is MAX_LEN = ORIG_MAX * factor, so factor is derived
# below from MAX_LEN -- override MAX_LEN alone and the rope factor follows.
ORIG_MAX="${ORIG_MAX:-4096}"

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

# --- Make the 1M context extension reproducible ----------------------------
# The extension is a config.json edit (YaRN factor + max_position_embeddings),
# NOT model weights and NOT tracked in git. Re-apply it idempotently here so the
# serve advertises MAX_LEN regardless of HF-cache state (fresh download, cache
# clean, or a different machine). factor is derived as MAX_LEN/ORIG_MAX to keep
# the YaRN invariant; everything else in config.json is left untouched.
CFG="$MODEL/config.json"
[ -f "$CFG" ] || { echo "ERROR: config.json not found at $CFG" >&2; exit 1; }
# HF cache stores snapshot files as symlinks into blobs/; de-reference to a real
# file first so we patch the snapshot, not the shared content-addressed blob.
if [ -L "$CFG" ]; then
  real="$(readlink -f "$CFG")"; cp "$real" "$CFG.tmp" && mv -f "$CFG.tmp" "$CFG"
fi
"$VENV/bin/python" - "$CFG" "$MAX_LEN" "$ORIG_MAX" <<'PY'
import json, sys
cfg_path, max_len, orig_max = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
with open(cfg_path) as f:
    cfg = json.load(f)
factor = float(max_len) / float(orig_max)          # YaRN: max_len = original_max * factor
fa = cfg["rope_parameters"]["full_attention"]       # KeyError here => wrong model, fail loud
changed = (cfg.get("max_position_embeddings") != max_len) or (fa.get("factor") != factor)
cfg["max_position_embeddings"] = max_len
fa["factor"] = factor
if changed:
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[serve] patched config: max_position_embeddings={max_len}, full_attention.factor={factor}")
else:
    print(f"[serve] config already at max_position_embeddings={max_len}, full_attention.factor={factor}")
PY

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
