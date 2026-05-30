#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible server with the custom INT4-KIVI KV cache.
# Env mirrors the verified run recipe in VLLM_SETUP.md.
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# IMPORTANT: do NOT pin VLLM_ATTENTION_BACKEND when kv_cache_dtype=int4_kivi --
# let the selector pick the custom Int4Kivi backend via supported_kv_cache_dtypes.

MODEL="${MODEL_ID:-poolside/Laguna-XS.2}"
KV_DTYPE="${KV_CACHE_DTYPE:-int4_kivi}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_MODEL_LEN:-14336}"
GPU_UTIL="${GPU_MEMORY_UTILIZATION:-0.90}"

# HF_TOKEN (endpoint secret) is required: Laguna-XS.2 is a gated/private repo.
exec vllm serve "${MODEL}" \
    --dtype bfloat16 \
    --kv-cache-dtype "${KV_DTYPE}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization "${GPU_UTIL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --enforce-eager \
    ${EXTRA_VLLM_ARGS:-}
