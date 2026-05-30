# syntax=docker/dockerfile:1.7
# ---------------------------------------------------------------------------
# Custom INT4-KIVI vLLM serving image for poolside/Laguna-XS.2
#
# The custom KV-cache backend lives in the ./vllm submodule fork
# (vllm/v1/attention/backends/int4_kivi_attn.py + ops/triton_int4_kivi.py).
# It is Triton + precompiled _C only -- no nvcc-JIT -- so it builds without
# CUDA >= 12.9. See VLLM_SETUP.md for the constraints this mirrors.
#
# NOTE ON HARDWARE: the benchmarks were on a B300 (sm_103a). This image targets
# an A100 (sm_80) HF endpoint. Triton recompiles per-arch at runtime, so the
# int4_kivi path is portable, but you will NOT reproduce the B300 speedup
# numbers on A100. Set TORCH_CUDA_ARCH_LIST to the target GPU's arch.
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    # A100 = sm_80. (H100=9.0a; B300=10.3a needs a CUDA>=12.9 base.)
    # Mostly moot with VLLM_USE_PRECOMPILED=1 (nothing CUDA-compiles at build),
    # but set to the target arch for any source build path.
    TORCH_CUDA_ARCH_LIST="8.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

# uv for the same install flow as VLLM_SETUP.md
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /opt/kv-quant

# --- Copy source (incl. the vllm submodule contents + int4_kivi package) -----
# The build context MUST already contain the submodule working tree with the
# custom backend (see the .gitmodules / submodule blocker note in the README).
COPY . /opt/kv-quant

# --- Build the venv exactly as VLLM_SETUP.md prescribes ----------------------
# Precompiled vLLM (_C/_moe_C) + Triton-only; no FlashInfer (forces Triton MoE
# + FLASH_ATTN, which is what the int4_kivi selector expects).
RUN uv venv /opt/venv --python 3.12 \
    && VLLM_USE_PRECOMPILED=1 uv pip install --python /opt/venv/bin/python \
         -e ./vllm --torch-backend=auto \
    && uv pip install --python /opt/venv/bin/python ninja \
    && uv pip uninstall --python /opt/venv/bin/python flashinfer-python || true \
    && uv pip install --python /opt/venv/bin/python \
         accelerate transformers datasets python-dotenv

ENV PATH="/opt/venv/bin:${PATH}"

# Runtime cwd MUST NOT contain a bare `vllm/` dir, or `import vllm` resolves to
# the submodule namespace pkg instead of the installed package (VLLM_SETUP.md).
WORKDIR /srv

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# vLLM OpenAI-compatible server. HF endpoint health probe -> GET /health.
EXPOSE 8000
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
