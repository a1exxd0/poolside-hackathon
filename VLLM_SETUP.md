# Running vLLM with the custom INT4-KIVI path on this B300

The B300 is **sm_103a**, but only the **CUDA 12.8** toolkit is installed (sm_103a
needs CUDA ≥12.9). torch is **2.11.0+cu130**. Consequence: **any runtime nvcc-JIT
fails** (`nvcc fatal: Unsupported gpu architecture 'compute_103a'`). The
precompiled vLLM `_C`/`_moe_C` and **all Triton paths work** (Triton uses the
driver/LLVM, not nvcc). So we run vLLM Triton-only + precompiled, no FlashInfer.

## Isolated env (does NOT touch the working `.venv` / torch 2.12)
```bash
uv venv .venv-vllm --python 3.12
VLLM_USE_PRECOMPILED=1 uv pip install --python .venv-vllm/bin/python -e ./vllm --torch-backend=auto
uv pip install --python .venv-vllm/bin/python ninja          # (only needed if any JIT path runs)
uv pip uninstall --python .venv-vllm/bin/python flashinfer-python   # force Triton MoE + FLASH_ATTN
```

## Working run recipe (verified: loads Laguna-XS.2, generates, `SMOKE OK`)
```bash
CUDA_HOME=/usr/local/cuda-12.8 \
VLLM_USE_FLASHINFER_SAMPLER=0 \
VLLM_ATTENTION_BACKEND=FLASH_ATTN \
.venv-vllm/bin/python <script>.py
```
Notes:
- **Run scripts from a non-`vllm/`-containing cwd** (put the script in `/tmp` or set
  PYTHONPATH) or `import vllm` resolves to the bare `vllm/` submodule dir
  (namespace pkg) and `from vllm import LLM` fails with "unknown location".
- Laguna is an **MoE** model → its experts run on Triton fused-MoE (flashinfer's
  cutlass MoE is the path that fails to JIT).
- `gpu_memory_utilization` must fit current free HBM (other GPU jobs compete).
- For `kv_cache_dtype=int4_kivi`, do NOT pin `VLLM_ATTENTION_BACKEND` — let the
  selector pick the custom Int4Kivi backend (chosen by `supported_kv_cache_dtypes`).
