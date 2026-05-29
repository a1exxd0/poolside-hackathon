# kv-quant

Page-aligned 4-bit KV cache with MSE-optimal blockwise scaling for vLLM.

See [PROBLEM.md](PROBLEM.md) for the full design, empirical results, and vLLM
implementation plan.

## What's here

```
kv_quant.py          Python prototype — QuantizedKVCache, MSE-optimal scale calibration
scripts/
  quant_inference.py   Run Laguna-XS.2, snapshot KV cache, report INT4 vs BF16 memory
  quant_accuracy.py    Per-layer RMSE + token agreement (BF16 vs INT4-simulated)
  humaneval_bench.py   HumanEval subset: pass@1 for BF16 and INT4-simulated KV
vllm/                vLLM submodule (see PROBLEM.md §vLLM Implementation Plan)
```

## Setup

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu128
uv pip install transformers accelerate datasets numpy
```

## Run

```bash
# Memory benchmark
python -m scripts.quant_inference

# Accuracy (per-layer RMSE + token agreement)
python -m scripts.quant_accuracy

# HumanEval subset (BF16 vs INT4, 20 problems)
python -m scripts.humaneval_bench --n 20
```

## Key results (Laguna-XS.2, B300)

| | |
|--|--|
| Memory reduction | **3.05×** vs BF16 (theoretical 3.56×) |
| RMSE improvement | **4.2%** MSE-optimal over absmax, uniform across all 40 layers |
| HumanEval pass@1 | **95%** BF16, **95%** INT4 — **100% agreement** |
| Reasoning agreement | 75% token-level, 96-token identical prefix |
