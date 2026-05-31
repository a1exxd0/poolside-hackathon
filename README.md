# 4-bit KV Cache Quantization (Laguna-XS.2)

In submission of Poolside AI's first hackathon, focussed on optimizing for the Laguna XS.2 with generalizability.

## Work Completed

4-bit quantization for all kv-caches on the Laguna XS.2 NVFP4 variant, with NVFP4 weights on experts but FP16 on attention. We found a slight regression after quantizing, and another after multiplying RoPE factor by 4. Future work would fine tune the model under larger contexts (although this is very difficult, especially for a 3bn active param model).

Block-based quantization on k=16 with custom kernels for k=16 vLLM default. Future work would generalise this too.

## Layout

- `kv_quant.py` — quantization primitives: formats (INT4 / INT3 / NVFP4-e2m1),
  block layouts (per-head-dim, per-channel), absmax / MSE-optimal calibration,
  the `{format} × {layout} × {calib}` sweep, and `roundtrip`.
- `scripts/quant_sweep.py` — reconstruction-RMSE grid over real KV activations.
- `scripts/quant_ab.py` — frozen-page teacher-forced KL A/B vs BF16 (incl. INT3).
- `scripts/quant_longctx.py` — long-context KL-by-position via an SDPA patch.
- `PROBLEM.md` — full analysis, vLLM kernel notes, cost/benefit.

## Run

```bash
uv sync
.venv/bin/python scripts/quant_sweep.py     # RMSE grid
.venv/bin/python scripts/quant_ab.py        # downstream KL A/B
.venv/bin/python scripts/quant_longctx.py   # long-context trend
```

Scripts target `poolside/Laguna-XS.2` and run on the Blackwell node.
