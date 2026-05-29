# 4-bit KV Cache Quantization (Laguna-XS.2)

A study of low-bit KV-cache quantization for vLLM. Full writeup, findings, and
implementation notes are in **[PROBLEM.md](PROBLEM.md)**.

## Finding

The lever for 4-bit KV is the block **layout**, not the number format or the
calibration. Uniform **INT4 with per-channel-K (KIVI) blocking** — a 16-token
block per channel for K, per-token for V — beats vLLM's shipped
NVFP4 / head_dim / absmax baseline by ~25% on K reconstruction error at identical
memory (3.56× vs BF16), and the advantage **grows with context** (neutral below
~512 tokens, ~20–25% KL reduction beyond 1k). 4-bit is the quality floor; INT3
costs ~2–3× the distortion. The catch: per-channel blocking can't ride NVFP4's
hardware microscale, so capturing it needs a software INT4 KV path.

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
