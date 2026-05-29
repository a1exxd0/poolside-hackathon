"""Accuracy analysis for INT4 KV cache quantization on Laguna-XS.2.

Two measurements:
  1. Per-layer reconstruction RMSE (all 40 layers, absmax vs MSE-optimal).
  2. Token agreement: generate with BF16 cache vs INT4-simulated cache
     (quantize+dequantize each layer's KV in-place at every decode step).

Usage:
    python -m scripts.quant_accuracy [--max-new 200]
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "/home/alex/poolside-hackathon-kv-quant")
from kv_quant import (
    BLOCK, PAGE, QMAX,
    _absmax_scale, _mse_optimal_scale,
    quantize_block, dequantize_block,
    quantize_page, QuantizedKVCache, measure_page_error,
)

MODEL = "poolside/Laguna-XS.2"
PROMPT = (
    "Solve step by step. A train leaves city A at 60 km/h. Two hours later a second "
    "train leaves the same station on the same track at 90 km/h. How many hours after "
    "the second train departs will it catch up to the first train? Show your reasoning."
)


def layer_rmse(k: torch.Tensor, use_mse: bool) -> float:
    """RMSE of INT4 round-trip on one layer's key cache [n_kv, seq, head_dim]."""
    kf = k.float().reshape(*k.shape[:-1], k.shape[-1] // BLOCK, BLOCK)
    scale_fn = _mse_optimal_scale if use_mse else _absmax_scale
    s = scale_fn(kf) if not use_mse else scale_fn(kf)
    q = quantize_block(kf, s)
    khat = dequantize_block(q, s)
    return ((kf - khat) ** 2).mean().sqrt().item()


def simulate_int4_generation(model, input_ids, tok, max_new: int) -> list[int]:
    """Generate greedily; after each step, quantize+dequantize every layer's KV."""
    cache = DynamicCache()
    device = input_ids.device

    with torch.no_grad():
        # Prefill
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                    cache_position=torch.arange(input_ids.shape[1], device=device),
                    position_ids=torch.arange(input_ids.shape[1], device=device).unsqueeze(0))
        _quantize_cache_inplace(cache)
        tokens = [out.logits[0, -1].argmax().item()]
        abs_pos = input_ids.shape[1]

        for _ in range(max_new - 1):
            cp = torch.tensor([abs_pos], device=device)
            out = model(input_ids=torch.tensor([[tokens[-1]]], device=device),
                        past_key_values=cache, use_cache=True,
                        cache_position=cp,
                        position_ids=cp.unsqueeze(0))
            _quantize_cache_inplace(cache)
            t = out.logits[0, -1].argmax().item()
            tokens.append(t)
            abs_pos += 1
            eos = getattr(model.config, "eos_token_id", None)
            eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos else set())
            if t in eos_set:
                break

    return tokens


def _quantize_cache_inplace(cache: DynamicCache) -> None:
    """Quantize+dequantize every layer's keys and values in the cache (INT4 simulation)."""
    for layer in cache.layers:
        layer.keys  = _round_trip(layer.keys)
        layer.values = _round_trip(layer.values)


def _round_trip(x: torch.Tensor) -> torch.Tensor:
    """Quantize x [B, n_heads, seq, head_dim] to INT4 and dequantize back."""
    B, H, S, D = x.shape
    xf = x.float().reshape(B, H, S, D // BLOCK, BLOCK)
    s = _mse_optimal_scale(xf)
    q = quantize_block(xf, s)
    return dequantize_block(q, s).reshape(B, H, S, D).to(x.dtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    msgs = [{"role": "user", "content": PROMPT}]
    input_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    ).to(device)
    print(f"[prompt] {input_ids.shape[1]} tokens\n")

    # ── 1. BF16 baseline generation ──────────────────────────────────────────
    print("[run] BF16 baseline ...", flush=True)
    cache_bf16 = DynamicCache()
    t0 = time.time()
    with torch.no_grad():
        out_bf16 = model.generate(
            input_ids, max_new_tokens=args.max_new,
            past_key_values=cache_bf16, use_cache=True, do_sample=False,
        )
    bf16_t = time.time() - t0
    bf16_tokens = out_bf16[0, input_ids.shape[1]:].tolist()

    # ── 2. Per-layer reconstruction stats ────────────────────────────────────
    print("\n[stats] Per-layer key-cache RMSE (absmax vs MSE-optimal)")
    print(f"  {'layer':>5}  {'absmax RMSE':>12}  {'opt RMSE':>10}  {'reduction':>10}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*10}  {'─'*10}")
    all_abs, all_opt = [], []
    for i, layer in enumerate(cache_bf16.layers):
        k = layer.keys[0].float().cpu()
        kf = k.reshape(*k.shape[:-1], k.shape[-1] // BLOCK, BLOCK)
        s_abs = _absmax_scale(kf)
        s_opt = _mse_optimal_scale(kf)
        rmse_abs = ((kf - dequantize_block(quantize_block(kf, s_abs), s_abs)) ** 2).mean().sqrt().item()
        rmse_opt = ((kf - dequantize_block(quantize_block(kf, s_opt), s_opt)) ** 2).mean().sqrt().item()
        all_abs.append(rmse_abs)
        all_opt.append(rmse_opt)
        pct = 100.0 * (rmse_abs - rmse_opt) / max(rmse_abs, 1e-12)
        print(f"  {i:>5}  {rmse_abs:>12.6f}  {rmse_opt:>10.6f}  {pct:>9.1f}%")
    avg_abs = sum(all_abs) / len(all_abs)
    avg_opt = sum(all_opt) / len(all_opt)
    avg_red = 100.0 * (avg_abs - avg_opt) / max(avg_abs, 1e-12)
    print(f"  {'avg':>5}  {avg_abs:>12.6f}  {avg_opt:>10.6f}  {avg_red:>9.1f}%")

    # ── 3. INT4-simulated generation ─────────────────────────────────────────
    print(f"\n[run] INT4-simulated generation (quant+dequant each step) ...", flush=True)
    t0 = time.time()
    int4_tokens = simulate_int4_generation(model, input_ids, tok, args.max_new)
    int4_t = time.time() - t0

    # ── 4. Token agreement ────────────────────────────────────────────────────
    n = min(len(bf16_tokens), len(int4_tokens))
    agree = sum(a == b for a, b in zip(bf16_tokens[:n], int4_tokens[:n]))
    prefix = 0
    for a, b in zip(bf16_tokens, int4_tokens):
        if a != b:
            break
        prefix += 1

    print("\n========== BF16 OUTPUT ==========")
    print(tok.decode(bf16_tokens, skip_special_tokens=True)[:800])
    print("\n========== INT4-SIMULATED OUTPUT ==========")
    print(tok.decode(int4_tokens, skip_special_tokens=True)[:800])

    print("\n========== ACCURACY SUMMARY ==========")
    print(f"  tokens compared:        {n}")
    print(f"  token agreement:        {agree}/{n}  ({100*agree/max(n,1):.1f}%)")
    print(f"  identical prefix:       {prefix} tokens")
    print(f"  avg key RMSE (absmax):  {avg_abs:.6f}")
    print(f"  avg key RMSE (optimal): {avg_opt:.6f}  ({avg_red:.1f}% reduction)")
    print(f"  BF16 time:  {bf16_t:.1f}s   INT4-sim time: {int4_t:.1f}s")


if __name__ == "__main__":
    main()
