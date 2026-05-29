"""Benchmark INT4 KV-cache quantization with and without TriAttention compression.

Runs two inference passes on Laguna-XS.2:
  1. Baseline     (compress=False, quant_cache=True): full KV cache + INT4 quant
  2. TriAttn+quant (compress=True, quant_cache=True): TriAttention eviction + INT4 quant

Reports per-run stats and a summary table.

Usage:
    python -m scripts.quant_inference [--budget 512] [--beta 128] [--sink 4] [--max-new 256]
"""

from __future__ import annotations

import argparse
import time

import torch

from triattention import collect_calibration, generate
from scripts._common import CALIBRATION_TEXTS, load_model

MODEL = "poolside/Laguna-XS.2"

PROMPT = (
    "Solve step by step. A train leaves city A at 60 km/h. Two hours later a second "
    "train leaves the same station on the same track at 90 km/h. How many hours after "
    "the second train departs will it catch up to the first train? Show your reasoning."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=512)
    ap.add_argument("--beta", type=int, default=128)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=256)
    args = ap.parse_args()

    model, tok, device = load_model(MODEL, dtype=torch.bfloat16)
    cfg = model.config
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    print(f"[load] {MODEL} on {device} | {cfg.num_hidden_layers} layers "
          f"({len(full)} full-attention) | kv_heads={cfg.num_key_value_heads} "
          f"head_dim={cfg.head_dim}")

    t0 = time.time()
    stats = collect_calibration(model, tok, CALIBRATION_TEXTS, max_length=512,
                                n_dominant=2, device=device)
    print(f"[calib] {len(stats.layers)} full layers, took {time.time()-t0:.1f}s")

    msgs = [{"role": "user", "content": PROMPT}]
    input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=False).to(device)
    print(f"[prompt] {input_ids.shape[1]} tokens")

    # --- Baseline: full cache + quant measurement ---
    print("\n[run] baseline (compress=False, quant_cache=True) ...")
    t0 = time.time()
    base = generate(model, input_ids, compress=False, quant_cache=True,
                    max_new_tokens=args.max_new, eos_token_id=cfg.eos_token_id)
    base_t = time.time() - t0

    base_quant_mb = base.kv_quant_bytes[-1] / 1e6 if base.kv_quant_bytes else 0.0
    base_bf16_mb  = base.kv_bf16_bytes[-1]  / 1e6 if base.kv_bf16_bytes  else 0.0
    base_ratio    = base_bf16_mb / base_quant_mb if base_quant_mb > 0 else float("nan")

    print("\n========== BASELINE output ==========")
    bg = base.sequences[0, input_ids.shape[1]:]
    print(tok.decode(bg, skip_special_tokens=True)[:800])
    print(f"\n[baseline] generated={base.num_generated} peak_kv={base.peak_kv_len} "
          f"quant={base_quant_mb:.1f}MB bf16={base_bf16_mb:.1f}MB "
          f"ratio={base_ratio:.2f}x time={base_t:.1f}s")

    # --- TriAttention + quant ---
    print("\n[run] TriAttn+quant (compress=True, quant_cache=True) ...")
    t0 = time.time()
    comp = generate(model, input_ids, stats=stats, compress=True, quant_cache=True,
                    budget=args.budget, beta=args.beta, sink=args.sink,
                    max_new_tokens=args.max_new, eos_token_id=cfg.eos_token_id)
    comp_t = time.time() - t0

    comp_quant_mb = comp.kv_quant_bytes[-1] / 1e6 if comp.kv_quant_bytes else 0.0
    comp_bf16_mb  = comp.kv_bf16_bytes[-1]  / 1e6 if comp.kv_bf16_bytes  else 0.0
    comp_ratio    = comp_bf16_mb / comp_quant_mb if comp_quant_mb > 0 else float("nan")

    print("\n========== TriAttn+quant output ==========")
    cg = comp.sequences[0, input_ids.shape[1]:]
    print(tok.decode(cg, skip_special_tokens=True)[:800])
    print(f"\n[TriAttn+quant] generated={comp.num_generated} peak_kv={comp.peak_kv_len} "
          f"compressions={comp.num_compressions} "
          f"quant={comp_quant_mb:.1f}MB bf16={comp_bf16_mb:.1f}MB "
          f"ratio={comp_ratio:.2f}x time={comp_t:.1f}s")

    # --- Summary table ---
    print("\n========== SUMMARY ==========")
    print(f"{'Mode':<16} | {'peak_kv':>7} | {'quant_MB':>8} | {'bf16_MB':>8} | {'ratio':>6}")
    print(f"{'-'*16}-+-{'-'*7}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")
    print(f"{'baseline':<16} | {base.peak_kv_len:>7} | {base_quant_mb:>8.1f} | {base_bf16_mb:>8.1f} | {base_ratio:>5.2f}x")
    print(f"{'TriAttn+quant':<16} | {comp.peak_kv_len:>7} | {comp_quant_mb:>8.1f} | {comp_bf16_mb:>8.1f} | {comp_ratio:>5.2f}x")


if __name__ == "__main__":
    main()
