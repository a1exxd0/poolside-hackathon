"""INT4 KV-cache quantization benchmark on Laguna-XS.2.

Runs standard greedy generation, then snapshots the final KV cache, quantizes
it with MSE-optimal blockwise scaling, and reports memory savings vs BF16.
Also measures per-layer reconstruction quality (absmax vs MSE-optimal scale).

Usage:
    python -m scripts.quant_inference [--max-new 256] [--prompt "..."]
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "/home/alex/poolside-hackathon-kv-quant")
from kv_quant import QuantizedKVCache, measure_page_error

MODEL = "poolside/Laguna-XS.2"
PROMPT = (
    "Solve step by step. A train leaves city A at 60 km/h. Two hours later a second "
    "train leaves the same station on the same track at 90 km/h. How many hours after "
    "the second train departs will it catch up to the first train? Show your reasoning."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--prompt", type=str, default=PROMPT)
    args = ap.parse_args()

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    cfg = model.config
    print(f"[load] {cfg.num_hidden_layers} layers, kv_heads={cfg.num_key_value_heads}, "
          f"head_dim={cfg.head_dim}, device={next(model.parameters()).device}")

    msgs = [{"role": "user", "content": args.prompt}]
    input_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    ).to(next(model.parameters()).device)
    print(f"[prompt] {input_ids.shape[1]} tokens", flush=True)

    # --- Generate ---
    cache = DynamicCache()
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=args.max_new,
            past_key_values=cache,
            use_cache=True,
            do_sample=False,
        )
    gen_t = time.time() - t0
    n_gen = out.shape[1] - input_ids.shape[1]

    print("\n========== OUTPUT ==========")
    print(tok.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)[:1200])

    # --- Measure KV cache memory ---
    n_layers = len(cache.layers)
    bf16_bytes = sum(
        (layer.keys.numel() + layer.values.numel()) * 2
        for layer in cache.layers
    )
    seq_len = cache.layers[0].keys.shape[-2] if n_layers else 0

    qcache = QuantizedKVCache()
    for i, layer in enumerate(cache.layers):
        qcache.update(i, layer.keys, layer.values)
    int4_bytes = qcache.mem_bytes()

    # Reconstruction quality on the first full-attention layer (index depends on model).
    # Laguna has 10 full-attention layers; sample the first one.
    sample_layer_idx = 0
    sample_k = cache.layers[sample_layer_idx].keys[0]   # [n_kv, seq, head_dim]
    err = measure_page_error(sample_k.float().cpu())

    bf16_mb = bf16_bytes / 1e6
    int4_mb = int4_bytes / 1e6
    ratio = bf16_bytes / max(int4_bytes, 1)

    print("\n========== KV CACHE ==========")
    print(f"  layers:          {n_layers}")
    print(f"  seq_len:         {seq_len}")
    print(f"  BF16:            {bf16_mb:.1f} MB")
    print(f"  INT4 (optimal):  {int4_mb:.1f} MB")
    print(f"  ratio:           {ratio:.2f}x")
    print(f"  absmax MSE:      {err['absmax_mse']:.6f}")
    print(f"  optimal MSE:     {err['optimal_mse']:.6f}")
    print(f"  MSE reduction:   {err['reduction_pct']:.1f}%  (MSE-optimal vs absmax)")
    print(f"\n  generated {n_gen} tokens in {gen_t:.1f}s  ({n_gen/gen_t:.1f} tok/s)")


if __name__ == "__main__":
    main()
