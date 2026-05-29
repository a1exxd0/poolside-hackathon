"""INT4-vs-NVFP4 KV-cache quantization sweep on Laguna-XS.2.

For each layer's real K and V activations, measures reconstruction RMSE across
the grid  {nvfp4, int4} × {headdim-block, per-channel-block} × {absmax, mse}.

The baseline cell (nvfp4 / headdim / absmax) is what vLLM's NVFP4 KV kernel ships
today. Every cell has identical memory (4-bit data + one scale per 16 elements),
so this isolates reconstruction quality. K and V are reported separately to
expose any KIVI-style asymmetry (per-channel K, per-token V).

Usage:
    python -m scripts.quant_sweep [--max-new 200] [--n-alphas 32]
"""
from __future__ import annotations

import argparse
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "/home/alex/poolside-hackathon-kv-quant")
from kv_quant import BLOCK, SWEEP_CELLS, SWEEP_BASELINE, sweep_tensor

MODEL = "poolside/Laguna-XS.2"
PROMPT = (
    "Solve step by step. A train leaves city A at 60 km/h. Two hours later a second "
    "train leaves the same station on the same track at 90 km/h. How many hours after "
    "the second train departs will it catch up to the first train? Show your reasoning."
)


def _cell_label(cell) -> str:
    return f"{cell[0]:<6} {cell[1]:<8} {cell[2]:<6}"


def _print_table(title: str, abs_rmse: dict, ratio: dict) -> None:
    base = SWEEP_BASELINE
    print(f"\n{title}")
    print(f"  {'format':<6} {'layout':<8} {'calib':<6} {'avg RMSE':>10} {'vs vLLM':>10}")
    print(f"  {'-'*6} {'-'*8} {'-'*6} {'-'*10} {'-'*10}")
    for c in [base] + [c for c in SWEEP_CELLS if c != base]:
        tag = "baseline" if c == base else f"{(1.0 - ratio[c]) * 100.0:+.1f}%"
        print(f"  {_cell_label(c)} {abs_rmse[c]:>10.5f} {tag:>10}")
    best = min(SWEEP_CELLS, key=lambda c: ratio[c])
    print(f"  best: {best[0]}/{best[1]}/{best[2]}  "
          f"({(1.0 - ratio[best]) * 100.0:+.1f}% vs baseline)")


def run_sweep(per_layer_kv, n_alphas, device):
    """per_layer_kv: list of (K, V) [n_kv, seq, head_dim]. Returns means + ratios."""
    base = SWEEP_BASELINE
    k_abs = {c: [] for c in SWEEP_CELLS}
    k_rat = {c: [] for c in SWEEP_CELLS}
    v_abs = {c: [] for c in SWEEP_CELLS}
    v_rat = {c: [] for c in SWEEP_CELLS}
    n_used = 0
    for K, V in per_layer_kv:
        if K.shape[1] < BLOCK:
            continue
        n_used += 1
        ks = sweep_tensor(K.to(device), n_alphas)
        vs = sweep_tensor(V.to(device), n_alphas)
        for c in SWEEP_CELLS:
            k_abs[c].append(ks[c]); k_rat[c].append(ks[c] / max(ks[base], 1e-12))
            v_abs[c].append(vs[c]); v_rat[c].append(vs[c] / max(vs[base], 1e-12))
    mean = lambda d: {c: sum(x) / max(len(x), 1) for c, x in d.items()}
    return n_used, mean(k_abs), mean(k_rat), mean(v_abs), mean(v_rat)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--n-alphas", type=int, default=32)
    args = ap.parse_args()

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    cfg = model.config
    print(f"[load] layers={cfg.num_hidden_layers} kv_heads={cfg.num_key_value_heads} "
          f"head_dim={getattr(cfg, 'head_dim', None)} device={device}", flush=True)

    msgs = [{"role": "user", "content": PROMPT}]
    input_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    ).to(device)

    cache = DynamicCache()
    with torch.no_grad():
        model.generate(input_ids, max_new_tokens=args.max_new,
                        past_key_values=cache, use_cache=True, do_sample=False)

    per_layer = [(layer.keys[0].float().cpu(), layer.values[0].float().cpu())
                 for layer in cache.layers]
    seq = per_layer[0][0].shape[1] if per_layer else 0
    print(f"[seq] {seq} tokens cached across {len(per_layer)} layers", flush=True)

    n_used, k_abs, k_rat, v_abs, v_rat = run_sweep(per_layer, args.n_alphas, device)

    print(f"\n{'='*56}")
    print(f"INT4 vs NVFP4 KV sweep  —  {n_used} layers, seq cropped to /{BLOCK}")
    print("All cells: identical memory (4-bit + 1 scale per 16 elems, ~3.56x vs BF16).")
    _print_table("KEY cache (avg RMSE over layers):", k_abs, k_rat)
    _print_table("VALUE cache (avg RMSE over layers):", v_abs, v_rat)


if __name__ == "__main__":
    main()
