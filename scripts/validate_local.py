"""End-to-end local validation of TriAttention on a small reasoning model.

Calibrates pre-RoPE Q/K statistics, then greedy-decodes the same reasoning
prompt twice — once with a full KV cache (baseline) and once with TriAttention
compression — and reports KV-memory reduction plus output agreement.

Usage:
    uv run python -m scripts.validate_local [--budget 256] [--beta 64] [--max-new 400]
"""

from __future__ import annotations

import argparse
import time

import torch

from triattention import collect_calibration, generate
from scripts._common import CALIBRATION_TEXTS, load_model

PROMPT = (
    "Solve step by step. A train leaves city A at 60 km/h. Two hours later a second "
    "train leaves the same station on the same track at 90 km/h. How many hours after "
    "the second train departs will it catch up to the first train? Show your reasoning."
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--beta", type=int, default=64)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    model, tok, device = load_model(args.model) if args.model else load_model()
    print(f"[load] model on {device} | q_heads={model.config.num_attention_heads} "
          f"kv_heads={model.config.num_key_value_heads} head_dim="
          f"{getattr(model.config, 'head_dim', None) or model.config.hidden_size // model.config.num_attention_heads}")

    t0 = time.time()
    stats = collect_calibration(model, tok, CALIBRATION_TEXTS, max_length=512, n_dominant=2, device=device)
    print(f"[calib] {len(stats.layers)} layers, ~{stats.num_tokens} tokens/layer, "
          f"R={stats.layers[0].R.mean():.3f} (mean concentration), took {time.time()-t0:.1f}s")

    msgs = [{"role": "user", "content": PROMPT}]
    input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
    print(f"[prompt] {input_ids.shape[1]} tokens")

    t0 = time.time()
    base = generate(model, input_ids, compress=False, max_new_tokens=args.max_new,
                    eos_token_id=tok.eos_token_id)
    base_t = time.time() - t0

    t0 = time.time()
    comp = generate(model, input_ids, stats=stats, compress=True, budget=args.budget,
                    beta=args.beta, sink=args.sink, max_new_tokens=args.max_new,
                    eos_token_id=tok.eos_token_id)
    comp_t = time.time() - t0

    # token-level agreement on the overlapping generated region
    bg = base.sequences[0, input_ids.shape[1]:]
    cg = comp.sequences[0, input_ids.shape[1]:]
    n = min(len(bg), len(cg))
    agree = (bg[:n] == cg[:n]).float().mean().item() if n else 0.0
    prefix = 0
    for i in range(n):
        if bg[i] == cg[i]:
            prefix += 1
        else:
            break

    print("\n========== BASELINE (full cache) ==========")
    print(tok.decode(bg, skip_special_tokens=True)[:1200])
    print(f"\n[baseline] generated={base.num_generated} peak_kv={base.peak_kv_len} time={base_t:.1f}s")

    print("\n========== TriAttention (compressed) ==========")
    print(tok.decode(cg, skip_special_tokens=True)[:1200])
    print(f"\n[compressed] generated={comp.num_generated} peak_kv={comp.peak_kv_len} "
          f"compressions={comp.num_compressions} time={comp_t:.1f}s")

    print("\n========== SUMMARY ==========")
    print(f"KV peak: {base.peak_kv_len} -> {comp.peak_kv_len} "
          f"({base.peak_kv_len / max(comp.peak_kv_len,1):.2f}x reduction)")
    print(f"token agreement (overlap): {agree:.1%} | identical prefix length: {prefix}/{n}")


if __name__ == "__main__":
    main()
