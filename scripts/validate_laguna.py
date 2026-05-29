"""End-to-end validation of TriAttention on the real Laguna-XS.2 model.

Laguna-XS.2 is a mixed-attention MoE (10 full-attention layers among 40, partial
RoPE 0.5 with YaRN, per-head q/k norm). TriAttention compresses only the 10
full-attention layers; the 30 sliding-window layers are bounded by the cache
itself. This script calibrates pre-RoPE Q/K stats, then greedy-decodes a
reasoning prompt with a full cache (baseline) and with compression, reporting
full-layer KV reduction and output agreement.

Usage:
    uv run python -m scripts.validate_laguna [--budget 512] [--beta 128] [--max-new 512]
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
    ap.add_argument("--max-new", type=int, default=512)
    args = ap.parse_args()

    model, tok, device = load_model(MODEL, dtype=torch.bfloat16)
    cfg = model.config
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    print(f"[load] {MODEL} on {device} | {cfg.num_hidden_layers} layers "
          f"({len(full)} full-attention: {full}) | kv_heads={cfg.num_key_value_heads} "
          f"head_dim={cfg.head_dim}")

    t0 = time.time()
    stats = collect_calibration(model, tok, CALIBRATION_TEXTS, max_length=512,
                                n_dominant=2, device=device)
    s0 = stats.layers[0]
    print(f"[calib] {len(stats.layers)} full layers, rotary_dim={s0.rotary_dim}, "
          f"omega_bands={s0.omega.numel()}, ~{stats.num_tokens} tok/layer, "
          f"R={s0.R.mean():.3f} (concentration), group_size={stats.group_size}, "
          f"took {time.time()-t0:.1f}s")

    msgs = [{"role": "user", "content": PROMPT}]
    input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=False).to(device)
    print(f"[prompt] {input_ids.shape[1]} tokens")

    t0 = time.time()
    base = generate(model, input_ids, compress=False, max_new_tokens=args.max_new,
                    eos_token_id=cfg.eos_token_id)
    base_t = time.time() - t0

    t0 = time.time()
    comp = generate(model, input_ids, stats=stats, compress=True, budget=args.budget,
                    beta=args.beta, sink=args.sink, max_new_tokens=args.max_new,
                    eos_token_id=cfg.eos_token_id)
    comp_t = time.time() - t0

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
    print(tok.decode(bg, skip_special_tokens=True)[:1500])
    print(f"\n[baseline] generated={base.num_generated} peak_full_kv={base.peak_kv_len} time={base_t:.1f}s")

    print("\n========== TriAttention (compressed) ==========")
    print(tok.decode(cg, skip_special_tokens=True)[:1500])
    print(f"\n[compressed] generated={comp.num_generated} peak_full_kv={comp.peak_kv_len} "
          f"compressions={comp.num_compressions} time={comp_t:.1f}s")

    print("\n========== SUMMARY ==========")
    print(f"full-layer KV peak: {base.peak_kv_len} -> {comp.peak_kv_len} "
          f"({base.peak_kv_len / max(comp.peak_kv_len,1):.2f}x reduction)")
    print(f"token agreement (overlap): {agree:.1%} | identical prefix length: {prefix}/{n}")


if __name__ == "__main__":
    main()
