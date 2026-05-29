"""Budget sweep for TriAttention on Laguna-XS.2 (loads + calibrates once).

Loads the 63 GB model a single time, calibrates pre-RoPE Q/K stats, decodes a
long-reasoning prompt once with a full cache, then re-decodes under several
(budget, beta) compression settings — reporting full-attention-layer KV
reduction and token agreement vs. the baseline for each.

Usage:
    uv run python -m scripts.sweep_laguna [--max-new 700]
"""

from __future__ import annotations

import argparse
import time

import torch

from triattention import collect_calibration, generate
from scripts._common import CALIBRATION_TEXTS, load_model

MODEL = "poolside/Laguna-XS.2"

# A prompt that elicits a long chain of reasoning so the cache actually grows
# past small budgets.
PROMPT = (
    "Prove carefully, step by step, that the square root of 2 is irrational. "
    "Then, as a separate problem, find all integer solutions to the equation "
    "x^2 - 3 y^2 = 1 with 0 < x < 50, showing your work for each. Be thorough."
)

CONFIGS = [  # (budget, beta)
    (512, 128),
    (256, 64),
    (128, 64),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=700)
    ap.add_argument("--sink", type=int, default=4)
    args = ap.parse_args()

    model, tok, device = load_model(MODEL, dtype=torch.bfloat16)
    cfg = model.config
    full = [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]
    print(f"[load] {MODEL} | {cfg.num_hidden_layers} layers, {len(full)} full-attention, "
          f"kv_heads={cfg.num_key_value_heads} head_dim={cfg.head_dim}", flush=True)

    t0 = time.time()
    stats = collect_calibration(model, tok, CALIBRATION_TEXTS, max_length=512,
                                n_dominant=2, device=device)
    s0 = stats.layers[0]
    print(f"[calib] rotary_dim={s0.rotary_dim} bands={s0.omega.numel()} "
          f"R={s0.R.mean():.3f} group_size={stats.group_size} ({time.time()-t0:.1f}s)", flush=True)

    msgs = [{"role": "user", "content": PROMPT}]
    input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                        return_tensors="pt", return_dict=False).to(device)
    print(f"[prompt] {input_ids.shape[1]} tokens", flush=True)

    t0 = time.time()
    base = generate(model, input_ids, compress=False, max_new_tokens=args.max_new,
                    eos_token_id=cfg.eos_token_id)
    bg = base.sequences[0, input_ids.shape[1]:]
    print(f"[baseline] generated={base.num_generated} peak_full_kv={base.peak_kv_len} "
          f"({time.time()-t0:.1f}s)", flush=True)

    print("\n========== BASELINE transcript (head) ==========", flush=True)
    print(tok.decode(bg, skip_special_tokens=True)[:900], flush=True)

    for budget, beta in CONFIGS:
        t0 = time.time()
        comp = generate(model, input_ids, stats=stats, compress=True, budget=budget,
                        beta=beta, sink=args.sink, max_new_tokens=args.max_new,
                        eos_token_id=cfg.eos_token_id)
        cg = comp.sequences[0, input_ids.shape[1]:]
        n = min(len(bg), len(cg))
        agree = (bg[:n] == cg[:n]).float().mean().item() if n else 0.0
        prefix = 0
        for i in range(n):
            if bg[i] == cg[i]:
                prefix += 1
            else:
                break
        red = base.peak_kv_len / max(comp.peak_kv_len, 1)
        print(f"\n[budget={budget} beta={beta}] peak_full_kv {base.peak_kv_len}->{comp.peak_kv_len} "
              f"({red:.2f}x) compressions={comp.num_compressions} gen={comp.num_generated} "
              f"agree={agree:.1%} prefix={prefix}/{n} ({time.time()-t0:.1f}s)", flush=True)
        print(f"  transcript(head): {tok.decode(cg, skip_special_tokens=True)[:500]!r}", flush=True)


if __name__ == "__main__":
    main()
