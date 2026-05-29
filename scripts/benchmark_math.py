"""MATH-500 accuracy + KV-memory benchmark: full-cache baseline vs TriAttention budgets.

Loads Laguna-XS.2 + calibrates once, then for a random subset of problems
greedy-decodes the full-cache baseline and each compressed budget, extracts the
``\\boxed{}`` final answer, and grades it. For every (problem, config) it stores
the full transcript and the per-decode-step KV-memory high-water, and reports
p50/p90/p99/max memory percentiles per config plus accuracy and mean
full-attention-layer KV reduction. Everything is written to a JSON results file.

Note on b2048: eviction only fires once stored keys exceed the budget. On
MATH-500 the prompt+generation rarely exceeds ~2048 tokens, so b2048 typically
never triggers and is identical to the baseline -- the per-problem peak full-KV
length printed below makes that gap explicit.

Usage:
    uv run python -m scripts.benchmark_math [--limit 5] [--seed 0] [--max-new 2048]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time

import numpy as np
import torch
from datasets import load_dataset

from triattention import collect_calibration, generate
from scripts._common import CALIBRATION_TEXTS, load_model

MODEL = "poolside/Laguna-XS.2"
CONFIGS = [("baseline", None, None), ("b2048", 2048, 128), ("b512", 512, 128), ("b256", 256, 64)]

INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Put your final answer inside \\boxed{}.\n\n"
)


def extract_boxed(text: str) -> str | None:
    """Return the content of the LAST ``\\boxed{...}`` (brace-balanced), or None."""
    idx = text.rfind("\\boxed")
    if idx == -1:
        return None
    i = text.find("{", idx)
    if i == -1:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def normalize(ans: str | None) -> str:
    """Light LaTeX normalization for robust string comparison."""
    if ans is None:
        return ""
    s = ans.strip()
    for a, b in [("\\left", ""), ("\\right", ""), ("\\!", ""), ("\\,", ""), ("\\;", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\$", ""), ("$", ""),
                 ("\\%", ""), ("%", ""), ("^{\\circ}", ""), ("^\\circ", ""), (" ", "")]:
        s = s.replace(a, b)
    s = re.sub(r"\\text\{[^}]*\}", "", s)
    s = re.sub(r"\\mbox\{[^}]*\}", "", s)
    s = s.rstrip(".")
    if s.startswith("{") and s.endswith("}"):
        s = s[1:-1]
    return s


def grade(pred: str | None, gold: str) -> bool:
    np_, ng = normalize(pred), normalize(gold)
    if np_ == ng and np_ != "":
        return True
    try:
        return abs(float(np_) - float(ng)) < 1e-6
    except (ValueError, TypeError):
        return False


def pct(series: list[int]) -> dict:
    """Percentiles of a per-step byte series, reported in MiB."""
    if not series:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    a = np.asarray(series, dtype=np.float64) / (1024 * 1024)
    return {
        "p50": round(float(np.percentile(a, 50)), 3),
        "p90": round(float(np.percentile(a, 90)), 3),
        "p99": round(float(np.percentile(a, 99)), 3),
        "max": round(float(a.max()), 3),
        "mean": round(float(a.mean()), 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-new", type=int, default=2048)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    full_ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    idxs = sorted(random.Random(args.seed).sample(range(len(full_ds)), args.limit))
    ds = full_ds.select(idxs)
    model, tok, device = load_model(MODEL, dtype=torch.bfloat16)
    print(f"[load] {MODEL} | random MATH-500 subset seed={args.seed} idx={idxs} | "
          f"configs={[c[0] for c in CONFIGS]} max_new={args.max_new}", flush=True)

    stats = collect_calibration(model, tok, CALIBRATION_TEXTS, max_length=512,
                                n_dominant=2, device=device)
    print(f"[calib] R={stats.layers[0].R.mean():.3f} rotary_dim={stats.layers[0].rotary_dim} "
          f"full_layers={len(stats.layer_indices)}", flush=True)

    # aggregates
    correct = {c[0]: 0 for c in CONFIGS}
    red_sum = {c[0]: 0.0 for c in CONFIGS}
    red_n = {c[0]: 0 for c in CONFIGS}
    mem_full = {c[0]: [] for c in CONFIGS}      # pooled per-step byte series across problems
    mem_slid = {c[0]: [] for c in CONFIGS}
    mem_tot = {c[0]: [] for c in CONFIGS}
    records = []
    t_start = time.time()

    for qi, (gi, ex) in enumerate(zip(idxs, ds)):
        gold = extract_boxed(ex["answer"]) or ex["answer"]
        msgs = [{"role": "user", "content": INSTRUCTION + ex["problem"]}]
        input_ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                            return_tensors="pt", return_dict=False).to(device)
        prompt_len = input_ids.shape[1]
        rec = {"dataset_index": gi, "unique_id": ex["unique_id"], "subject": ex["subject"],
               "level": ex["level"], "problem": ex["problem"], "gold_answer": ex["answer"],
               "gold_extracted": normalize(gold), "prompt_tokens": prompt_len, "configs": {}}
        base_peak = None
        for name, budget, beta in CONFIGS:
            res = generate(model, input_ids, stats=(None if budget is None else stats),
                           compress=budget is not None, budget=budget or 2048, beta=beta or 128,
                           sink=args.sink, max_new_tokens=args.max_new,
                           eos_token_id=model.config.eos_token_id, record_kv=True)
            text = tok.decode(res.sequences[0, prompt_len:], skip_special_tokens=True)
            pred = extract_boxed(text)
            ok = grade(pred, gold)
            correct[name] += ok
            if name == "baseline":
                base_peak = res.peak_kv_len
            else:
                red_sum[name] += base_peak / max(res.peak_kv_len, 1)
                red_n[name] += 1
            mem_full[name] += res.kv_bytes_full
            mem_slid[name] += res.kv_bytes_sliding
            mem_tot[name] += res.kv_bytes_total
            rec["configs"][name] = {
                "budget": budget, "beta": beta, "correct": bool(ok),
                "pred_extracted": normalize(pred), "gen_tokens": res.num_generated,
                "peak_full_kv_len": res.peak_kv_len, "compressions": res.num_compressions,
                "kv_mem_mib": {"full": pct(res.kv_bytes_full),
                               "sliding": pct(res.kv_bytes_sliding),
                               "total": pct(res.kv_bytes_total)},
                "transcript": text,
            }
        records.append(rec)
        peaks = {n: rec["configs"][n]["peak_full_kv_len"] for n, _, _ in CONFIGS}
        print(f"Q{qi} (idx {gi}, {ex['subject'][:10]} L{ex['level']}) prompt={prompt_len} "
              f"gold={normalize(gold)[:20]!r} | peak_full_kv={peaks} | "
              f"correct={ {n: rec['configs'][n]['correct'] for n,_,_ in CONFIGS} } "
              f"| running_acc={ {n: f'{correct[n]}/{qi+1}' for n,_,_ in CONFIGS} } "
              f"({time.time()-t_start:.0f}s)", flush=True)

    summary = {}
    print(f"\n========== SUMMARY ({len(ds)} problems, seed {args.seed}) ==========", flush=True)
    print(f"  {'config':9s} {'acc':>10s} {'fullKVred':>10s} | full-KV MiB p50/p90/p99/max", flush=True)
    for name, _, _ in CONFIGS:
        acc = correct[name] / len(ds)
        red = (red_sum[name] / red_n[name]) if red_n[name] else 1.0
        mf, ms, mt = pct(mem_full[name]), pct(mem_slid[name]), pct(mem_tot[name])
        summary[name] = {"accuracy": acc, "correct": correct[name], "n": len(ds),
                         "mean_full_kv_reduction": round(red, 3),
                         "kv_mem_mib_full": mf, "kv_mem_mib_sliding": ms, "kv_mem_mib_total": mt}
        print(f"  {name:9s} {acc:9.1%} {red:9.2f}x | "
              f"{mf['p50']:.2f}/{mf['p90']:.2f}/{mf['p99']:.2f}/{mf['max']:.2f}  "
              f"(sliding p50={ms['p50']:.2f}, total p50={mt['p50']:.2f})", flush=True)

    out = args.out or f"results/math_bench_seed{args.seed}_n{args.limit}_{int(time.time())}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({
            "model": MODEL, "seed": args.seed, "indices": idxs, "max_new": args.max_new,
            "configs": [{"name": n, "budget": b, "beta": be} for n, b, be in CONFIGS],
            "calibration": {"R": float(stats.layers[0].R.mean()),
                            "rotary_dim": stats.layers[0].rotary_dim,
                            "full_layers": stats.layer_indices},
            "summary": summary, "problems": records,
        }, f, indent=2)
    print(f"\n[saved] {out}  ({len(records)} problems, transcripts included)", flush=True)


if __name__ == "__main__":
    main()
