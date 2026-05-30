"""Long-context CODING benchmark A/B for the bf16 residual-window lever.

Isolates the PROGRESS.md "Long-context quality" lever on the HF generate path:
executed HumanEval pass@1 in the LONG regime (in-distribution code prefix), for

  * int4  residual=0    -> the existing Int4KiviCache behaviour  (BEFORE)
  * int4  residual=R     -> bf16 residual window kept lossless     (AFTER)
  * bf16   DynamicCache  -> ceiling                                (optional)

Same prompts, same greedy decode for every config -> apples-to-apples; the only
thing that changes between BEFORE and AFTER is how many recent tokens stay bf16.

Env: N, PREFIX_TOKENS, MAXNEW, RESIDUAL (default 128), SKIP_BF16=1.
"""
import json
import os
import sys
import time

WORKTREE = "/home/alex/poolside-hackathon-kv-quant/.claude/worktrees/kv-quant-long-context"
sys.path.insert(0, WORKTREE)

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

# Import the worktree's int4_kivi package FIRST so it is cached in sys.modules
# before the helper scripts below run their own sys.path.insert(0, <main-tree>),
# which would otherwise shadow int4_kivi with the main worktree's copy (no
# residual_hf_cache there).
from int4_kivi.hf_cache import Int4KiviCache
from int4_kivi.residual_hf_cache import ResidualInt4KiviCache
from scripts.humaneval_bench import extract_code, run_tests
from scripts.longctx_code_bench import build_input_ids, build_prefix_text

MODEL = "poolside/Laguna-XS.2"
N = int(os.environ.get("N", "20"))
START = int(os.environ.get("START", "0"))  # problem offset (harder slices live later)
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "12000"))
MAXNEW = int(os.environ.get("MAXNEW", "256"))
RESIDUAL = int(os.environ.get("RESIDUAL", "128"))
SKIP_BF16 = os.environ.get("SKIP_BF16") == "1"


def make_cache(kind, config):
    if kind == "bf16":
        return DynamicCache()
    if kind == "int4_r0":
        return Int4KiviCache(config=config)
    if kind == "int4_res":
        return ResidualInt4KiviCache(config=config, residual=RESIDUAL)
    raise ValueError(kind)


@torch.no_grad()
def gen(model, input_ids, kind):
    L = input_ids.shape[1]
    cache = make_cache(kind, model.config)
    out = model.generate(input_ids, max_new_tokens=MAXNEW, past_key_values=cache,
                         use_cache=True, do_sample=False, num_beams=1)
    return out[0, L:].tolist(), cache


def main():
    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto").eval()
    device = next(model.parameters()).device

    ds = load_dataset("openai/openai_humaneval", split="test").select(
        range(START, START + N))
    print(f"[prefix] building ~{PREFIX_TOKENS}-token code prefix ...", flush=True)
    prefix_text, prefix_tok = build_prefix_text(tok, PREFIX_TOKENS)
    print(f"[prefix] actual {prefix_tok} tokens", flush=True)

    configs = ["int4_r0", "int4_res"] + ([] if SKIP_BF16 else ["bf16"])
    passes = {c: 0 for c in configs}
    ratios = {"int4_r0": [], "int4_res": []}
    ctxs = []

    for i, prob in enumerate(ds):
        ids = build_input_ids(tok, prob["prompt"], prefix_text, device)
        ctxs.append(int(ids.shape[1]))
        row = {}
        for c in configs:
            t0 = time.time()
            tokens, cache = gen(model, ids, c)
            dt = time.time() - t0
            resp = tok.decode(tokens, skip_special_tokens=True)
            ok, _ = run_tests(extract_code(resp, prob["prompt"]), prob["test"],
                              prob["entry_point"])
            passes[c] += int(ok)
            row[c] = ("✓" if ok else "✗", dt)
            if c in ratios:
                try:
                    ratios[c].append(cache.compression_ratio_vs_bf16())
                except Exception:
                    pass
        msg = "  ".join(f"{c}={row[c][0]}({row[c][1]:.0f}s)" for c in configs)
        print(f"  {i+1:>3}/{N}  ctx={ids.shape[1]:>6}  {msg}", flush=True)

    print("\n" + "=" * 70)
    print(f"  LONG-CTX HumanEval pass@1   (N={N}, prefix~{prefix_tok}, residual={RESIDUAL})")
    print("=" * 70)
    for c in configs:
        extra = ""
        if c in ratios and ratios[c]:
            extra = f"   cache {sum(ratios[c])/len(ratios[c]):.2f}x vs bf16"
        print(f"  {c:<10} {passes[c]:>3}/{N}  ({100*passes[c]/N:.0f}%){extra}")
    if "int4_r0" in passes and "int4_res" in passes:
        d = passes["int4_res"] - passes["int4_r0"]
        print(f"\n  residual-window lever: {passes['int4_r0']}/{N} -> "
              f"{passes['int4_res']}/{N}  ({d:+d} problems)")
    print(f"  ctx {min(ctxs)}..{max(ctxs)} tokens")

    json.dump({"N": N, "prefix_tok": prefix_tok, "residual": RESIDUAL,
               "passes": passes,
               "ratios": {k: (sum(v)/len(v) if v else None) for k, v in ratios.items()},
               "ctx_min": min(ctxs), "ctx_max": max(ctxs)},
              open("/tmp/longctx_residual_ab.json", "w"))
    print("\nLONGCTX_RESIDUAL_AB DONE")


if __name__ == "__main__":
    main()
