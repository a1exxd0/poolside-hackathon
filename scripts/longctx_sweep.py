"""Context-length SWEEP of HumanEval pass@1: BF16 vs INT4-KIVI on Laguna-XS.2.

Runs the long-prefix HumanEval protocol at increasing prefix lengths in ONE
process (model loaded once) and prints a context-vs-accuracy table.  Reuses the
validated prefix builder, prompt construction, extraction and execution from
``scripts.longctx_code_bench`` / ``scripts.humaneval_bench`` -- this file only
orchestrates the sweep.

For each prefix size we report: BF16 pass@1, INT4-KIVI pass@1, per-problem
agreement, actual mean context length, INT4 cache compression ratio, mean gen
time, and (at the largest cell that ran) the INT4 vs BF16 KV memory.

Usage
-----
  .venv/bin/python scripts/longctx_sweep.py --n 8 --max-new 128 \
      --prefixes 8000,16000,32000,64000,128000
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = "/home/alex/poolside-hackathon-kv-quant"
sys.path.insert(0, ROOT)

from scripts.humaneval_bench import extract_code, run_tests
from scripts.longctx_code_bench import build_prefix_text, build_input_ids
from int4_kivi.hf_cache import Int4KiviCache

MODEL = "poolside/Laguna-XS.2"


@torch.no_grad()
def generate(model, input_ids, max_new, mode, config):
    cache = Int4KiviCache(config=config) if mode == "int4" else DynamicCache()
    out = model.generate(
        input_ids, max_new_tokens=max_new, past_key_values=cache,
        use_cache=True, do_sample=False, num_beams=1,
    )
    return out[0, input_ids.shape[1]:].tolist(), cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--prefixes", type=str, default="8000,16000,32000,64000,128000")
    ap.add_argument("--n-large", type=int, default=0,
                    help="if >0, override --n for prefixes >= --large-thresh")
    ap.add_argument("--large-thresh", type=int, default=100000)
    args = ap.parse_args()

    prefixes = [int(x) for x in args.prefixes.split(",") if x]

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"
    ).eval()
    device = next(model.parameters()).device

    full_ds = load_dataset("openai/openai_humaneval", split="test")

    table = []  # (prefix_target, ctx_mean, n, bf, iq, agree, ratio, t_mean,
                #  int4_bytes, bf16_bytes)

    for ptarget in prefixes:
        n = args.n
        if args.n_large and ptarget >= args.large_thresh:
            n = args.n_large
        ds = full_ds.select(range(n))

        print(f"\n{'':=<92}")
        print(f"  PREFIX TARGET {ptarget} tokens   (n={n}, max_new={args.max_new})")
        print(f"{'':=<92}", flush=True)
        prefix_text, prefix_tok = build_prefix_text(tok, ptarget)
        print(f"  [prefix] actual prefix length: {prefix_tok} tokens", flush=True)
        print(f"  {'#':>3} {'task_id':<16} {'BF16':>5} {'INT4':>5} {'ctx':>8} "
              f"{'bf16_s':>7} {'int4_s':>7}", flush=True)

        res = {"bf16": [], "int4": []}
        ctxs, ratios, times = [], [], []
        last_int4_bytes = last_bf16_bytes = 0
        for idx, prob in enumerate(ds):
            prompt, tests, entry = prob["prompt"], prob["test"], prob["entry_point"]
            input_ids = build_input_ids(tok, prompt, prefix_text, device)
            ctx = int(input_ids.shape[1]); ctxs.append(ctx)
            row, tnote = {}, {}
            for mode in ("bf16", "int4"):
                t0 = time.time()
                ids, cache = generate(model, input_ids, args.max_new, mode,
                                      model.config)
                dt = time.time() - t0; tnote[mode] = dt
                resp = tok.decode(ids, skip_special_tokens=True)
                code = extract_code(resp, prompt)
                passed, _ = run_tests(code, tests, entry)
                res[mode].append(passed); row[mode] = passed
                if mode == "int4":
                    try:
                        ratios.append(cache.compression_ratio_vs_bf16())
                        last_int4_bytes = cache.nbytes()
                        last_bf16_bytes = cache.bf16_nbytes()
                    except Exception:
                        pass
                    times.append(dt)
            print(f"  {idx+1:>3} {prob['task_id']:<16} "
                  f"{'PASS' if row['bf16'] else 'fail':>5} "
                  f"{'PASS' if row['int4'] else 'fail':>5} {ctx:>8} "
                  f"{tnote['bf16']:>6.1f}s {tnote['int4']:>6.1f}s", flush=True)

        bf, iq = sum(res["bf16"]), sum(res["int4"])
        agree = sum(a == b for a, b in zip(res["bf16"], res["int4"]))
        ctx_m = sum(ctxs) // len(ctxs)
        ratio_m = (sum(ratios) / len(ratios)) if ratios else 0.0
        t_m = (sum(times) / len(times)) if times else 0.0
        print(f"  -> ctx~{ctx_m}  BF16={bf}/{n}  INT4={iq}/{n}  "
              f"agree={agree}/{n}  ratio={ratio_m:.2f}x  "
              f"int4_KV={last_int4_bytes/2**20:.0f}MiB "
              f"bf16_KV={last_bf16_bytes/2**20:.0f}MiB", flush=True)
        table.append((ptarget, ctx_m, n, bf, iq, agree, ratio_m, t_m,
                      last_int4_bytes, last_bf16_bytes))

    # ---- final table ----
    print(f"\n{'':#<100}")
    print("  CONTEXT-LENGTH SWEEP  -- HumanEval pass@1, BF16 vs INT4-KIVI "
          "(executed, greedy, batch-1)")
    print(f"{'':#<100}")
    print(f"  {'ctx_tok':>8} {'n':>3} {'BF16':>10} {'INT4-KIVI':>12} "
          f"{'agree':>8} {'ratio':>7} {'int4_KV':>9} {'bf16_KV':>9} {'t/gen':>7}")
    print(f"{'':-<100}")
    for (pt, ctx, n, bf, iq, ag, ratio, tm, ib, bb) in table:
        print(f"  {ctx:>8} {n:>3} {bf:>3}/{n:<3}({100*bf/n:>3.0f}%) "
              f"{iq:>3}/{n:<3}({100*iq/n:>3.0f}%) {ag:>3}/{n:<3} "
              f"{ratio:>6.2f}x {ib/2**20:>7.0f}M {bb/2**20:>7.0f}M {tm:>6.1f}s")
    print(f"{'':-<100}")
    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
