"""Long-context coding benchmark: BF16 (DynamicCache) vs INT4-KIVI (Int4KiviCache).

Measures EXECUTED pass@1 on HumanEval in two regimes, to verify the thesis
(PROBLEM.md) that INT4-KIVI pass@1 ~= BF16 pass@1 and that the parity HOLDS at
long context -- the regime KV-cache quantization actually exists for.

Regimes
-------
* short : plain HumanEval. The user message is just the function to complete,
          so the KV cache is small (~hundreds of tokens) when decoding starts.
* long  : the SAME problems, but a long, in-distribution Python-source prefix
          (concatenated transformers ``modeling_*.py``, like
          ``scripts/quant_longctx.py``) is prepended to the user message. The
          prefix is identical for BF16 and INT4-KIVI, so the comparison stays
          apples-to-apples; it only lengthens the context, it does not change
          the task. We report the actual total context length and, for
          INT4-KIVI, the cache compression ratio vs BF16.

Both modes are batch-1 greedy (do_sample=False) -- the regime Int4KiviCache
supports. We reuse ``extract_code`` and ``run_tests`` from the existing,
validated ``scripts/humaneval_bench.py`` rather than re-implementing them.

Usage
-----
    .venv/bin/python scripts/longctx_code_bench.py --n 20 --prefix-tokens 4000
Start smaller to validate the pipeline:
    .venv/bin/python scripts/longctx_code_bench.py --n 8 --prefix-tokens 4000
"""
from __future__ import annotations

import argparse
import glob
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = "/home/alex/poolside-hackathon-kv-quant"
sys.path.insert(0, ROOT)

# Reuse the validated extraction + execution path.
from scripts.humaneval_bench import extract_code, run_tests
from int4_kivi.hf_cache import Int4KiviCache

MODEL = "poolside/Laguna-XS.2"


# --------------------------------------------------------------------------- #
# Long in-distribution code prefix (real Python source -> in-distribution)
# --------------------------------------------------------------------------- #
def build_prefix_text(tok, target_tokens: int) -> tuple[str, int]:
    """Concatenate transformers modeling_*.py source until >= target_tokens.

    Returns (text, n_tokens). The text is plain Python source, so it is
    in-distribution for a code model and only lengthens the context.
    """
    files = sorted(
        glob.glob(
            f"{ROOT}/.venv/**/transformers/**/modeling_*.py",
            recursive=True,
        )
    )
    if not files:
        # Fall back to repo source if the venv layout differs.
        files = sorted(glob.glob(f"{ROOT}/**/*.py", recursive=True))

    texts: list[str] = []
    for f in files:
        try:
            texts.append(open(f).read())
        except OSError:
            continue
        joined = "\n\n".join(texts)
        ids = tok(joined, return_tensors="pt").input_ids[0]
        if ids.shape[0] >= target_tokens:
            # Trim to exactly target_tokens tokens for a clean, fixed prefix.
            ids = ids[:target_tokens]
            text = tok.decode(ids, skip_special_tokens=True)
            return text, int(ids.shape[0])

    # Ran out of files before reaching the target -- use whatever we have.
    joined = "\n\n".join(texts)
    ids = tok(joined, return_tensors="pt").input_ids[0]
    return joined, int(ids.shape[0])


# --------------------------------------------------------------------------- #
# Prompt construction
# --------------------------------------------------------------------------- #
def build_input_ids(tok, prompt: str, prefix_text: str | None, device):
    """Build chat-templated input_ids for one HumanEval problem.

    ``prefix_text`` (long regime) is prepended to the user message as a quoted
    reference block; it does not change the requested task.
    """
    sys_msg = (
        "You are a Python coding assistant. Complete the function below. "
        "Return a fenced ```python``` code block containing the complete "
        "function (including signature and docstring)."
    )
    if prefix_text:
        user_msg = (
            "Here is some reference Python source code for context. You do not "
            "need to use it; it is provided only as background.\n\n"
            f"```python\n{prefix_text}\n```\n\n"
            "Now, ignoring the reference above, complete this Python function:\n\n"
            f"```python\n{prompt}```"
        )
    else:
        user_msg = f"Complete this Python function:\n\n```python\n{prompt}```"

    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    input_ids = tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    ).to(device)
    return input_ids


# --------------------------------------------------------------------------- #
# Generation (batch-1 greedy, identical decode for both caches)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model, input_ids, max_new: int, mode: str, config):
    """Greedy-decode with either a BF16 DynamicCache or an Int4KiviCache."""
    L = input_ids.shape[1]
    if mode == "int4":
        cache = Int4KiviCache(config=config)
    else:
        cache = DynamicCache()
    out = model.generate(
        input_ids,
        max_new_tokens=max_new,
        past_key_values=cache,
        use_cache=True,
        do_sample=False,
        num_beams=1,
    )
    token_ids = out[0, L:].tolist()
    return token_ids, cache


# --------------------------------------------------------------------------- #
# One regime (short or long) over all problems
# --------------------------------------------------------------------------- #
def run_regime(model, tok, ds, args, device, regime: str, prefix_text, prefix_tok):
    label = regime.upper()
    print(f"\n{'':═<86}")
    print(f"  REGIME: {label}"
          + (f"   (prefix ~{prefix_tok} tok)" if regime == "long" else "   (plain HumanEval)"))
    print(f"{'':═<86}")
    print(f"  {'#':>3}  {'task_id':<26}  {'BF16':>5}  {'INT4':>5}  {'ctx':>6}  {'note'}")
    print(f"{'':─<86}")

    results = {"bf16": [], "int4": []}
    ctx_lens: list[int] = []
    int4_ratios: list[float] = []
    sample_pass_completion = None

    for idx, prob in enumerate(ds):
        task = prob["task_id"]
        prompt = prob["prompt"]
        tests = prob["test"]
        entry = prob["entry_point"]

        input_ids = build_input_ids(
            tok, prompt, prefix_text if regime == "long" else None, device
        )
        ctx_len = int(input_ids.shape[1])
        ctx_lens.append(ctx_len)

        row = {}
        note = ""
        for mode in ("bf16", "int4"):
            t0 = time.time()
            token_ids, cache = generate(
                model, input_ids, args.max_new, mode, model.config
            )
            dt = time.time() - t0
            response = tok.decode(token_ids, skip_special_tokens=True)
            code = extract_code(response, prompt)
            passed, err = run_tests(code, tests, entry)
            results[mode].append(passed)
            row[mode] = passed
            if mode == "int4":
                try:
                    int4_ratios.append(cache.compression_ratio_vs_bf16())
                except Exception:
                    pass
                note = f"{dt:.0f}s"
                if passed and sample_pass_completion is None:
                    sample_pass_completion = (task, ctx_len, code)

        bsym = "✓" if row["bf16"] else "✗"
        isym = "✓" if row["int4"] else "✗"
        print(f"  {idx+1:>3}  {task:<26}  {bsym:>5}  {isym:>5}  {ctx_len:>6}  {note}",
              flush=True)

    print(f"{'':─<86}")
    return results, ctx_lens, int4_ratios, sample_pass_completion


# --------------------------------------------------------------------------- #
# Summary helpers
# --------------------------------------------------------------------------- #
def summarize(regime, results, ctx_lens, int4_ratios, n):
    bf = sum(results["bf16"])
    iq = sum(results["int4"])
    agree = sum(a == b for a, b in zip(results["bf16"], results["int4"]))
    print(f"\n  [{regime.upper()}] pass@1   BF16 = {bf}/{n} ({100*bf/n:.0f}%)   "
          f"INT4-KIVI = {iq}/{n} ({100*iq/n:.0f}%)")
    print(f"  [{regime.upper()}] BF16-vs-INT4 agreement: {agree}/{n} "
          f"({100*agree/n:.0f}% same pass/fail)")
    if ctx_lens:
        print(f"  [{regime.upper()}] context length: min={min(ctx_lens)} "
              f"max={max(ctx_lens)} mean={sum(ctx_lens)//len(ctx_lens)} tokens")
    if int4_ratios:
        avg = sum(int4_ratios) / len(int4_ratios)
        print(f"  [{regime.upper()}] INT4-KIVI cache compression vs BF16: "
              f"{avg:.2f}x (mean over problems)")
    return bf, iq, agree


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="number of HumanEval problems")
    ap.add_argument("--max-new", type=int, default=512, help="max new tokens")
    ap.add_argument("--prefix-tokens", type=int, default=4000,
                    help="target long-context prefix length in tokens")
    ap.add_argument("--regime", choices=["short", "long", "both"], default="both")
    args = ap.parse_args()

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"[data] HumanEval ({args.n} problems) ...", flush=True)
    ds = load_dataset("openai/openai_humaneval", split="test").select(range(args.n))

    prefix_text, prefix_tok = None, 0
    if args.regime in ("long", "both"):
        print(f"[prefix] building ~{args.prefix_tokens}-token in-distribution "
              f"code prefix ...", flush=True)
        prefix_text, prefix_tok = build_prefix_text(tok, args.prefix_tokens)
        print(f"[prefix] actual prefix length: {prefix_tok} tokens", flush=True)

    regimes = (["short", "long"] if args.regime == "both" else [args.regime])
    summary = {}
    samples = {}
    for regime in regimes:
        res, ctx, ratios, sample = run_regime(
            model, tok, ds, args, device, regime, prefix_text, prefix_tok
        )
        summary[regime] = (res, ctx, ratios)
        samples[regime] = sample

    # ---- final table ----
    print(f"\n{'':█<86}")
    print("  FINAL RESULTS  (executed pass@1, greedy, batch-1)")
    print(f"{'':█<86}")
    for regime in regimes:
        res, ctx, ratios = summary[regime]
        summarize(regime, res, ctx, ratios, args.n)

    # ---- representative passing INT4-KIVI completion (prefer long) ----
    chosen = samples.get("long") or samples.get("short")
    if chosen:
        task, ctx_len, code = chosen
        print(f"\n{'':─<86}")
        print(f"  Representative PASSING INT4-KIVI completion  "
              f"(task={task}, ctx={ctx_len} tok)")
        print(f"{'':─<86}")
        print(code.rstrip()[:1600])
        print(f"{'':─<86}")

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
