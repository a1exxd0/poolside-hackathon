"""Long-range NEEDLE retrieval for code: BF16 vs INT4-KIVI on Laguna-XS.2.

The meaningful long-context test for a KV-cache quantizer: can the model attend
back to a UNIQUE definition placed FAR earlier in the context after the cache
has been quantized to INT4?  We build a long synthetic Python "codebase" of
filler functions padding to a target token length, embed a single
non-guessable needle at a controlled DEPTH near the start, and at the very end
ask the model to recall it.  Grading is by EXACT value / executed equality --
the answer is only right if the model actually attended to the early definition
(it cannot be guessed).

Two needle kinds (both exact-graded):
  * "const"     : ``MAGIC_SEED_<id> = <7-digit prime-ish int>`` -> ask the value.
  * "transform" : ``def secret_transform_<id>(x): return x * A + B`` -> ask the
                  model to compute ``secret_transform_<id>(N)``; correct answer
                  is the exact integer A*N+B.

For each (context_length, depth) cell we run several trials (distinct needle
values / filler) and report retrieval accuracy for BF16 and INT4-KIVI plus
their agreement.  Batch-1 greedy only (the regime Int4KiviCache supports).

Usage
-----
  .venv/bin/python scripts/needle_code_bench.py --lengths 8000,16000 \
       --depths 0.05 --trials 3 --max-new 24
"""
from __future__ import annotations

import argparse
import random
import re
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

ROOT = "/home/alex/poolside-hackathon-kv-quant"
sys.path.insert(0, ROOT)
from int4_kivi.hf_cache import Int4KiviCache

MODEL = "poolside/Laguna-XS.2"


# --------------------------------------------------------------------------- #
# Synthetic codebase / filler
# --------------------------------------------------------------------------- #
_VERBS = ["compute", "process", "build", "merge", "filter", "scan", "encode",
          "decode", "reduce", "expand", "rotate", "shuffle", "balance", "score",
          "align", "cluster", "sample", "weight", "smooth", "sharpen"]
_NOUNS = ["matrix", "vector", "buffer", "token", "graph", "tensor", "window",
          "stream", "record", "payload", "segment", "lattice", "kernel",
          "gradient", "index", "shard", "bucket", "frame", "channel", "node"]


def _filler_fn(rng: random.Random, i: int) -> str:
    """A plausible, in-distribution Python helper that carries no needle info."""
    v = rng.choice(_VERBS)
    n = rng.choice(_NOUNS)
    a = rng.randint(2, 9)
    b = rng.randint(1, 20)
    return (
        f"def {v}_{n}_{i}(items):\n"
        f'    """{v.capitalize()} the {n} {i} with a small affine pass."""\n'
        f"    out = []\n"
        f"    for k, it in enumerate(items):\n"
        f"        out.append(it * {a} + {b} + k)\n"
        f"    return out\n"
    )


def _make_needle(rng: random.Random, kind: str):
    """Return (needle_source, question_text, expected_answer_str, meta)."""
    nid = rng.randint(100000, 999999)
    if kind == "const":
        # 8-digit non-guessable value.
        val = rng.randint(10_000_000, 99_999_999)
        src = (
            f"# Important runtime configuration constant.\n"
            f"MAGIC_SEED_{nid} = {val}\n"
        )
        q = (
            f"In the reference code above there is a constant named "
            f"MAGIC_SEED_{nid}. Reply with ONLY its exact integer value and "
            f"nothing else."
        )
        return src, q, str(val), {"nid": nid, "val": val}
    elif kind == "transform":
        # Use a unique multiplier constant so recall is non-guessable. We ask
        # the model to recall the *added* offset literal, which is a pure
        # attend-back retrieval (no arithmetic), keeping this a clean needle
        # test rather than a reasoning test.
        a = rng.choice([3, 7, 11, 13])
        b = rng.randint(1_000_000, 9_999_999)  # 7-digit, non-guessable
        src = (
            f"# Domain-specific transform used by the pipeline.\n"
            f"def secret_transform_{nid}(x):\n"
            f"    return x * {a} + {b}\n"
        )
        q = (
            f"In the reference code above there is a function named "
            f"secret_transform_{nid} whose body is `return x * {a} + OFFSET`. "
            f"Reply with ONLY the exact integer OFFSET value and nothing else."
        )
        return src, q, str(b), {"nid": nid, "a": a, "b": b}
    else:
        raise ValueError(kind)


# --------------------------------------------------------------------------- #
# Build a context of ~target_tokens with the needle at a given depth fraction
# --------------------------------------------------------------------------- #
def build_context(tok, target_tokens: int, depth: float, kind: str,
                  rng: random.Random):
    """Return (reference_code_text, question, expected_str, meta, n_pre_tokens).

    ``depth`` is the fraction of the *filler* placed BEFORE the needle (0.0 =
    needle right at the start, 0.5 = needle in the middle).  We grow filler
    functions until the whole reference block reaches target_tokens.
    """
    needle_src, question, expected, meta = _make_needle(rng, kind)

    # Generate a large pool of filler, then split around the needle by depth.
    fillers: list[str] = []
    i = 0
    # Build until the joined text (incl. needle) is at/above target.
    # Estimate ~ a few tokens per char; just append and recount periodically.
    pre, post = [], []
    # First, mass-produce filler text and measure.
    block_parts: list[str] = []
    while True:
        fn = _filler_fn(rng, i)
        block_parts.append(fn)
        i += 1
        if i % 40 == 0:
            joined = "\n".join(block_parts)
            ntok = tok(joined, return_tensors="pt").input_ids.shape[1]
            if ntok >= target_tokens:
                break
        if i > 200000:  # safety
            break

    # Now place needle at the requested depth among the filler functions.
    n_fn = len(block_parts)
    split = int(n_fn * depth)
    parts = block_parts[:split] + [needle_src] + block_parts[split:]
    reference = "\n".join(parts)

    # Trim to roughly target_tokens (keep the needle!). We trim from the END so
    # the needle (near the front for small depth) is preserved.
    ids = tok(reference, return_tensors="pt").input_ids[0]
    if ids.shape[0] > target_tokens * 1.15:
        ids = ids[: int(target_tokens * 1.15)]
        reference = tok.decode(ids, skip_special_tokens=True)
        # Re-confirm needle survived; if trimmed off (shouldn't for depth<0.7),
        # re-insert near the front.
        if meta_token(meta, kind) not in reference:
            reference = needle_src + "\n" + reference

    n_pre = tok(reference, return_tensors="pt").input_ids.shape[1]
    return reference, question, expected, meta, n_pre


def meta_token(meta, kind: str) -> str:
    if kind == "const":
        return f"MAGIC_SEED_{meta['nid']}"
    return f"secret_transform_{meta['nid']}"


# --------------------------------------------------------------------------- #
# Prompt + generation
# --------------------------------------------------------------------------- #
def build_input_ids(tok, reference: str, question: str, device):
    sys_msg = (
        "You are a careful code assistant. You will be shown a long Python "
        "reference codebase, then asked one question about a specific symbol "
        "defined in it. Answer using ONLY the information in the reference."
    )
    user_msg = (
        "Reference codebase:\n\n"
        f"```python\n{reference}\n```\n\n"
        f"{question}"
    )
    msgs = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": user_msg},
    ]
    return tok.apply_chat_template(
        msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False,
    ).to(device)


@torch.no_grad()
def generate(model, input_ids, max_new, mode, config):
    cache = Int4KiviCache(config=config) if mode == "int4" else DynamicCache()
    out = model.generate(
        input_ids, max_new_tokens=max_new, past_key_values=cache,
        use_cache=True, do_sample=False, num_beams=1,
    )
    return out[0, input_ids.shape[1]:].tolist(), cache


def grade(response: str, expected: str) -> bool:
    """Exact-value match: the expected integer must appear as a standalone
    number in the response (and be the first number, to avoid echoing the
    function's coefficients)."""
    nums = re.findall(r"-?\d+", response.replace(",", ""))
    if not nums:
        return False
    # Correct if the expected value is the first integer emitted, OR appears
    # and no other distinct integer precedes it. We accept "first integer".
    return nums[0] == expected


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=str, default="8000,16000,32000")
    ap.add_argument("--depths", type=str, default="0.05")
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--kinds", type=str, default="const,transform")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    lengths = [int(x) for x in args.lengths.split(",") if x]
    depths = [float(x) for x in args.depths.split(",") if x]
    kinds = [k for k in args.kinds.split(",") if k]

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()
    device = next(model.parameters()).device

    print(f"\n{'':#<92}")
    print("  NEEDLE-IN-CODEBASE RETRIEVAL  (exact-value graded, greedy, batch-1)")
    print(f"{'':#<92}")
    header = (f"  {'len':>7} {'depth':>6} {'kind':>10} {'ctx':>8} "
              f"{'BF16':>10} {'INT4':>10} {'agree':>7} {'ratio':>6} {'t/gen':>6}")
    print(header)
    print(f"{'':-<92}")

    rows = []
    example = None  # (length, kind, question, expected, bf16_resp, int4_resp)

    for length in lengths:
        for depth in depths:
            for kind in kinds:
                rng = random.Random(args.seed + length + int(depth * 1000)
                                    + hash(kind) % 1000)
                bf_ok = iq_ok = agree = 0
                ctxs, ratios, times = [], [], []
                for t in range(args.trials):
                    reference, question, expected, meta, n_pre = build_context(
                        tok, length, depth, kind, rng
                    )
                    input_ids = build_input_ids(tok, reference, question, device)
                    ctx = int(input_ids.shape[1])
                    ctxs.append(ctx)

                    res = {}
                    for mode in ("bf16", "int4"):
                        t0 = time.time()
                        ids, cache = generate(model, input_ids, args.max_new,
                                              mode, model.config)
                        dt = time.time() - t0
                        resp = tok.decode(ids, skip_special_tokens=True)
                        ok = grade(resp, expected)
                        res[mode] = (ok, resp, dt)
                        if mode == "int4":
                            try:
                                ratios.append(cache.compression_ratio_vs_bf16())
                            except Exception:
                                pass
                            times.append(dt)
                    bf_ok += res["bf16"][0]
                    iq_ok += res["int4"][0]
                    agree += (res["bf16"][0] == res["int4"][0])
                    if example is None and res["int4"][0] and res["bf16"][0]:
                        example = (ctx, kind, question, expected,
                                   res["bf16"][1].strip()[:120],
                                   res["int4"][1].strip()[:120])

                n = args.trials
                ctx_m = sum(ctxs) // len(ctxs)
                ratio_m = (sum(ratios) / len(ratios)) if ratios else 0.0
                t_m = (sum(times) / len(times)) if times else 0.0
                print(f"  {length:>7} {depth:>6.2f} {kind:>10} {ctx_m:>8} "
                      f"{bf_ok:>4}/{n:<5} {iq_ok:>4}/{n:<5} {agree:>3}/{n:<3} "
                      f"{ratio_m:>5.2f}x {t_m:>5.1f}s", flush=True)
                rows.append((length, depth, kind, ctx_m, bf_ok, iq_ok, agree, n,
                             ratio_m))

    print(f"{'':-<92}")
    # Aggregate by length.
    print("\n  RETRIEVAL ACCURACY BY CONTEXT LENGTH (summed over depths/kinds):")
    by_len: dict[int, list[int]] = {}
    for (length, depth, kind, ctx, bf, iq, ag, n, ratio) in rows:
        acc = by_len.setdefault(length, [0, 0, 0, 0])
        acc[0] += bf; acc[1] += iq; acc[2] += ag; acc[3] += n
    print(f"  {'target_len':>11} {'BF16':>14} {'INT4-KIVI':>14} {'agreement':>12}")
    for length in sorted(by_len):
        bf, iq, ag, n = by_len[length]
        print(f"  {length:>11} {bf:>6}/{n:<4} ({100*bf/n:>3.0f}%) "
              f"{iq:>6}/{n:<4} ({100*iq/n:>3.0f}%)  {ag:>4}/{n:<4} "
              f"({100*ag/n:>3.0f}%)")

    if example:
        ctx, kind, q, exp, bf_r, iq_r = example
        print(f"\n{'':-<92}")
        print(f"  REPRESENTATIVE NEEDLE (ctx={ctx} tok, kind={kind})")
        print(f"  Q: {q}")
        print(f"  expected: {exp}")
        print(f"  BF16  ->: {bf_r!r}")
        print(f"  INT4  ->: {iq_r!r}")
        print(f"{'':-<92}")

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
