"""
Dynamic Tokenization + FVT for Laguna — end-to-end demo.

Primary path (Feher et al., 2025 — arXiv:2411.18553, §3.1):
  Batch-level BPE-style merging: count adjacent pair frequencies, merge the
  most frequent pair (never crossing word boundaries), repeat m times, then
  embed merged tokens via FVT.  Select with --method bpe (default).

Legacy path (Nawrot et al., 2023):
  Boundary detection (whitespace / entropy / unigram) followed by FVT.
  Select with --method whitespace|entropy|unigram.

Usage
-----
  python dynamic_inference.py --prompt "Explain transformers"
  python dynamic_inference.py --num-merges 10 --prompt "Explain transformers"
  python dynamic_inference.py --sample --prompt "Explain transformers"
  python dynamic_inference.py --method whitespace --prompt "Explain transformers"
"""

import argparse

from dynamic_tokenizer import EmbeddingCache, shortening_factor
from inference import (
    load_model_and_tokenizer,
    encode_user_prompt,
    baseline_generate,
    dynamic_generate,
)

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    "Explain the concept of dynamic programming in simple terms.",
    "What is the capital of France and why is it historically significant?",
    "Write a Python function that computes the Fibonacci sequence.",
]


def run_demo(
    model,
    tokenizer,
    method: str,
    max_new_tokens: int = 64,
    num_merges: int | None = None,
    sample_merges: bool = False,
):
    cache = EmbeddingCache()

    print(f"\n{'='*70}")
    print(f"Method: {method.upper()}")
    if method == "bpe":
        merge_desc = "sampled" if sample_merges else (f"m={num_merges}" if num_merges else "50% of mmax")
        print(f"Merges : {merge_desc}")
    print(f"{'='*70}")

    for prompt in DEMO_PROMPTS:
        print(f"\nPrompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        input_ids = encode_user_prompt(prompt, tokenizer, model)
        orig_len = input_ids.shape[0]

        baseline_resp, baseline_time, baseline_gen, _ttft, _mem = baseline_generate(
            input_ids, model, tokenizer, max_new_tokens
        )

        dyn_resp, merged_len, dyn_time, dyn_gen, _ttft, _mem, _btime = dynamic_generate(
            input_ids, model, tokenizer, method, cache, max_new_tokens,
            num_merges=num_merges, sample_merges=sample_merges,
        )

        sf = shortening_factor(orig_len, merged_len)
        baseline_tps = baseline_gen / baseline_time if baseline_time > 0 else 0
        dyn_tps = dyn_gen / dyn_time if dyn_time > 0 else 0

        print(f"  Original tokens : {orig_len}  →  {merged_len} merged  (SF {sf:.2f}x)")
        print(f"  Baseline : {baseline_time:.2f}s  {baseline_gen} gen_tok  {baseline_tps:.1f} tok/s")
        if baseline_tps > 0:
            print(f"  Dynamic  : {dyn_time:.2f}s  {dyn_gen} gen_tok  {dyn_tps:.1f} tok/s"
                  f"  ({dyn_tps/baseline_tps:.2f}x)")
        else:
            print(f"  Dynamic  : {dyn_time:.2f}s  {dyn_gen} gen_tok  {dyn_tps:.1f} tok/s")
        print(f"  Baseline output : {baseline_resp[:120]!r}")
        print(f"  Dynamic output  : {dyn_resp[:120]!r}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Dynamic tokenization inference demo for Laguna"
    )
    parser.add_argument(
        "--method",
        choices=["bpe", "whitespace", "entropy", "unigram", "all"],
        default="bpe",
        help=(
            "bpe: batch-level BPE merging (Feher et al. 2025, default); "
            "whitespace/entropy/unigram: legacy boundary-detection methods (Nawrot et al. 2023)"
        ),
    )
    parser.add_argument(
        "--num-merges",
        type=int,
        default=None,
        dest="num_merges",
        help="Number of BPE merge steps (bpe method only). Default: 50%% of mmax.",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample m ~ U(0, mmax) per prompt instead of a fixed merge count (bpe method only).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run (overrides built-in demo prompts)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    methods = ["bpe", "whitespace", "entropy", "unigram"] if args.method == "all" else [args.method]

    if args.prompt:
        cache = EmbeddingCache()
        input_ids = encode_user_prompt(args.prompt, tokenizer, model)
        orig_len = input_ids.shape[0]

        print(f"\nPrompt : {args.prompt}")
        print(f"Tokens : {orig_len}  |  max_new_tokens={args.max_new_tokens}")
        print("─" * 70)

        baseline_resp, baseline_time, baseline_gen, baseline_ttft, _mem = baseline_generate(
            input_ids, model, tokenizer, args.max_new_tokens
        )
        baseline_tps = baseline_gen / baseline_time if baseline_time > 0 else 0
        print(f"[baseline] {baseline_time:.2f}s  {baseline_gen} gen_tok  {baseline_tps:.1f} tok/s")
        print(f"[baseline] {baseline_resp}\n")

        for method in methods:
            dyn_resp, merged_len, dyn_time, dyn_gen, _ttft, _mem, _btime = dynamic_generate(
                input_ids, model, tokenizer, method, cache, args.max_new_tokens,
                num_merges=args.num_merges, sample_merges=args.sample,
            )
            sf = shortening_factor(orig_len, merged_len)
            dyn_tps = dyn_gen / dyn_time if dyn_time > 0 else 0
            tps_ratio = dyn_tps / baseline_tps if baseline_tps > 0 else float("inf")
            print(f"[{method}] {merged_len}/{orig_len} tokens  SF {sf:.2f}x  "
                  f"{dyn_time:.2f}s  {dyn_gen} gen_tok  {dyn_tps:.1f} tok/s  ({tps_ratio:.2f}x)")
            print(f"[{method}] {dyn_resp}\n")
    else:
        for method in methods:
            run_demo(
                model, tokenizer, method, args.max_new_tokens,
                num_merges=args.num_merges, sample_merges=args.sample,
            )


if __name__ == "__main__":
    main()
