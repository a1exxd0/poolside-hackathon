"""
Dynamic Tokenization + Pooling for Laguna — end-to-end demo.

Combines:
  Paper 2 (Nawrot et al., 2023) — boundary detection (where to merge)
  Paper 1 (Feher et al., 2025)  — FVT embeddings    (how to merge)

Pipeline
--------
  1. Tokenise prompt with Laguna's standard tokenizer.
  2. Detect segment boundaries (whitespace / entropy / unigram).
  3. Average-pool subword embeddings within each segment (FVT).
  4. Run frozen Laguna on the shorter inputs_embeds sequence.
  5. Generate continuation tokens with the compressed KV cache as prefix.

Usage
-----
  python dynamic_inference.py --method whitespace
  python dynamic_inference.py --method entropy
  python dynamic_inference.py --method unigram
  python dynamic_inference.py --method all        # compare all methods
"""

import argparse
import re
import time
import os

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from boundary_detector import detect_boundaries
from dynamic_tokenizer import (
    apply_dynamic_tokenization,
    EmbeddingCache,
    shortening_factor,
)

load_dotenv()

MODEL_ID = "poolside/Laguna-XS.2"


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> reasoning content from model output.

    Laguna prepends <think> to the generation prompt.  <think> is a special
    token so it is skipped by the tokenizer's decode, but </think> is a
    regular text token that stays in the output.  This strips the dangling
    closing tag (and any full think blocks, should they appear).
    """
    # Leading </think> — opening tag was in the input prompt, not in output
    text = re.sub(r'^</think>\s*', '', text)
    # Full <think>...</think> blocks (e.g. if the model re-opens one)
    text = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL)
    return text.strip()

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Core inference helpers
# ---------------------------------------------------------------------------

def encode_prompt(prompt: str, tokenizer, model) -> torch.Tensor:
    """Apply the Laguna chat template and return a 1-D input_ids tensor."""
    messages = [{"role": "user", "content": prompt}]
    result = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    # apply_chat_template returns a plain tensor or a BatchEncoding depending
    # on the transformers version; normalise to a tensor here.
    if not isinstance(result, torch.Tensor):
        result = result["input_ids"]
    return result.squeeze(0).to(model.device)  # [L]


def baseline_generate(input_ids: torch.Tensor, model, tokenizer, max_new_tokens: int = 64):
    """Standard Laguna generation (no dynamic tokenization)."""
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - t0
    response = tokenizer.decode(
        output_ids[0][input_ids.shape[0]:], skip_special_tokens=True
    )
    response = strip_think_block(response)
    return response, elapsed




def dynamic_generate(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    method: str,
    cache: EmbeddingCache,
    max_new_tokens: int = 64,
) -> tuple:
    """
    Full dynamic-tokenization generation:
      - Compress the prompt with FVT (Paper 1) guided by Paper 2 boundaries.
      - Pass the merged inputs_embeds directly to model.generate().

    Passing inputs_embeds to generate() avoids past_key_values cache format
    incompatibilities across transformers versions: the model handles prefill
    and generation in one shot from the compressed embedding sequence.

    Returns (response_text, original_len, merged_len, total_elapsed).
    """
    original_len = input_ids.shape[0]

    t0 = time.perf_counter()

    # Step 1: detect boundaries (Paper 2)
    boundaries = detect_boundaries(
        input_ids,
        method=method,
        tokenizer=tokenizer,
        model=model,
    )

    # Step 2: FVT merge (Paper 1)
    embed_table = model.model.embed_tokens.weight  # [V, D]
    inputs_embeds, segments = apply_dynamic_tokenization(
        input_ids,
        boundaries,
        embed_table,
    )
    merged_len = inputs_embeds.shape[0]

    # Step 3: generate directly from merged embeddings.
    # model.generate() accepts inputs_embeds; it runs prefill + sampling
    # in one pass so we avoid past_key_values handoff issues.
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds.unsqueeze(0),  # [1, S, D]
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    elapsed = time.perf_counter() - t0
    response = strip_think_block(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return response, original_len, merged_len, elapsed


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_PROMPTS = [
    "Explain the concept of dynamic programming in simple terms.",
    "What is the capital of France and why is it historically significant?",
    "Write a Python function that computes the Fibonacci sequence.",
]


def run_demo(model, tokenizer, method: str, max_new_tokens: int = 64):
    cache = EmbeddingCache()

    print(f"\n{'='*70}")
    print(f"Method: {method.upper()}")
    print(f"{'='*70}")

    for prompt in DEMO_PROMPTS:
        print(f"\nPrompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")

        input_ids = encode_prompt(prompt, tokenizer, model)
        orig_len = input_ids.shape[0]

        # Baseline
        baseline_resp, baseline_time = baseline_generate(
            input_ids, model, tokenizer, max_new_tokens
        )

        # Dynamic tokenization
        dyn_resp, orig_len, merged_len, dyn_time = dynamic_generate(
            input_ids, model, tokenizer, method, cache, max_new_tokens
        )

        sf = shortening_factor(orig_len, merged_len)
        speedup = baseline_time / dyn_time if dyn_time > 0 else float("inf")

        print(f"  Original tokens : {orig_len}")
        print(f"  Merged tokens   : {merged_len}  (shortening factor {sf:.2f}x)")
        print(f"  Baseline time   : {baseline_time:.2f}s")
        print(f"  Dynamic time    : {dyn_time:.2f}s  (speedup {speedup:.2f}x)")
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
        choices=["whitespace", "entropy", "unigram", "all"],
        default="whitespace",
        help="Boundary detection method (Paper 2)",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=64,
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

    methods = ["whitespace", "entropy", "unigram"] if args.method == "all" else [args.method]

    if args.prompt:
        # Single-prompt comparison mode: baseline first, then each dynamic method.
        cache = EmbeddingCache()
        input_ids = encode_prompt(args.prompt, tokenizer, model)
        orig_len = input_ids.shape[0]

        print(f"\nPrompt : {args.prompt}")
        print(f"Tokens : {orig_len}  |  max_new_tokens={args.max_new_tokens}")
        print("─" * 70)

        baseline_resp, baseline_time = baseline_generate(
            input_ids, model, tokenizer, args.max_new_tokens
        )
        print(f"[baseline] {baseline_time:.2f}s")
        print(f"[baseline] {baseline_resp}\n")

        for method in methods:
            dyn_resp, orig_len, merged_len, dyn_time = dynamic_generate(
                input_ids, model, tokenizer, method, cache, args.max_new_tokens
            )
            sf = shortening_factor(orig_len, merged_len)
            speedup = baseline_time / dyn_time if dyn_time > 0 else float("inf")
            print(f"[{method}] {merged_len}/{orig_len} tokens  SF {sf:.2f}x  "
                  f"{dyn_time:.2f}s  speedup {speedup:.2f}x")
            print(f"[{method}] {dyn_resp}\n")
    else:
        for method in methods:
            run_demo(model, tokenizer, method, args.max_new_tokens)


if __name__ == "__main__":
    main()
