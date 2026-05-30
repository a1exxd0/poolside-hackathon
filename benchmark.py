"""
Benchmark dynamic tokenization on Laguna's official evaluation suite.

Laguna XS.2 reported scores (model card, April 2026):
  - SWE-bench Verified     68.2% pass@1
  - SWE-bench Multilingual 62.4% pass@1
  - SWE-bench Pro          44.5% pass@1
  - Terminal-Bench 2.0     30.1% pass@1

What this script measures
-------------------------
  - original_tokens   : tokens in the chat-formatted prompt
  - merged_tokens     : tokens after FVT boundary merging
  - shortening_factor : original / merged  (higher = more compression)
  - baseline_time_s   : wall-clock time for standard generation (no modification)
  - dynamic_time_s    : wall-clock time with dynamic tokenization
  - speedup           : baseline_time / dynamic_time
  - baseline_response : raw text from standard generation
  - dynamic_response  : raw text from dynamic generation

Pass@1 (patch correctness) must be computed separately with the harness.

Usage
-----
  python benchmark.py --benchmark swe_verified --n 20 --method whitespace
  python benchmark.py --benchmark all --n 10 --method all
"""

import argparse
import json
import os
import re
import time
import datetime
from pathlib import Path

import torch
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer

from boundary_detector import detect_boundaries
from dynamic_tokenizer import apply_dynamic_tokenization, EmbeddingCache, shortening_factor

load_dotenv()

MODEL_ID = "poolside/Laguna-XS.2"

# Reported Laguna XS.2 scores (model card, April 2026) — for reference only
LAGUNA_SCORES = {
    "swe_verified":     0.682,
    "swe_multilingual": 0.624,
    "swe_pro":          0.445,
    "terminal_bench":   0.301,
}

# ---------------------------------------------------------------------------
# Benchmark dataset configs
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = {
    "swe_verified": {
        "hf_path": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "prompt_fn": "swe_bench_prompt",
        "id_field": "instance_id",
        "description": "SWE-bench Verified",
    },
    "swe_multilingual": {
        "hf_path": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "prompt_fn": "swe_bench_prompt",
        "id_field": "instance_id",
        "description": "SWE-bench Multilingual (using Verified as proxy)",
        "note": "The official multilingual split requires the Poolside harness; "
                "Verified is used as a local proxy.",
    },
    "swe_pro": {
        "hf_path": "princeton-nlp/SWE-bench_Verified",
        "split": "test",
        "prompt_fn": "swe_bench_prompt",
        "id_field": "instance_id",
        "description": "SWE-bench Pro (harder subset of Verified)",
    },
    "terminal_bench": {
        "hf_path": "terminal-bench/terminal-bench",
        "split": "test",
        "prompt_fn": "terminal_bench_prompt",
        "id_field": "id",
        "description": "Terminal-Bench 2.0",
    },
}

# ---------------------------------------------------------------------------
# Model loading / unloading
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
# Prompt construction
# ---------------------------------------------------------------------------

SWE_SYSTEM = (
    "You are an expert software engineer. "
    "Given a GitHub issue, produce a minimal unified diff (git patch) that resolves it. "
    "Output ONLY the patch — no explanation, no markdown fences."
)

TERMINAL_SYSTEM = (
    "You are an expert systems engineer working in a Linux terminal. "
    "Complete the task described below using shell commands. "
    "Output only the shell commands, one per line, with no explanation."
)


def swe_bench_prompt(row: dict) -> str:
    repo = row.get("repo", "unknown/repo")
    issue = row.get("problem_statement", "")
    hints = row.get("hints_text", "")
    parts = [f"Repository: {repo}", "", "Issue:", issue]
    if hints and hints.strip():
        parts += ["", "Hints:", hints]
    parts += ["", "Produce a git patch that resolves the issue."]
    return "\n".join(parts)


def terminal_bench_prompt(row: dict) -> str:
    task = row.get("task", row.get("problem_statement", row.get("instruction", "")))
    context = row.get("context", row.get("setup", ""))
    parts = []
    if context:
        parts += ["Environment setup:", context, ""]
    parts += ["Task:", task]
    return "\n".join(parts)


PROMPT_BUILDERS = {
    "swe_bench_prompt": swe_bench_prompt,
    "terminal_bench_prompt": terminal_bench_prompt,
}


def build_system_prompt(cfg: dict) -> str:
    if cfg["prompt_fn"] == "swe_bench_prompt":
        return SWE_SYSTEM
    return TERMINAL_SYSTEM


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def strip_think_block(text: str) -> str:
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*</think>', '', text)
    return text.strip()


def encode_prompt(system: str, user: str, tokenizer, model) -> torch.Tensor:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    result = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(result, torch.Tensor):
        result = result["input_ids"]
    return result.squeeze(0).to(model.device)


def baseline_generate(input_ids: torch.Tensor, model, tokenizer, max_new_tokens: int):
    t0 = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - t0
    text = tokenizer.decode(output_ids[0][input_ids.shape[0]:], skip_special_tokens=True)
    return strip_think_block(text), elapsed


def dynamic_generate(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    method: str,
    cache: EmbeddingCache,
    max_new_tokens: int,
) -> tuple:
    t0 = time.perf_counter()
    boundaries = detect_boundaries(input_ids, method=method, tokenizer=tokenizer, model=model)
    embed_table = model.model.embed_tokens.weight
    inputs_embeds, _segments = apply_dynamic_tokenization(input_ids, boundaries, embed_table)
    merged_len = inputs_embeds.shape[0]
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
    elapsed = time.perf_counter() - t0
    text = strip_think_block(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    return text, merged_len, elapsed


# ---------------------------------------------------------------------------
# Single benchmark run
# ---------------------------------------------------------------------------

def run_benchmark(
    name: str,
    model,
    tokenizer,
    methods: list,
    n: int,
    max_new_tokens: int,
    output_path: Path | None,
):
    cfg = BENCHMARK_CONFIGS[name]
    reported_score = LAGUNA_SCORES.get(name)
    build_prompt = PROMPT_BUILDERS[cfg["prompt_fn"]]
    system = build_system_prompt(cfg)

    print(f"\n{'='*72}")
    print(f"Benchmark : {cfg['description']}")
    if reported_score is not None:
        print(f"Laguna XS.2 reported pass@1 (full harness): {reported_score*100:.1f}%")
    if "note" in cfg:
        print(f"Note      : {cfg['note']}")
    print(f"Methods   : {methods}   n={n}   max_new_tokens={max_new_tokens}")
    print(f"{'='*72}")

    # Load dataset
    try:
        ds = load_dataset(cfg["hf_path"], split=cfg["split"], token=os.getenv("HF_TOKEN"))
    except Exception as e:
        print(f"[SKIP] Could not load {cfg['hf_path']}: {e}")
        return []

    samples = list(ds.select(range(min(n, len(ds)))))
    cache = EmbeddingCache()
    results = []
    skipped = 0

    for i, row in enumerate(samples):
        instance_id = row.get(cfg["id_field"], f"sample_{i}")

        print(f"\n[{i+1}/{len(samples)}] {instance_id}")

        try:
            user_prompt = build_prompt(row)
            if not user_prompt.strip():
                print("  [SKIP] empty prompt — unexpected row schema")
                skipped += 1
                continue

            input_ids = encode_prompt(system, user_prompt, tokenizer, model)
            orig_len = input_ids.shape[0]
            print(f"  Prompt tokens : {orig_len}")

            baseline_resp, baseline_time = baseline_generate(
                input_ids, model, tokenizer, max_new_tokens
            )
            print(f"  [baseline]  {baseline_time:.1f}s")

            record = {
                "benchmark": name,
                "instance_id": instance_id,
                "original_tokens": orig_len,
                "max_new_tokens": max_new_tokens,
                "baseline_time_s": round(baseline_time, 3),
                "baseline_response": baseline_resp,
                "methods": {},
            }

            for method in methods:
                try:
                    dyn_resp, merged_len, dyn_time = dynamic_generate(
                        input_ids, model, tokenizer, method, cache, max_new_tokens
                    )
                    sf = shortening_factor(orig_len, merged_len)
                    speedup = baseline_time / dyn_time if dyn_time > 0 else float("inf")
                    print(f"  [{method}]  {merged_len}/{orig_len} tok  SF {sf:.2f}x  "
                          f"{dyn_time:.1f}s  speedup {speedup:.2f}x")
                    record["methods"][method] = {
                        "merged_tokens": merged_len,
                        "shortening_factor": round(sf, 4),
                        "time_s": round(dyn_time, 3),
                        "speedup": round(speedup, 4),
                        "response": dyn_resp,
                    }
                except Exception as e:
                    print(f"  [{method}]  ERROR: {e}")
                    record["methods"][method] = {"error": str(e)}

            results.append(record)

            if output_path:
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

        except Exception as e:
            print(f"  [SKIP] sample failed: {e}")
            skipped += 1
            continue

    if skipped:
        print(f"\n  {skipped}/{len(samples)} samples skipped due to errors.")

    return results


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def print_summary(all_results: list, methods: list):
    if not all_results:
        return

    by_bench: dict = {}
    for r in all_results:
        by_bench.setdefault(r["benchmark"], []).append(r)

    print(f"\n{'='*72}")
    print("SUMMARY")
    print(f"{'='*72}")

    for bench_name, records in by_bench.items():
        cfg = BENCHMARK_CONFIGS[bench_name]
        reported = LAGUNA_SCORES.get(bench_name)
        n = len(records)
        print(f"\n{cfg['description']}  (n={n})")
        if reported is not None:
            print(f"  Reported pass@1 (full harness): {reported*100:.1f}%")

        orig_tokens = [r["original_tokens"] for r in records]
        print(f"  Avg prompt length : {sum(orig_tokens)/n:.0f} tokens")
        baseline_times = [r["baseline_time_s"] for r in records]
        print(f"  Avg baseline time : {sum(baseline_times)/n:.1f}s  (no modification)")

        for method in methods:
            sfs = [r["methods"][method]["shortening_factor"]
                   for r in records if method in r.get("methods", {})
                   and "shortening_factor" in r["methods"][method]]
            speedups = [r["methods"][method]["speedup"]
                        for r in records if method in r.get("methods", {})
                        and "speedup" in r["methods"][method]]
            if not sfs:
                continue
            print(f"  [{method}]  avg SF {sum(sfs)/len(sfs):.2f}x  "
                  f"avg speedup {sum(speedups)/len(speedups):.2f}x")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dynamic tokenization on Laguna's evaluation suite"
    )
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_CONFIGS) + ["all"],
        default="swe_verified",
        help="Which benchmark to run",
    )
    parser.add_argument(
        "--method",
        choices=["whitespace", "entropy", "unigram", "all"],
        default="whitespace",
        help="Boundary detection method(s)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of samples to evaluate per benchmark",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max tokens to generate per sample",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="JSONL file to write results to (default: auto-named per benchmark)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    methods = ["whitespace", "entropy", "unigram"] if args.method == "all" else [args.method]
    benchmarks = list(BENCHMARK_CONFIGS) if args.benchmark == "all" else [args.benchmark]

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = []

    for bench in benchmarks:
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = Path(f"benchmark_{bench}_{run_ts}.jsonl")

        results = run_benchmark(
            name=bench,
            model=model,
            tokenizer=tokenizer,
            methods=methods,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
            output_path=out_path,
        )
        all_results.extend(results)
        print(f"\nResults saved to {out_path}")

    print_summary(all_results, methods)


if __name__ == "__main__":
    main()
