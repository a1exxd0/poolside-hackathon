"""
Benchmark dynamic tokenization on Laguna's official evaluation suite.

Laguna XS.2 reported scores (model card, April 2026):
  - SWE-bench Verified  68.2% pass@1
  - Terminal-Bench 2.0  30.1% pass@1  (dataset may not be on HuggingFace)

What this script measures
-------------------------
  - original_tokens        : tokens in the chat-formatted prompt
  - merged_tokens          : tokens after FVT boundary merging
  - shortening_factor      : original / merged  (higher = more compression)
  - boundary_time_s        : wall-clock time for boundary detection + FVT only
  - time_s                 : total wall-clock time (boundary + FVT + generation)
  - ttft_s                 : time to first generated token (from generation start)
  - attn_flops_reduction_pct : attention-layer FLOPs reduction (O(n²) scaling only;
                               excludes FFN layers which are O(n), so this is an
                               upper-bound / best-case estimate)
  - tok_per_s              : generation throughput
  - tps_ratio              : tok_per_s / baseline_tok_per_s
  - rouge_l                : ROUGE-L F1 vs baseline output
  - patch_valid            : heuristic unified-diff check

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
from rouge_score import rouge_scorer as rouge_lib

from dynamic_tokenizer import EmbeddingCache, compute_mmax, shortening_factor
from inference import (
    load_model_and_tokenizer,
    encode_chat_prompt,
    baseline_generate,
    dynamic_generate,
)

load_dotenv()

# Reported Laguna XS.2 scores (model card, April 2026) — for reference only
LAGUNA_SCORES = {
    "swe_verified":   0.682,
    "terminal_bench": 0.301,
}

_rouge = rouge_lib.RougeScorer(["rougeL"], use_stemmer=False)

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


# ---------------------------------------------------------------------------
# Benchmark dataset configs
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS = {
    "swe_verified": {
        "hf_path":      "princeton-nlp/SWE-bench_Verified",
        "split":        "test",
        "prompt_fn":    swe_bench_prompt,
        "system_prompt": SWE_SYSTEM,
        "id_field":     "instance_id",
        "description":  "SWE-bench Verified",
    },
    "terminal_bench": {
        "hf_path":      "terminal-bench/terminal-bench",
        "split":        "test",
        "prompt_fn":    terminal_bench_prompt,
        "system_prompt": TERMINAL_SYSTEM,
        "id_field":     "id",
        "description":  "Terminal-Bench 2.0",
    },
}

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def is_valid_patch(text: str) -> bool:
    """Heuristic: does the output look like a unified diff?"""
    return bool(re.search(r'^(diff --git|---|@@|\+\+\+)', text, re.MULTILINE))


def rouge_l(reference: str, hypothesis: str) -> float:
    """ROUGE-L F1 between two strings."""
    if not reference.strip() or not hypothesis.strip():
        return 0.0
    return _rouge.score(reference, hypothesis)["rougeL"].fmeasure


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
    out_dir: Path,
    run_ts: str,
    merge_pct: float = 0.5,
):
    cfg = BENCHMARK_CONFIGS[name]
    reported_score = LAGUNA_SCORES.get(name)
    build_prompt = cfg["prompt_fn"]
    system = cfg["system_prompt"]

    print(f"\n{'='*72}")
    print(f"Benchmark : {cfg['description']}")
    if reported_score is not None:
        print(f"Laguna XS.2 reported pass@1 (full harness): {reported_score*100:.1f}%")
    if "note" in cfg:
        print(f"Note      : {cfg['note']}")
    print(f"Methods   : {methods}   n={n}   max_new_tokens={max_new_tokens}")
    print(f"{'='*72}")

    try:
        ds = load_dataset(cfg["hf_path"], split=cfg["split"], token=os.getenv("HF_TOKEN"))
    except Exception as e:
        print(f"[SKIP] Could not load {cfg['hf_path']}: {e}")
        return []

    def pred_path(method_name: str) -> Path:
        return out_dir / f"{name}_{method_name}_{run_ts}.jsonl"

    out_files = {"baseline": open(pred_path("baseline"), "w", encoding="utf-8")}
    for m in methods:
        out_files[m] = open(pred_path(m), "w", encoding="utf-8")

    samples = list(ds.select(range(min(n, len(ds)))))
    cache = EmbeddingCache()
    results = []
    skipped = 0

    try:
        for i, row in enumerate(samples):
            instance_id = row.get(cfg["id_field"], f"sample_{i}")

            print(f"\n[{i+1}/{len(samples)}] {instance_id}")

            try:
                user_prompt = build_prompt(row)
                if not user_prompt.strip():
                    print("  [SKIP] empty prompt — unexpected row schema")
                    skipped += 1
                    continue

                input_ids = encode_chat_prompt(system, user_prompt, tokenizer, model)
                orig_len = input_ids.shape[0]
                print(f"  Prompt tokens : {orig_len}")

                baseline_resp, baseline_time, baseline_gen_tokens, baseline_ttft, baseline_mem_mb = \
                    baseline_generate(input_ids, model, tokenizer, max_new_tokens)
                baseline_tps = baseline_gen_tokens / baseline_time if baseline_time > 0 else 0
                baseline_valid = is_valid_patch(baseline_resp)
                print(f"  [baseline]  {baseline_time:.1f}s  TTFT {baseline_ttft:.2f}s  "
                      f"{baseline_gen_tokens} gen_tok  {baseline_tps:.1f} tok/s  "
                      f"mem +{baseline_mem_mb:.0f}MB  patch={'✓' if baseline_valid else '✗'}")

                out_files["baseline"].write(json.dumps({
                    "instance_id": instance_id,
                    "model_patch": baseline_resp,
                    "model_name_or_path": "laguna-xs2-baseline",
                    "original_tokens": orig_len,
                    "time_s": round(baseline_time, 3),
                    "ttft_s": round(baseline_ttft, 3),
                    "gen_tokens": baseline_gen_tokens,
                    "tok_per_s": round(baseline_tps, 2),
                    "peak_kv_mb": round(baseline_mem_mb, 1),
                    "patch_valid": baseline_valid,
                }) + "\n")
                out_files["baseline"].flush()

                record = {
                    "benchmark": name,
                    "instance_id": instance_id,
                    "original_tokens": orig_len,
                    "baseline_time_s": round(baseline_time, 3),
                    "baseline_ttft_s": round(baseline_ttft, 3),
                    "baseline_gen_tokens": baseline_gen_tokens,
                    "baseline_peak_kv_mb": round(baseline_mem_mb, 1),
                    "baseline_valid_patch": baseline_valid,
                    "methods": {},
                }

                for method in methods:
                    try:
                        # For the BPE path, compute mmax and derive m from merge_pct.
                        if method == "bpe":
                            mmax = compute_mmax(input_ids, tokenizer)
                            num_merges = max(1, int(mmax * merge_pct))
                        else:
                            mmax = None
                            num_merges = None

                        dyn_resp, merged_len, dyn_time, dyn_gen_tokens, dyn_ttft, dyn_mem_mb, boundary_time_s = \
                            dynamic_generate(input_ids, model, tokenizer, method, cache, max_new_tokens,
                                             num_merges=num_merges)
                        sf = shortening_factor(orig_len, merged_len)
                        dyn_tps = dyn_gen_tokens / dyn_time if dyn_time > 0 else 0
                        tps_ratio = dyn_tps / baseline_tps if baseline_tps > 0 else float("inf")
                        ttft_ratio = dyn_ttft / baseline_ttft if baseline_ttft > 0 else float("inf")
                        attn_flops_reduction = (1 - (merged_len / orig_len) ** 2) * 100
                        rl = rouge_l(baseline_resp, dyn_resp)
                        patch_ok = is_valid_patch(dyn_resp)
                        len_ratio = dyn_gen_tokens / baseline_gen_tokens if baseline_gen_tokens > 0 else float("inf")

                        print(f"  [{method}]  {merged_len}/{orig_len} tok  SF {sf:.2f}x  "
                              f"{attn_flops_reduction:.0f}% attn-FLOP↓  "
                              f"TTFT {dyn_ttft:.2f}s ({ttft_ratio:.2f}x)  "
                              f"{dyn_gen_tokens} gen_tok  {dyn_tps:.1f} tok/s  "
                              f"mem +{dyn_mem_mb:.0f}MB  ROUGE-L {rl:.2f}  patch={'✓' if patch_ok else '✗'}")

                        bpe_fields = (
                            {"mmax": mmax, "m_used": num_merges}
                            if method == "bpe" else {}
                        )
                        out_files[method].write(json.dumps({
                            "instance_id": instance_id,
                            "model_patch": dyn_resp,
                            "model_name_or_path": f"laguna-xs2-{method}",
                            "original_tokens": orig_len,
                            "merged_tokens": merged_len,
                            "shortening_factor": round(sf, 4),
                            "attn_flops_reduction_pct": round(attn_flops_reduction, 2),
                            "boundary_time_s": round(boundary_time_s, 3),
                            "time_s": round(dyn_time, 3),
                            "ttft_s": round(dyn_ttft, 3),
                            "ttft_ratio": round(ttft_ratio, 4),
                            "gen_tokens": dyn_gen_tokens,
                            "tok_per_s": round(dyn_tps, 2),
                            "tps_ratio": round(tps_ratio, 4),
                            "peak_kv_mb": round(dyn_mem_mb, 1),
                            "rouge_l": round(rl, 4),
                            "patch_valid": patch_ok,
                            "output_length_ratio": round(len_ratio, 4),
                            **bpe_fields,
                        }) + "\n")
                        out_files[method].flush()

                        record["methods"][method] = {
                            "merged_tokens": merged_len,
                            "shortening_factor": round(sf, 4),
                            "attn_flops_reduction_pct": round(attn_flops_reduction, 2),
                            "boundary_time_s": round(boundary_time_s, 3),
                            "time_s": round(dyn_time, 3),
                            "ttft_s": round(dyn_ttft, 3),
                            "ttft_ratio": round(ttft_ratio, 4),
                            "gen_tokens": dyn_gen_tokens,
                            "tok_per_s": round(dyn_tps, 2),
                            "tps_ratio": round(tps_ratio, 4),
                            "peak_kv_mb": round(dyn_mem_mb, 1),
                            "rouge_l": round(rl, 4),
                            "patch_valid": patch_ok,
                            "output_length_ratio": round(len_ratio, 4),
                            **bpe_fields,
                        }
                    except Exception as e:
                        print(f"  [{method}]  ERROR: {e}")
                        record["methods"][method] = {"error": str(e)}

                results.append(record)

            except Exception as e:
                print(f"  [SKIP] sample failed: {e}")
                skipped += 1
                continue

    finally:
        for f in out_files.values():
            f.close()

    if skipped:
        print(f"\n  {skipped}/{len(samples)} samples skipped due to errors.")

    print(f"\n  Predictions written (harness-ready):")
    for method_name in out_files:
        p = pred_path(method_name)
        print(f"    {p}")
    print(f"\n  To score with the official harness:")
    hf_path = cfg["hf_path"]
    for method_name in list(out_files):
        p = pred_path(method_name)
        print(f"    python -m swebench.harness.run_evaluation "
              f"--dataset_name {hf_path} "
              f"--predictions_path {p} "
              f"--run_id laguna-xs2-{method_name} "
              f"--max_workers 4")

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
        baseline_ttfts = [r.get("baseline_ttft_s", 0) for r in records]
        baseline_mem = [r.get("baseline_peak_kv_mb", 0) for r in records]
        baseline_valid = sum(1 for r in records if r.get("baseline_valid_patch"))
        print(f"  Avg prompt length  : {sum(orig_tokens)/n:.0f} tokens")
        print(f"  Baseline TTFT      : {sum(baseline_ttfts)/n:.2f}s")
        print(f"  Baseline peak mem  : +{sum(baseline_mem)/n:.0f}MB")
        print(f"  Baseline patch✓    : {baseline_valid}/{n}")

        for method in methods:
            valid = [r["methods"][method] for r in records
                     if method in r.get("methods", {})
                     and "shortening_factor" in r["methods"][method]]
            if not valid:
                continue
            avg_sf         = sum(v["shortening_factor"] for v in valid) / len(valid)
            avg_flops      = sum(v["attn_flops_reduction_pct"] for v in valid) / len(valid)
            avg_ttft       = sum(v["ttft_s"] for v in valid) / len(valid)
            avg_ttft_ratio = sum(v["ttft_ratio"] for v in valid) / len(valid)
            avg_tps        = sum(v["tok_per_s"] for v in valid) / len(valid)
            avg_mem        = sum(v["peak_kv_mb"] for v in valid) / len(valid)
            avg_rl         = sum(v["rouge_l"] for v in valid) / len(valid)
            avg_btime      = sum(v["boundary_time_s"] for v in valid) / len(valid)
            n_patch        = sum(1 for v in valid if v.get("patch_valid"))
            print(f"  [{method}]"
                  f"  SF {avg_sf:.2f}x"
                  f"  attn-FLOP↓ {avg_flops:.0f}%"
                  f"  TTFT {avg_ttft:.2f}s ({avg_ttft_ratio:.2f}x)"
                  f"  {avg_tps:.1f} tok/s"
                  f"  mem +{avg_mem:.0f}MB"
                  f"  ROUGE-L {avg_rl:.2f}"
                  f"  boundary {avg_btime:.3f}s"
                  f"  patch✓ {n_patch}/{len(valid)}")


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
        choices=["bpe", "whitespace", "entropy", "unigram", "all"],
        default="bpe",
        help=(
            "bpe: batch-level BPE merging (Feher et al. 2025, default); "
            "whitespace/entropy/unigram: legacy boundary-detection methods (Nawrot et al. 2023)"
        ),
    )
    parser.add_argument(
        "--merge-pct",
        type=float,
        default=0.5,
        dest="merge_pct",
        help=(
            "Fraction of mmax to use as the BPE merge count (bpe method only). "
            "0.5 = 50%% of mmax, matching the paper's XNLI setting (default: 0.5)."
        ),
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
        "--output_dir",
        type=str,
        default="predictions",
        help="Directory to write per-method prediction JSONL files (default: ./predictions)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    methods = ["bpe", "whitespace", "entropy", "unigram"] if args.method == "all" else [args.method]
    benchmarks = list(BENCHMARK_CONFIGS) if args.benchmark == "all" else [args.benchmark]

    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    for bench in benchmarks:
        results = run_benchmark(
            name=bench,
            model=model,
            tokenizer=tokenizer,
            methods=methods,
            n=args.n,
            max_new_tokens=args.max_new_tokens,
            out_dir=out_dir,
            run_ts=run_ts,
            merge_pct=args.merge_pct,
        )
        all_results.extend(results)

    print_summary(all_results, methods)


if __name__ == "__main__":
    main()
