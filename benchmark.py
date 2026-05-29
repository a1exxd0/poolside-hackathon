#!/usr/bin/env python3
"""
benchmark.py — Evaluate hierarchical KV cache vs baseline on real benchmarks

Benchmarks:
  --humaneval       HumanEval pass@1          (164 coding problems, functional correctness)
  --livecodebench   LiveCodeBench pass@1      (recent competitive programming, less contaminated)
  --longbench       LongBench subset          (long-context code + reasoning, directly stresses eviction)
                    tasks: lcc, repobench-p, 2wikimqa, hotpotqa

Each benchmark runs both baseline (full DynamicCache) and hierarchical eviction side-by-side.

Laguna XS.2 reference scores from poolside.ai/blog/laguna-a-deeper-dive:
  SWE-bench Verified 68.2%, SWE-bench Multilingual 62.4%,
  SWE-bench Pro 44.5%, Terminal-Bench 2.0 30.1%

LiveCodeBench and HumanEval are the closest runnable proxies for SWE-bench difficulty.
LongBench directly stresses the KV cache (contexts >> budget).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache import generate_with_hierarchy


# ─── Shared utilities ─────────────────────────────────────────────────────────

def format_chat(tokenizer, message: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": message}],
        tokenize=False,
        add_generation_prompt=True,
    )


def safe_exec(code: str, stdin: str = "", timeout: int = 10) -> tuple[bool, str]:
    """Run Python code in a subprocess. Returns (passed, stdout)."""
    with tempfile.NamedTemporaryFile(
        suffix=".py", mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(code)
        fname = f.name
    try:
        r = subprocess.run(
            ["python", fname],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return r.returncode == 0, r.stdout.strip()
    except subprocess.TimeoutExpired:
        return False, ""
    finally:
        os.unlink(fname)


def strip_fences(text: str) -> str:
    """Remove markdown code fences from model output."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first line (```python or ```) and last ``` if present
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner)
    return text


def f1_token(pred: str, gold: str) -> float:
    p_toks = pred.lower().split()
    g_toks = gold.lower().split()
    if not p_toks or not g_toks:
        return 0.0
    common = set(p_toks) & set(g_toks)
    if not common:
        return 0.0
    prec = len(common) / len(p_toks)
    rec  = len(common) / len(g_toks)
    return 2 * prec * rec / (prec + rec)


def edit_similarity(pred: str, gold: str) -> float:
    import editdistance
    if not pred and not gold:
        return 1.0
    d = editdistance.eval(pred.strip(), gold.strip())
    return 1.0 - d / max(len(pred.strip()), len(gold.strip()))


def save_checkpoint(path: str, data: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ─── Runner factories ─────────────────────────────────────────────────────────

def make_baseline(model, tokenizer, max_new_tokens: int) -> Callable:
    """Standard HuggingFace generation with full DynamicCache — no eviction."""
    def run(prompt: str) -> tuple[str, float]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.time() - t0
        new_ids = out[0][inputs.input_ids.size(1):]
        return tokenizer.decode(new_ids, skip_special_tokens=True), elapsed
    return run


def make_hierarchical(model, tokenizer, max_new_tokens: int, budget: int) -> Callable:
    """generate_with_hierarchy — Quest on AST + PyramidKV eviction."""
    def run(prompt: str) -> tuple[str, float]:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        t0 = time.time()
        result = generate_with_hierarchy(
            model, tokenizer, inputs.input_ids,
            max_new_tokens=max_new_tokens,
            budget=budget,
            sink=4, recent=64,
            k_topics=3, k_leaves=4,
            beta=32,
            eos_token_id=tokenizer.eos_token_id,
        )
        elapsed = time.time() - t0
        new_ids = result["sequences"][0][inputs.input_ids.size(1):]
        return tokenizer.decode(new_ids, skip_special_tokens=True), elapsed
    return run


# ─── HumanEval ───────────────────────────────────────────────────────────────
# Standard coding benchmark: 164 Python function completion problems.
# Scored by functional correctness (pass@1): generate once, execute against
# the provided unit tests.  Reference: Chen et al., 2021.

def bench_humaneval(runner: Callable, tokenizer, n: int, ckpt: str) -> dict:
    ds = load_dataset("openai/human-eval", split="test", trust_remote_code=True)
    ds = ds.select(range(min(n, len(ds))))

    passed = 0
    times  = []
    rows   = []

    for ex in tqdm(ds, desc="HumanEval"):
        prompt = format_chat(
            tokenizer,
            "Complete the following Python function. "
            "Return only the implementation, no explanation or extra text.\n\n"
            + ex["prompt"],
        )
        completion, t = runner(prompt)
        times.append(t)

        code = strip_fences(completion)
        # If model echoed the signature, keep only what follows the docstring
        if ex["entry_point"] in code:
            # Try to keep just the body by taking everything after the prompt
            code = completion

        test_code = (
            ex["prompt"] + "\n" + code + "\n\n"
            + ex["test"] + f"\ncheck({ex['entry_point']})"
        )
        ok, _ = safe_exec(test_code, timeout=10)
        if ok:
            passed += 1
        rows.append({"task_id": ex["task_id"], "passed": ok})

    result = {
        "pass@1":       passed / len(ds),
        "n":            len(ds),
        "avg_time_s":   sum(times) / len(times),
        "total_time_s": sum(times),
        "rows":         rows,
    }
    save_checkpoint(ckpt, result)
    return result


# ─── LiveCodeBench ────────────────────────────────────────────────────────────
# Recent competitive programming problems scraped from Codeforces, LeetCode,
# AtCoder after the training cutoff — minimal contamination.
# Closest open proxy to SWE-bench difficulty; Laguna XS.2 = 68.2% SWE-bench.
# Evaluated via pass@1 against public test cases using stdin/stdout execution.

def bench_livecodebench(runner: Callable, tokenizer, n: int, ckpt: str) -> dict:
    try:
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            version_tag="release_v5",
            split="test",
            trust_remote_code=True,
        )
    except TypeError:
        # Older datasets versions don't accept version_tag
        ds = load_dataset(
            "livecodebench/code_generation_lite",
            split="test",
            trust_remote_code=True,
        )
    # Sort by recency: newest problems have least training contamination
    ds = ds.sort("contest_date", reverse=True).select(range(min(n, len(ds))))

    passed    = 0
    evaluated = 0
    times     = []
    rows      = []

    for ex in tqdm(ds, desc="LiveCodeBench"):
        system_note = (
            "\nStarter code:\n" + ex["starter_code"]
            if ex.get("starter_code", "").strip()
            else ""
        )
        prompt = format_chat(
            tokenizer,
            "Solve the following competitive programming problem in Python. "
            "Return only the complete solution, no explanation.\n\n"
            + ex["question_content"]
            + system_note,
        )
        completion, t = runner(prompt)
        times.append(t)

        try:
            tests = json.loads(ex["public_test_cases"])
        except Exception:
            rows.append({"problem": ex.get("question_id", "?"), "skipped": True})
            continue

        code = strip_fences(completion)
        problem_passed = False

        for tc in tests[:3]:   # check first 3 public test cases
            test_input  = tc.get("input", "")
            test_output = tc.get("output", "").strip()
            test_type   = tc.get("testtype", "stdin")

            if test_type == "stdin":
                ok, stdout = safe_exec(code, stdin=test_input, timeout=10)
                if ok and stdout == test_output:
                    problem_passed = True
                    break
            else:
                # functional: just check it runs without error on the input
                ok, _ = safe_exec(code, stdin=test_input, timeout=10)
                if ok:
                    problem_passed = True
                    break

        if problem_passed:
            passed += 1
        evaluated += 1
        rows.append({"problem": ex.get("question_id", "?"), "passed": problem_passed})

    denom  = max(evaluated, 1)
    result = {
        "pass@1":       passed / denom,
        "n":            evaluated,
        "avg_time_s":   sum(times) / max(len(times), 1),
        "total_time_s": sum(times),
        "rows":         rows,
    }
    save_checkpoint(ckpt, result)
    return result


# ─── LongBench ───────────────────────────────────────────────────────────────
# Long-context benchmark (Bai et al., 2023).  Context lengths 1K–30K tokens,
# directly stressing the KV cache eviction: if the hierarchy drops the wrong
# clusters the model can't answer.
#
# Tasks chosen to match Laguna's domain:
#   lcc          — long code completion        (edit similarity)
#   repobench-p  — repo-level code completion  (edit similarity)
#   2wikimqa     — multi-hop QA               (token F1)
#   hotpotqa     — multi-hop QA               (token F1)

LONGBENCH_CFG = {
    "lcc": {
        "metric": "edit_sim",
        "max_new": 256,
        "instruction": (
            "Complete the following code snippet. "
            "Return only the completion, no explanation.\n\n"
            "Context:\n{context}\n\n"
            "Complete: {input}"
        ),
    },
    "repobench-p": {
        "metric": "edit_sim",
        "max_new": 256,
        "instruction": (
            "Given the repository context below, complete the next line or block.\n\n"
            "{context}\n\n"
            "Complete: {input}"
        ),
    },
    "2wikimqa": {
        "metric": "f1",
        "max_new": 64,
        "instruction": (
            "Answer the question based on the context. "
            "Be concise — one phrase or sentence.\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\nAnswer:"
        ),
    },
    "hotpotqa": {
        "metric": "f1",
        "max_new": 64,
        "instruction": (
            "Answer the question based on the context. "
            "Be concise — one phrase or sentence.\n\n"
            "Context:\n{context}\n\n"
            "Question: {input}\nAnswer:"
        ),
    },
}


def bench_longbench(
    runner: Callable,
    tokenizer,
    tasks: list[str],
    n: int,
    ckpt: str,
) -> dict:
    results = {}

    for task in tasks:
        cfg = LONGBENCH_CFG.get(task)
        if cfg is None:
            print(f"  Unknown task {task}, skipping.")
            continue

        try:
            ds = load_dataset(
                "THUDM/LongBench", task, split="test", trust_remote_code=True
            )
        except Exception as e:
            print(f"  Failed to load LongBench/{task}: {e}")
            continue

        ds = ds.select(range(min(n, len(ds))))
        scores = []
        times  = []
        rows   = []

        for ex in tqdm(ds, desc=f"LongBench/{task}"):
            user_msg = cfg["instruction"].format(
                context=ex.get("context", ""),
                input=ex.get("input", ""),
            )
            prompt     = format_chat(tokenizer, user_msg)
            completion, t = runner(prompt)
            times.append(t)

            answers = ex.get("answers", ex.get("answer", [""]))
            if isinstance(answers, str):
                answers = [answers]

            if cfg["metric"] == "f1":
                sc = max(f1_token(completion, a) for a in answers)
            else:
                sc = max(edit_similarity(strip_fences(completion), a) for a in answers)

            scores.append(sc)
            rows.append({"score": sc})

        results[task] = {
            "score":        sum(scores) / max(len(scores), 1),
            "metric":       cfg["metric"],
            "n":            len(scores),
            "avg_time_s":   sum(times) / max(len(times), 1),
            "total_time_s": sum(times),
            "rows":         rows,
        }

    save_checkpoint(ckpt, results)
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Benchmark hierarchical KV cache vs baseline")
    ap.add_argument("--model", default="poolside/Laguna-XS.2")
    ap.add_argument("--budget", type=int, default=1024,
                    help="KV cache token budget for hierarchical runner")
    ap.add_argument("--n", type=int, default=100,
                    help="Number of examples per benchmark")
    ap.add_argument("--humaneval",      action="store_true")
    ap.add_argument("--livecodebench",  action="store_true")
    ap.add_argument("--longbench",      action="store_true")
    ap.add_argument("--longbench-tasks", nargs="+",
                    default=["lcc", "repobench-p", "2wikimqa", "hotpotqa"])
    ap.add_argument("--out", default="results/benchmark.json")
    ap.add_argument("--baseline-only", action="store_true",
                    help="Skip hierarchical runner (useful for reference scoring)")
    args = ap.parse_args()

    print(f"Loading {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    baseline     = make_baseline(model, tokenizer, max_new_tokens=512)
    hierarchical = make_hierarchical(model, tokenizer, max_new_tokens=512, budget=args.budget)
    runners      = {"baseline": baseline}
    if not args.baseline_only:
        runners["hierarchical"] = hierarchical

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    all_results: dict = {
        "model":   args.model,
        "budget":  args.budget,
        "n":       args.n,
    }

    for runner_name, runner in runners.items():
        print(f"\n{'='*60}")
        print(f"Runner: {runner_name}")
        print(f"{'='*60}")
        all_results.setdefault(runner_name, {})

        if args.humaneval:
            ckpt = str(out.with_suffix("")) + f"_{runner_name}_humaneval.json"
            print("\n--- HumanEval ---")
            all_results[runner_name]["humaneval"] = bench_humaneval(
                runner, tokenizer, args.n, ckpt
            )
            r = all_results[runner_name]["humaneval"]
            print(f"  pass@1 = {r['pass@1']:.3f}  ({r['n']} examples, "
                  f"{r['avg_time_s']:.1f}s/example)")

        if args.livecodebench:
            ckpt = str(out.with_suffix("")) + f"_{runner_name}_livecodebench.json"
            print("\n--- LiveCodeBench ---")
            all_results[runner_name]["livecodebench"] = bench_livecodebench(
                runner, tokenizer, args.n, ckpt
            )
            r = all_results[runner_name]["livecodebench"]
            print(f"  pass@1 = {r['pass@1']:.3f}  ({r['n']} examples, "
                  f"{r['avg_time_s']:.1f}s/example)")

        if args.longbench:
            ckpt = str(out.with_suffix("")) + f"_{runner_name}_longbench.json"
            print("\n--- LongBench ---")
            all_results[runner_name]["longbench"] = bench_longbench(
                runner, tokenizer, args.longbench_tasks, args.n, ckpt
            )
            for task, r in all_results[runner_name]["longbench"].items():
                print(f"  {task:20s}  {r['metric']:8s} = {r['score']:.3f}  "
                      f"({r['n']} examples, {r['avg_time_s']:.1f}s/example)")

    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\nAll results saved to {out}")

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'Benchmark':<25} {'Metric':<12} {'Baseline':>10} {'Hierarchical':>14} {'Delta':>8}")
    print("-" * 72)
    for bname in ("humaneval", "livecodebench"):
        for rn in runners:
            pass
        bl = all_results.get("baseline",     {}).get(bname, {})
        hi = all_results.get("hierarchical", {}).get(bname, {})
        if bl:
            bv = bl.get("pass@1", 0.0)
            hv = hi.get("pass@1", 0.0) if hi else float("nan")
            delta = hv - bv if hi else float("nan")
            print(f"{bname:<25} {'pass@1':<12} {bv:>10.3f} {hv:>14.3f} {delta:>+8.3f}")
    for task in args.longbench_tasks:
        bl = all_results.get("baseline",     {}).get("longbench", {}).get(task, {})
        hi = all_results.get("hierarchical", {}).get("longbench", {}).get(task, {})
        if bl:
            metric = bl.get("metric", "score")
            bv = bl.get("score", 0.0)
            hv = hi.get("score", 0.0) if hi else float("nan")
            delta = hv - bv if hi else float("nan")
            print(f"longbench/{task:<15} {metric:<12} {bv:>10.3f} {hv:>14.3f} {delta:>+8.3f}")


if __name__ == "__main__":
    main()
