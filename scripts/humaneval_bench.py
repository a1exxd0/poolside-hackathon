"""HumanEval subset: pass@1 for BF16 vs INT4-simulated KV cache on Laguna-XS.2.

Loads the first --n HumanEval problems, generates completions under two cache
regimes, executes each solution against the reference test suite, and reports
pass@1 for both modes.

INT4 simulation: after every decode step, each layer's K and V tensors are
quantized to INT4 (MSE-optimal blockwise scale) and immediately dequantized back
— the worst-case accuracy test; errors accumulate across the full generation.

Usage:
    python -m scripts.humaneval_bench [--n 20] [--max-new 512]
"""
from __future__ import annotations

import argparse
import contextlib
import io
import re
import signal
import subprocess
import sys
import tempfile
import textwrap
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "/home/alex/poolside-hackathon-kv-quant")
from kv_quant import BLOCK, _mse_optimal_scale, quantize_block, dequantize_block

MODEL = "poolside/Laguna-XS.2"

# ---------------------------------------------------------------------------
# INT4 simulation helpers
# ---------------------------------------------------------------------------

def _int4_round_trip(x: torch.Tensor) -> torch.Tensor:
    B, H, S, D = x.shape
    xf = x.float().reshape(B, H, S, D // BLOCK, BLOCK)
    s = _mse_optimal_scale(xf)
    return dequantize_block(quantize_block(xf, s), s).reshape(B, H, S, D).to(x.dtype)


def _quantize_cache(cache: DynamicCache) -> None:
    for layer in cache.layers:
        layer.keys   = _int4_round_trip(layer.keys)
        layer.values = _int4_round_trip(layer.values)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate(model, input_ids: torch.Tensor, max_new: int, int4: bool) -> str:
    tok_obj = getattr(model, "_tokenizer_ref", None)
    device = input_ids.device
    cache = DynamicCache()
    L = input_ids.shape[1]

    with torch.no_grad():
        if int4:
            # Manual decode loop so we can quantize the cache each step.
            cp = torch.arange(L, device=device)
            out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                        cache_position=cp, position_ids=cp.unsqueeze(0))
            _quantize_cache(cache)
            tokens = [out.logits[0, -1].argmax().item()]
            abs_pos = L
            eos = getattr(model.config, "eos_token_id", None)
            eos_set = set(eos) if isinstance(eos, (list, tuple)) else ({eos} if eos else set())
            for _ in range(max_new - 1):
                cp2 = torch.tensor([abs_pos], device=device)
                out = model(input_ids=torch.tensor([[tokens[-1]]], device=device),
                            past_key_values=cache, use_cache=True,
                            cache_position=cp2, position_ids=cp2.unsqueeze(0))
                _quantize_cache(cache)
                t = out.logits[0, -1].argmax().item()
                tokens.append(t)
                abs_pos += 1
                if t in eos_set:
                    break
            return tokens
        else:
            out = model.generate(input_ids, max_new_tokens=max_new,
                                 past_key_values=cache, use_cache=True, do_sample=False)
            return out[0, L:].tolist()


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

def extract_code(response: str, prompt: str) -> str:
    """Return the best completion string to append to the HumanEval prompt."""
    # Prefer fenced code blocks
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if m:
        block = m.group(1)
        # If the block re-states the function signature, use it wholesale.
        if prompt.split("def ", 1)[-1].split("(")[0].strip() in block:
            return block
        return prompt + block

    # Otherwise, strip everything before the first indented line (function body).
    lines = response.splitlines()
    body = []
    in_body = False
    for line in lines:
        if not in_body:
            if line.startswith("    ") or line.startswith("\t"):
                in_body = True
        if in_body:
            # Stop at a new top-level def (next function)
            if line.startswith("def ") and body:
                break
            body.append(line)
    if body:
        return prompt + "\n".join(body) + "\n"

    # Fallback: append raw response
    return prompt + response


# ---------------------------------------------------------------------------
# Test execution
# ---------------------------------------------------------------------------

_TIMEOUT = 10  # seconds per test

def run_tests(solution_code: str, test_code: str, entry_point: str) -> tuple[bool, str]:
    """Execute solution + tests in a subprocess. Returns (passed, error_msg)."""
    full = solution_code + "\n\n" + test_code + f"\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        r = subprocess.run(
            [sys.executable, fname],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        if r.returncode == 0:
            return True, ""
        return False, (r.stderr or r.stdout).strip()[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:
        return False, str(e)
    finally:
        import os; os.unlink(fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",        type=int, default=20,  help="number of HumanEval problems")
    ap.add_argument("--max-new",  type=int, default=512, help="max new tokens per completion")
    ap.add_argument("--bf16-only", action="store_true",  help="skip INT4 simulation")
    args = ap.parse_args()

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device

    print(f"[data] loading HumanEval ({args.n} problems) ...", flush=True)
    ds = load_dataset("openai/openai_humaneval", split="test").select(range(args.n))

    modes = ["bf16"] + ([] if args.bf16_only else ["int4"])
    results: dict[str, list[bool]] = {m: [] for m in modes}
    errors:  dict[str, list[str]]  = {m: [] for m in modes}

    print(f"\n{'':─<72}")
    print(f"  {'#':>3}  {'task_id':<30}  {'BF16':>6}  {'INT4':>6}  {'note'}")
    print(f"{'':─<72}")

    for idx, prob in enumerate(ds):
        task   = prob["task_id"]
        prompt = prob["prompt"]          # function sig + docstring
        tests  = prob["test"]            # check() function body
        entry  = prob["entry_point"]

        sys_msg = (
            "You are a Python coding assistant. Complete the function below. "
            "Return a fenced ```python``` code block containing the complete "
            "function (including signature and docstring)."
        )
        user_msg = f"Complete this Python function:\n\n```python\n{prompt}```"
        msgs = [{"role": "system", "content": sys_msg},
                {"role": "user",   "content": user_msg}]
        input_ids = tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False,
        ).to(device)

        row_pass, row_note = {}, {}
        for mode in modes:
            t0 = time.time()
            token_ids = _generate(model, input_ids, args.max_new, int4=(mode == "int4"))
            elapsed = time.time() - t0
            response = tok.decode(token_ids, skip_special_tokens=True)
            code = extract_code(response, prompt)
            passed, err = run_tests(code, tests, entry)
            results[mode].append(passed)
            errors[mode].append("" if passed else err[:80])
            row_pass[mode] = passed
            row_note[mode] = f"{elapsed:.0f}s"

        bf16_sym = "✓" if row_pass.get("bf16") else "✗"
        int4_sym  = ("✓" if row_pass.get("int4") else "✗") if "int4" in modes else "─"
        print(f"  {idx+1:>3}  {task:<30}  {bf16_sym:>6}  {int4_sym:>6}  {row_note.get('bf16','')}", flush=True)

    print(f"{'':─<72}")

    # Summary
    print(f"\n{'':═<72}")
    print("  RESULTS")
    print(f"{'':═<72}")
    for mode in modes:
        passed = sum(results[mode])
        n = len(results[mode])
        fails = [ds[i]["task_id"] for i, p in enumerate(results[mode]) if not p]
        print(f"  {mode.upper():<8}  pass@1 = {passed}/{n}  ({100*passed/n:.0f}%)")
        if fails:
            print(f"            failed: {', '.join(fails)}")
    print()

    if "bf16" in modes and "int4" in modes:
        agree = sum(a == b for a, b in zip(results["bf16"], results["int4"]))
        print(f"  BF16 vs INT4 agreement: {agree}/{args.n} problems ({100*agree/args.n:.0f}%)")
        # Problems where INT4 passed but BF16 failed (or vice versa)
        diff = [(ds[i]["task_id"], results["bf16"][i], results["int4"][i])
                for i in range(args.n) if results["bf16"][i] != results["int4"][i]]
        if diff:
            print("  Differences:")
            for tid, b, q in diff:
                print(f"    {tid}  bf16={'✓' if b else '✗'}  int4={'✓' if q else '✗'}")
    print()


if __name__ == "__main__":
    main()
