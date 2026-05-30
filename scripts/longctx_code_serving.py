"""Long-context CODING benchmark on the vLLM SERVING path: executed HumanEval
pass@1 for bf16 vs our custom INT4-KIVI KV cache.

Unlike the exact-integer needle (which a single quantized-logit digit-flip can
fail), this measures *executed* pass@1 -- the model must emit a function that
actually passes the hidden unit tests.  That is the metric KV-cache quant has to
preserve, and it is far less brittle than digit-exact recall.

Two regimes, identical prompts for both dtypes (so it is apples-to-apples):
  * short : plain HumanEval (KV cache ~hundreds of tokens at decode start).
  * long  : the SAME problems with a long, in-distribution Python-source prefix
            prepended (real ``transformers`` modeling_*.py), so the model must
            decode while attending back over a fully-quantized long context.

Run once per dtype (mirrors needle_serving.py):
  KVD=auto        -> bf16 KV cache (ceiling)
  KVD=int4_kivi   -> our custom INT4-KIVI backend
N, PREFIX_TOKENS, MAXNEW are env-overridable.  Greedy (temperature 0).
"""
import glob
import json
import os
import re
import subprocess
import sys
import tempfile
import time

from datasets import load_dataset
from vllm import LLM, SamplingParams

ROOT = "/home/alex/poolside-hackathon-kv-quant"
MODEL = "poolside/Laguna-XS.2"

KVD = os.environ.get("KVD", "int4_kivi")
N = int(os.environ.get("N", "20"))
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "12000"))
MAXNEW = int(os.environ.get("MAXNEW", "256"))


# --------------------------------------------------------------------------- #
# Code extraction + execution  (verbatim from scripts/humaneval_bench.py)
# --------------------------------------------------------------------------- #
def extract_code(response: str, prompt: str) -> str:
    m = re.search(r"```(?:python)?\n(.*?)```", response, re.DOTALL)
    if m:
        block = m.group(1)
        if prompt.split("def ", 1)[-1].split("(")[0].strip() in block:
            return block
        return prompt + block
    lines = response.splitlines()
    body, in_body = [], False
    for line in lines:
        if not in_body and (line.startswith("    ") or line.startswith("\t")):
            in_body = True
        if in_body:
            if line.startswith("def ") and body:
                break
            body.append(line)
    if body:
        return prompt + "\n".join(body) + "\n"
    return prompt + response


def run_tests(solution_code: str, test_code: str, entry_point: str):
    full = solution_code + "\n\n" + test_code + f"\ncheck({entry_point})\n"
    with tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False) as f:
        f.write(full)
        fname = f.name
    try:
        r = subprocess.run(
            [sys.executable, fname], capture_output=True, text=True, timeout=10
        )
        if r.returncode == 0:
            return True, ""
        return False, (r.stderr or r.stdout).strip()[-300:]
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    finally:
        try:
            os.unlink(fname)
        except OSError:
            pass


# --------------------------------------------------------------------------- #
# Long in-distribution code prefix (real transformers source -> in-dist)
# --------------------------------------------------------------------------- #
def build_prefix_text(tok, target_tokens: int):
    files = []
    for venv in (".venv-vllm", ".venv"):
        files = sorted(glob.glob(f"{ROOT}/{venv}/**/transformers/**/modeling_*.py",
                                 recursive=True))
        if files:
            break
    if not files:
        files = sorted(glob.glob(f"{ROOT}/**/*.py", recursive=True))
    texts = []
    for f in files:
        try:
            texts.append(open(f).read())
        except OSError:
            continue
        ids = tok("\n\n".join(texts))["input_ids"]
        if len(ids) >= target_tokens:
            return tok.decode(ids[:target_tokens]), target_tokens
    ids = tok("\n\n".join(texts))["input_ids"]
    return tok.decode(ids), len(ids)


# --------------------------------------------------------------------------- #
SYS_MSG = (
    "You are a Python coding assistant. Complete the function below. Return a "
    "fenced ```python``` code block containing the complete function (including "
    "signature and docstring)."
)


def build_msgs(prompt, prefix_text):
    if prefix_text:
        user = ("Here is some reference Python source code for context. You do "
                "not need to use it; it is provided only as background.\n\n"
                f"```python\n{prefix_text}\n```\n\n"
                "Now, ignoring the reference above, complete this Python "
                f"function:\n\n```python\n{prompt}```")
    else:
        user = f"Complete this Python function:\n\n```python\n{prompt}```"
    return [{"role": "system", "content": SYS_MSG},
            {"role": "user", "content": user}]


# --------------------------------------------------------------------------- #
llm = LLM(model=MODEL, dtype="bfloat16", kv_cache_dtype=KVD,
          gpu_memory_utilization=0.55, max_model_len=PREFIX_TOKENS + 2048,
          enforce_eager=True)
tok = llm.get_tokenizer()

prefix_text, prefix_tok = build_prefix_text(tok, PREFIX_TOKENS)
ds = load_dataset("openai/openai_humaneval", split="test").select(range(N))
sp = SamplingParams(temperature=0.0, max_tokens=MAXNEW)

summary = {}
for regime in ("short", "long"):
    pfx = prefix_text if regime == "long" else None
    convs = [build_msgs(p["prompt"], pfx) for p in ds]
    t0 = time.time()
    outs = llm.chat(convs, sp, add_generation_prompt=True)
    gen_s = time.time() - t0
    ctx = [len(o.prompt_token_ids) for o in outs]

    npass = 0
    for prob, o in zip(ds, outs):
        sol = extract_code(o.outputs[0].text, prob["prompt"])
        ok, _ = run_tests(sol, prob["test"], prob["entry_point"])
        npass += int(ok)
    summary[regime] = {"pass": npass, "n": len(ds),
                       "ctx_min": min(ctx), "ctx_max": max(ctx), "gen_s": gen_s}
    print(f"=== [{KVD}] {regime.upper()} HumanEval pass@1 ===")
    print(f"  {npass}/{len(ds)} ({100*npass/len(ds):.0f}%)  "
          f"ctx {min(ctx)}..{max(ctx)}  gen {gen_s:.0f}s")

json.dump({"kvd": KVD, "prefix_tok": prefix_tok, "summary": summary},
          open(f"/tmp/longctx_code_serving_{KVD}.json", "w"))
print(f"LONGCTX_CODE_SERVING DONE [{KVD}]")
