"""Single-stream DECODE LATENCY on the vLLM serving path (the latency regime).

The batched HumanEval bench (`longctx_code_serving.py`) runs all prompts at once,
so decode happens at large batch — the regime where the fused INT4 read is still
~14-17x of FlashAttention and the bf16-tensor-core win is smallest.  This script
measures the OTHER regime: one request at a time (`max_num_seqs=1`), long context,
pure-decode steps.  That is where the batch-1 kernel speedup must show up if it is
worth anything, and it is a real serving regime (interactive / low-concurrency
single-stream latency).

We report median per-output-token decode latency (ms/tok) and tokens/s over R
requests of a fixed long prefix + fixed decode length.  Run once per dtype:
  KVD=auto       -> bf16 KV cache (ceiling)
  KVD=int4_kivi  -> our INT4-KIVI backend (swap the OLD kernel in to A/B)

Env: KVD, PREFIX_TOKENS (12000), MAXNEW (256), R (8 requests), MODEL.
"""
import glob
import json
import os
import statistics
import time

from vllm import LLM, SamplingParams

ROOT = "/home/alex/poolside-hackathon-kv-quant"
MODEL = os.environ.get("MODEL", "poolside/Laguna-XS.2")
KVD = os.environ.get("KVD", "int4_kivi")
PREFIX_TOKENS = int(os.environ.get("PREFIX_TOKENS", "12000"))
MAXNEW = int(os.environ.get("MAXNEW", "256"))
R = int(os.environ.get("R", "8"))


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


# max_num_seqs=1 -> the engine can only ever run ONE sequence at a time, so every
# decode step is batch-1: the latency regime the batch-1 kernel win targets.
llm = LLM(model=MODEL, dtype="bfloat16", kv_cache_dtype=KVD,
          gpu_memory_utilization=0.55, max_model_len=PREFIX_TOKENS + 2048,
          max_num_seqs=1, enforce_eager=True)
tok = llm.get_tokenizer()
prefix_text, prefix_tok = build_prefix_text(tok, PREFIX_TOKENS)

# Fixed decode length so latency is comparable across dtypes: ignore EOS, force
# exactly MAXNEW tokens.  Greedy.
sp = SamplingParams(temperature=0.0, max_tokens=MAXNEW, min_tokens=MAXNEW,
                    ignore_eos=True)

base_msgs = [
    {"role": "system", "content": "You are a Python coding assistant."},
    {"role": "user", "content":
        "Here is some reference Python source code for context.\n\n"
        f"```python\n{prefix_text}\n```\n\n"
        "Now write a Python function `solve(n)` that returns the n-th Fibonacci "
        "number, with a docstring and an iterative implementation."},
]

# Warmup (compile kernels / JIT) — not timed.
llm.chat([base_msgs], sp, add_generation_prompt=True)

per_req_ms_tok = []
per_req_gen_s = []
out_tokens = []
for r in range(R):
    t0 = time.time()
    outs = llm.chat([base_msgs], sp, add_generation_prompt=True)
    dt = time.time() - t0
    n_out = len(outs[0].outputs[0].token_ids)
    per_req_gen_s.append(dt)
    out_tokens.append(n_out)
    per_req_ms_tok.append(dt / max(n_out, 1) * 1e3)

med_ms = statistics.median(per_req_ms_tok)
med_s = statistics.median(per_req_gen_s)
tok_s = 1000.0 / med_ms
print(f"=== [{KVD}] single-stream decode latency (max_num_seqs=1) ===")
print(f"  prefix={prefix_tok} tok  decode={MAXNEW} tok  R={R} requests")
print(f"  median {med_ms:.3f} ms/tok   {tok_s:.1f} tok/s   "
      f"req {med_s:.2f}s  (out tok {min(out_tokens)}..{max(out_tokens)})")
json.dump({"kvd": KVD, "prefix_tok": prefix_tok, "maxnew": MAXNEW, "R": R,
           "median_ms_per_tok": med_ms, "tok_per_s": tok_s,
           "median_req_s": med_s, "per_req_ms_tok": per_req_ms_tok},
          open(f"/tmp/decode_latency_{KVD}.json", "w"))
print(f"DECODE_LATENCY_SERVING DONE [{KVD}]")
