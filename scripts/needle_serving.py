"""Stable long-context needle retrieval on the vLLM SERVING path.

Same construction as the throwaway /tmp probe, but with enough trials per length
to get a low-noise bf16-vs-int4_kivi comparison (the n=5 probe could not resolve
a ~20% K-error change from sampling noise).  Exact-integer graded.

Run once per dtype:
  KVD=auto        -> bf16 KV cache (ceiling)
  KVD=int4_kivi   -> our custom INT4-KIVI backend
TRIALS and LENGTHS are env-overridable.
"""
import os, json, random, time
from vllm import LLM, SamplingParams

KVD = os.environ.get("KVD", "int4_kivi")
TRIALS = int(os.environ.get("TRIALS", "20"))
LENGTHS = [int(x) for x in os.environ.get("LENGTHS", "8000,16000,32000").split(",")]
DEPTH = 0.06  # needle near the start (in a frozen, fully-quantized region)

llm = LLM(model="poolside/Laguna-XS.2", dtype="bfloat16", kv_cache_dtype=KVD,
          gpu_memory_utilization=0.55, max_model_len=34000, enforce_eager=True)
tok = llm.get_tokenizer()


def filler(rng, n):
    out = []
    for i in range(n):
        k = rng.randint(2, 99)
        out.append(f"def _f{i}(x):\n    # assorted helper {i}\n    return x * {k} + {i % 7}\n")
    return "\n".join(out)


def build(length_tok, depth, magic_id, magic_val, rng):
    needle = f"MAGIC_SEED_{magic_id} = {magic_val}  # unique constant, remember it\n\n"
    blocks, ntok, placed = [], 0, False
    while ntok < length_tok:
        if not placed and ntok >= depth * length_tok:
            blocks.append(needle); placed = True
        b = filler(rng, 20)
        blocks.append(b)
        ntok += len(tok(b)["input_ids"])
    if not placed:
        blocks.insert(1, needle)
    body = "\n".join(blocks)
    q = (f"\n\n# End of module. Recall the unique constant defined far above.\n"
         f"# Write only the integer value assigned to MAGIC_SEED_{magic_id}.\n"
         f"# MAGIC_SEED_{magic_id} =")
    return body + q


prompts, meta = [], []
rng = random.Random(0)
for L in LENGTHS:
    for t in range(TRIALS):
        mid = f"{L}_{t}"
        val = rng.randint(1000003, 9999991)
        prompts.append(build(L, DEPTH, mid, val, rng))
        meta.append((L, val))

ctx_tok = [len(tok(p)["input_ids"]) for p in prompts]
print(f"[{KVD}] {len(prompts)} prompts ({TRIALS}/len), ctx {min(ctx_tok)}..{max(ctx_tok)}")
sp = SamplingParams(temperature=0.0, max_tokens=12)
t0 = time.time()
outs = llm.generate(prompts, sp)
gen_s = time.time() - t0

res = {L: {"pass": 0, "n": 0} for L in LENGTHS}
for (L, val), o in zip(meta, outs):
    txt = o.outputs[0].text.replace(",", "").replace("_", "")
    res[L]["pass"] += int(str(val) in txt)
    res[L]["n"] += 1

print(f"=== [{KVD}] needle retrieval ({TRIALS} trials/len) ===")
for L in LENGTHS:
    r = res[L]
    print(f"  len~{L:>6}: {r['pass']:>2}/{r['n']:>2}  ({100*r['pass']/r['n']:.0f}%)")
tot_p = sum(r['pass'] for r in res.values()); tot_n = sum(r['n'] for r in res.values())
print(f"[{KVD}] TOTAL {tot_p}/{tot_n} ({100*tot_p/tot_n:.0f}%)  gen {gen_s:.0f}s")
json.dump({"kvd": KVD, "res": res, "gen_s": gen_s, "trials": TRIALS},
          open(f"/tmp/needle_serving_{KVD}.json", "w"))
print(f"NEEDLE_SERVING DONE [{KVD}]")
