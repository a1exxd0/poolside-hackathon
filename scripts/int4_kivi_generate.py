"""End-to-end INT4-KIVI generation vs BF16 DynamicCache on Laguna-XS.2.

For a few prompts (incl. a coding prompt) runs GREEDY generation twice:
  (a) baseline transformers DynamicCache (bf16)
  (b) Int4KiviCache  (completed 16-token pages stored INT4-KIVI, dequant on read)

Prints both completions side by side, matching leading tokens, overall top-1
agreement, and the INT4-KIVI cache memory vs the equivalent bf16 cache.

Usage:
    .venv/bin/python scripts/int4_kivi_generate.py [--max-new 200]
"""

from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "/home/alex/poolside-hackathon-kv-quant")
from int4_kivi.hf_cache import Int4KiviCache

MODEL = "poolside/Laguna-XS.2"

PROMPTS = [
    "In one short paragraph, explain what a hash table is and why lookups are fast.",
    "List the first 8 prime numbers and then briefly say what makes a number prime.",
    "Write a Python function `is_palindrome(s)` that returns True if the string is a "
    "palindrome ignoring case and non-alphanumeric characters, then explain how it works.",
]


@torch.no_grad()
def greedy(model, input_ids, cache, max_new, eos_ids, device):
    """Greedy decode with a given (already-empty) cache. Returns token id list."""
    toks: list[int] = []
    pos = torch.arange(input_ids.shape[1], device=device)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                cache_position=pos, position_ids=pos.unsqueeze(0))
    nxt = out.logits[0, -1].argmax().item()
    toks.append(nxt)
    abs_pos = input_ids.shape[1]
    for _ in range(max_new - 1):
        if nxt in eos_ids:
            break
        cp = torch.tensor([abs_pos], device=device)
        out = model(input_ids=torch.tensor([[nxt]], device=device),
                    past_key_values=cache, use_cache=True,
                    cache_position=cp, position_ids=cp.unsqueeze(0))
        nxt = out.logits[0, -1].argmax().item()
        toks.append(nxt)
        abs_pos += 1
    return toks


@torch.no_grad()
def teacher_forced_logits(model, input_ids, gold, cache, device):
    """Replay `gold` tokens through `cache`; return stacked logits [len(gold), V].

    Both caches see *identical* context (BF16's own gold tokens), so this
    isolates the cache's numerical fidelity from greedy-path divergence.
    """
    pos = torch.arange(input_ids.shape[1], device=device)
    out = model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                cache_position=pos, position_ids=pos.unsqueeze(0))
    logits = [out.logits[0, -1].float().cpu()]
    abs_pos = input_ids.shape[1]
    for t in gold[:-1]:
        cp = torch.tensor([abs_pos], device=device)
        out = model(input_ids=torch.tensor([[t]], device=device),
                    past_key_values=cache, use_cache=True,
                    cache_position=cp, position_ids=cp.unsqueeze(0))
        logits.append(out.logits[0, -1].float().cpu())
        abs_pos += 1
    return torch.stack(logits)


def fidelity(ref_logits, kivi_logits):
    top1 = (ref_logits.argmax(-1) == kivi_logits.argmax(-1)).float().mean().item()
    logp = F.log_softmax(ref_logits, -1)
    logq = F.log_softmax(kivi_logits, -1)
    kl = (logp.exp() * (logp - logq)).sum(-1).mean().item()
    return top1, kl


def eos_set(model):
    eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        return set(eos)
    return {eos} if eos is not None else set()


def leading_match(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def top1_agreement(a: list[int], b: list[int]) -> float:
    m = min(len(a), len(b))
    if m == 0:
        return 0.0
    return sum(1 for i in range(m) if a[i] == b[i]) / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=200)
    args = ap.parse_args()

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    eos_ids = eos_set(model)

    all_tf_top1 = []
    all_free_top1 = []
    for pi, prompt in enumerate(PROMPTS):
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False).to(device)

        bf16_cache = DynamicCache(config=model.config)
        bf16_toks = greedy(model, input_ids, bf16_cache, args.max_new, eos_ids, device)

        kivi_cache = Int4KiviCache(config=model.config)
        kivi_toks = greedy(model, input_ids, kivi_cache, args.max_new, eos_ids, device)

        # Teacher-forced fidelity: feed BF16's own tokens through both caches so
        # the context is identical and we measure the cache, not greedy drift.
        ref_lg = teacher_forced_logits(
            model, input_ids, bf16_toks, DynamicCache(config=model.config), device)
        kivi_lg = teacher_forced_logits(
            model, input_ids, bf16_toks, Int4KiviCache(config=model.config), device)
        tf_top1, tf_kl = fidelity(ref_lg, kivi_lg)
        all_tf_top1.append(tf_top1)

        bf16_text = tok.decode(bf16_toks, skip_special_tokens=True)
        kivi_text = tok.decode(kivi_toks, skip_special_tokens=True)
        lead = leading_match(bf16_toks, kivi_toks)
        t1 = top1_agreement(bf16_toks, kivi_toks)
        all_free_top1.append(t1)

        seq_len = input_ids.shape[1] + len(bf16_toks)
        comp = kivi_cache.compression_ratio_vs_bf16()
        kivi_mb = kivi_cache.nbytes() / 1e6
        bf16_mb = kivi_cache.bf16_nbytes() / 1e6

        print("\n" + "=" * 90)
        print(f"PROMPT {pi}: {prompt[:80]}{'...' if len(prompt) > 80 else ''}")
        print(f"  seq_len={seq_len}  gen_tokens(bf16)={len(bf16_toks)} "
              f"gen_tokens(kivi)={len(kivi_toks)}")
        print(f"  teacher-forced top-1 (identical context) : {100*tf_top1:.1f}%  "
              f"(mean KL {tf_kl:.5f})")
        print(f"  free-running leading matching tokens     : {lead}")
        print(f"  free-running top-1 agreement             : {100*t1:.1f}%")
        print(f"  INT4-KIVI cache memory  : {kivi_mb:.2f} MB  vs bf16 {bf16_mb:.2f} MB "
              f"({comp:.2f}x compression)")
        print("\n  --- BF16 (DynamicCache) ------------------------------------------")
        print("  " + bf16_text.replace("\n", "\n  "))
        print("\n  --- INT4-KIVI (Int4KiviCache) ------------------------------------")
        print("  " + kivi_text.replace("\n", "\n  "))

    print("\n" + "=" * 90)
    print(f"OVERALL mean teacher-forced top-1 (identical context) over "
          f"{len(PROMPTS)} prompts: {100*sum(all_tf_top1)/len(all_tf_top1):.1f}%")
    print(f"OVERALL mean free-running top-1 over {len(PROMPTS)} prompts: "
          f"{100*sum(all_free_top1)/len(all_free_top1):.1f}%")


if __name__ == "__main__":
    main()
