"""Downstream evidence: INT4/INT3 + per-channel-K (KIVI) vs vLLM's NVFP4 KV baseline.

Across several prompts on Laguna-XS.2, reports for each scheme:
  * K-RMSE / V-RMSE   — reconstruction error (cheap proxy; tracks KL)
  * top-1 agreement   — teacher-forced vs BF16 (identical context, no drift)
  * mean KL(bf16||scheme) in nats — output-distribution distortion

Protocol is production-faithful: each 16-token page is quantized once when it
fills (frozen thereafter), and the partial hot page stays BF16. Teacher forcing
replays BF16's own tokens so every scheme sees identical context.

Schemes:
  nvfp4-baseline  K,V = nvfp4 / headdim / absmax        (what vLLM ships, 4-bit)
  int4-kivi       K = int4 / per-channel / mse,  V = int4 / per-token / mse
  int3-kivi       K = int3 / per-channel / mse,  V = int3 / per-token / mse
  int3-naive      K,V = int3 / headdim / absmax         (3-bit done the vLLM way)

Usage:
    python -m scripts.quant_ab [--max-new 384] [--n-prompts 3]
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

sys.path.insert(0, "/home/alex/poolside-hackathon-kv-quant")
from kv_quant import BLOCK, PAGE, roundtrip, rmse_cell

MODEL = "poolside/Laguna-XS.2"

PROMPTS = [
    "Solve step by step. A train leaves city A at 60 km/h. Two hours later a second "
    "train leaves the same station on the same track at 90 km/h. How many hours after "
    "the second train departs will it catch up to the first train? Show your reasoning.",
    "Explain step by step how the quicksort algorithm works, including how partitioning "
    "works and its time complexity in the best, average, and worst cases. Give a small "
    "worked example.",
    "Write a Python function that merges two sorted linked lists into one sorted list, "
    "then explain step by step how it works and analyze its time and space complexity.",
]

# scheme -> per-K / per-V (format, layout, calib) cells + data bit-width.
SCHEMES = {
    "nvfp4-baseline": {"k": ("nvfp4", "headdim", "absmax"), "v": ("nvfp4", "headdim", "absmax"), "bits": 4},
    "int4-kivi":      {"k": ("int4", "channel", "mse"),     "v": ("int4", "headdim", "mse"),     "bits": 4},
    "int3-kivi":      {"k": ("int3", "channel", "mse"),     "v": ("int3", "headdim", "mse"),     "bits": 3},
    "int3-naive":     {"k": ("int3", "headdim", "absmax"),  "v": ("int3", "headdim", "absmax"),  "bits": 3},
}
BASELINE = "nvfp4-baseline"


def mem_ratio(bits: int) -> float:
    """vs BF16: data bits/8 + one 1-byte scale per 16-elem block."""
    return 2.0 / (bits / 8.0 + 1.0 / BLOCK)


class PageSim:
    """Freeze-at-fill quantization on a live DynamicCache: completed pages are
    quantized once and kept; the partial hot page stays BF16."""

    def __init__(self, scheme):
        self.scheme = scheme
        self.n_frozen = 0

    def update(self, cache):
        if self.scheme is None:
            return
        n_pages = cache.layers[0].keys.shape[2] // PAGE
        if n_pages <= self.n_frozen:
            return
        lo, hi = self.n_frozen * PAGE, n_pages * PAGE
        for layer in cache.layers:
            k, v = layer.keys[0], layer.values[0]
            qk = roundtrip(k[:, lo:hi], *self.scheme["k"])
            qv = roundtrip(v[:, lo:hi], *self.scheme["v"])
            layer.keys = torch.cat([k[:, :lo], qk, k[:, hi:]], dim=1).unsqueeze(0)
            layer.values = torch.cat([v[:, :lo], qv, v[:, hi:]], dim=1).unsqueeze(0)
        self.n_frozen = n_pages


def _eos_set(model):
    eos = getattr(model.config, "eos_token_id", None)
    if isinstance(eos, (list, tuple)):
        return set(eos)
    return {eos} if eos is not None else set()


def _prefill(model, input_ids, cache, device):
    pos = torch.arange(input_ids.shape[1], device=device)
    return model(input_ids=input_ids, past_key_values=cache, use_cache=True,
                 cache_position=pos, position_ids=pos.unsqueeze(0))


def _step(model, tok_id, cache, abs_pos, device):
    cp = torch.tensor([abs_pos], device=device)
    return model(input_ids=torch.tensor([[tok_id]], device=device), past_key_values=cache,
                 use_cache=True, cache_position=cp, position_ids=cp.unsqueeze(0))


def gen_bf16(model, input_ids, max_new, device, eos):
    """BF16 greedy; returns (gold_tokens, ref_logits [N,V] cpu, bf16 cache)."""
    cache = DynamicCache()
    logits, toks = [], []
    with torch.no_grad():
        out = _prefill(model, input_ids, cache, device)
        logits.append(out.logits[0, -1].float().cpu())
        toks.append(out.logits[0, -1].argmax().item())
        abs_pos = input_ids.shape[1]
        for _ in range(max_new - 1):
            out = _step(model, toks[-1], cache, abs_pos, device)
            logits.append(out.logits[0, -1].float().cpu())
            toks.append(out.logits[0, -1].argmax().item())
            abs_pos += 1
            if toks[-1] in eos:
                break
    return toks, torch.stack(logits), cache


def teacher_forced(model, input_ids, gold, scheme, device):
    """Replay gold through a frozen-page scheme cache; logits [len(gold), V] cpu."""
    cache = DynamicCache()
    sim = PageSim(scheme)
    logits = []
    with torch.no_grad():
        out = _prefill(model, input_ids, cache, device)
        sim.update(cache)
        logits.append(out.logits[0, -1].float().cpu())
        abs_pos = input_ids.shape[1]
        for t in gold[:-1]:
            out = _step(model, t, cache, abs_pos, device)
            sim.update(cache)
            logits.append(out.logits[0, -1].float().cpu())
            abs_pos += 1
    return torch.stack(logits)


def fidelity(ref, scheme_logits):
    top1 = (scheme_logits.argmax(-1) == ref.argmax(-1)).float().mean().item()
    logp = torch.log_softmax(ref, dim=-1)
    logq = torch.log_softmax(scheme_logits, dim=-1)
    kl = (logp.exp() * (logp - logq)).sum(-1).mean().item()
    return top1, kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--n-prompts", type=int, default=3)
    args = ap.parse_args()

    print(f"[load] {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    device = next(model.parameters()).device
    eos = _eos_set(model)

    agg = {n: {"top1": [], "kl": [], "krmse": [], "vrmse": []} for n in SCHEMES}
    for pi, prompt in enumerate(PROMPTS[:args.n_prompts]):
        input_ids = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True, return_tensors="pt", return_dict=False).to(device)
        t0 = time.time()
        gold, ref_logits, cache = gen_bf16(model, input_ids, args.max_new, device, eos)
        ctx = input_ids.shape[1] + len(gold)

        for layer in cache.layers:                       # RMSE on the BF16 cache
            S = layer.keys.shape[2]
            nf = (S // BLOCK) * BLOCK
            K, V = layer.keys[0, :, :nf], layer.values[0, :, :nf]
            for n, s in SCHEMES.items():
                agg[n]["krmse"].append(rmse_cell(K, *s["k"]))
                agg[n]["vrmse"].append(rmse_cell(V, *s["v"]))

        for n, s in SCHEMES.items():                     # teacher-forced KL
            top1, kl = fidelity(ref_logits, teacher_forced(model, input_ids, gold, s, device))
            agg[n]["top1"].append(top1)
            agg[n]["kl"].append(kl)
        print(f"[prompt {pi}] ctx={ctx} tokens, {time.time()-t0:.0f}s", flush=True)

    avg = lambda xs: sum(xs) / max(len(xs), 1)
    base_kl = avg(agg[BASELINE]["kl"])
    print("\n" + "=" * 78)
    print(f"AGGREGATE over {args.n_prompts} prompts  (production-faithful frozen-page protocol)")
    print(f"  {'scheme':<15} {'bits':>4} {'mem×':>5} {'K-RMSE':>8} {'V-RMSE':>8} "
          f"{'top-1':>7} {'KL':>8} {'KL vs base':>11}")
    print(f"  {'-'*15} {'-'*4} {'-'*5} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*11}")
    for n, s in SCHEMES.items():
        kl = avg(agg[n]["kl"])
        kld = "baseline" if n == BASELINE else f"{100*(base_kl-kl)/max(base_kl,1e-12):+.0f}%"
        print(f"  {n:<15} {s['bits']:>4} {mem_ratio(s['bits']):>4.2f}x "
              f"{avg(agg[n]['krmse']):>8.5f} {avg(agg[n]['vrmse']):>8.5f} "
              f"{100*avg(agg[n]['top1']):>6.1f}% {kl:>8.5f} {kld:>11}")


if __name__ == "__main__":
    main()
