"""Long-context test: does the INT4-KIVI vs NVFP4-baseline gap grow with context?

Scores long in-distribution sequences (real source code, in-distribution for a
code model) in a SINGLE forward per scheme. KV quantization is injected by
patching scaled_dot_product_attention so position t attends to a quantized
prefix. Per-position KL(bf16||scheme) and top-1 agreement are binned by sequence
position, revealing whether quantization error compounds as context lengthens.

All cached K/V are quantized (the <16-token bf16 hot page is omitted — negligible
at long context, and a slightly conservative choice).

Usage:
    python -m scripts.quant_longctx [--ctx 8000] [--windows 3]
"""
from __future__ import annotations

import argparse
import glob
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/alex/poolside-hackathon-kv-quant")
from kv_quant import roundtrip

MODEL = "poolside/Laguna-XS.2"
N_ALPHAS = 32

SCHEMES = {
    "nvfp4-baseline": {"k": ("nvfp4", "headdim", "absmax"), "v": ("nvfp4", "headdim", "absmax")},
    "int4-kivi":      {"k": ("int4", "channel", "mse"),     "v": ("int4", "headdim", "mse")},
    "int3-kivi":      {"k": ("int3", "channel", "mse"),     "v": ("int3", "headdim", "mse")},
    "int3-naive":     {"k": ("int3", "headdim", "absmax"),  "v": ("int3", "headdim", "absmax")},
}
BASELINE = "nvfp4-baseline"
BINS = [(0, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192)]

_ORIG_SDPA = F.scaled_dot_product_attention
_SCHEME = None
_HITS = 0


def _q_per_head(x, cell):
    """Quantize [H, S, D] head-by-head (caps the MSE-search peak memory)."""
    return torch.stack([roundtrip(x[h:h + 1], *cell, n_alphas=N_ALPHAS)[0]
                        for h in range(x.shape[0])])


def _patched_sdpa(query, key, value, *a, **kw):
    global _HITS
    if _SCHEME is not None:
        _HITS += 1
        key = _q_per_head(key[0], _SCHEME["k"]).unsqueeze(0)
        value = _q_per_head(value[0], _SCHEME["v"]).unsqueeze(0)
    return _ORIG_SDPA(query, key, value, *a, **kw)


def token_pool(tok):
    files = sorted(glob.glob(
        "/home/alex/poolside-hackathon-kv-quant/.venv/**/transformers/**/modeling_*.py",
        recursive=True))
    texts, total = [], 0
    for f in files:
        try:
            t = open(f).read()
        except OSError:
            continue
        texts.append(t)
        total += len(t)
        if total > 600_000:
            break
    ids = tok("\n\n".join(texts), return_tensors="pt").input_ids[0]
    return ids


@torch.no_grad()
def logits_of(model, ids):
    return model(input_ids=ids.unsqueeze(0)).logits[0]      # [S, V], bf16


def kl_top1(ref, lg, ref_arg, chunk=512):
    """Per-position KL(bf16||scheme) and top-1 match. ref/lg bf16 [S,V]."""
    S = ref.shape[0]
    kls, t1 = [], []
    for i in range(0, S, chunk):
        r, q = ref[i:i + chunk].float(), lg[i:i + chunk].float()
        rlp, qlp = torch.log_softmax(r, -1), torch.log_softmax(q, -1)
        kls.append((rlp.exp() * (rlp - qlp)).sum(-1))
        t1.append((q.argmax(-1) == ref_arg[i:i + chunk]).float())
    return torch.cat(kls), torch.cat(t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=8000)
    ap.add_argument("--windows", type=int, default=3)
    args = ap.parse_args()
    global _SCHEME, _HITS

    F.scaled_dot_product_attention = _patched_sdpa
    torch.nn.functional.scaled_dot_product_attention = _patched_sdpa

    print(f"[load] {MODEL}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa")
    model.eval()
    device = next(model.parameters()).device

    pool = token_pool(tok)
    ctx = (args.ctx // 16) * 16
    nwin = min(args.windows, pool.shape[0] // ctx)
    print(f"[seq] {nwin} windows x {ctx} tokens (pool={pool.shape[0]})", flush=True)

    bins = [(lo, hi) for lo, hi in BINS if lo < ctx]
    acc = {n: {b: {"kl": [], "t1": []} for b in bins} for n in SCHEMES}

    for w in range(nwin):
        ids = pool[w * ctx:(w + 1) * ctx].to(device)
        _SCHEME = None
        ref = logits_of(model, ids)
        ref_arg = ref.argmax(-1)
        for name, sch in SCHEMES.items():
            _HITS = 0
            _SCHEME = sch
            lg = logits_of(model, ids)
            _SCHEME = None
            assert _HITS > 0, f"SDPA patch never fired for {name} — wrong attn path"
            kl, t1 = kl_top1(ref, lg, ref_arg)
            for lo, hi in bins:
                acc[name][(lo, hi)]["kl"].append(kl[lo:min(hi, ctx)].mean().item())
                acc[name][(lo, hi)]["t1"].append(t1[lo:min(hi, ctx)].mean().item())
            del lg
        print(f"[window {w}] done", flush=True)

    avg = lambda xs: sum(xs) / max(len(xs), 1)
    print("\n" + "=" * 74)
    print(f"PER-POSITION KL(bf16||scheme), avg over {nwin} windows")
    print(f"  {'position':<12}" + "".join(f"{n[:13]:>14}" for n in SCHEMES))
    print(f"  {'-'*12}" + "".join(f" {'-'*13}" for _ in SCHEMES))
    for lo, hi in bins:
        row = f"  {f'{lo}-{min(hi,ctx)}':<12}"
        for n in SCHEMES:
            row += f"{avg(acc[n][(lo,hi)]['kl']):>14.5f}"
        print(row)

    print(f"\nKIVI vs baseline: KL reduction by position  (does the win grow?)")
    print(f"  {'position':<12}{'int4-kivi':>14}{'int3-kivi':>14}")
    print(f"  {'-'*12}{' '+'-'*13}{' '+'-'*13}")
    for lo, hi in bins:
        b = avg(acc[BASELINE][(lo, hi)]["kl"])
        i4 = avg(acc["int4-kivi"][(lo, hi)]["kl"])
        i3 = avg(acc["int3-kivi"][(lo, hi)]["kl"])
        row = f"  {f'{lo}-{min(hi,ctx)}':<12}"
        row += f"{100*(b-i4)/max(b,1e-12):>13.0f}%"
        row += f"{100*(b-i3)/max(b,1e-12):>13.0f}%"
        print(row)

    print(f"\nTOP-1 agreement vs bf16 by position")
    print(f"  {'position':<12}" + "".join(f"{n[:13]:>14}" for n in SCHEMES))
    for lo, hi in bins:
        row = f"  {f'{lo}-{min(hi,ctx)}':<12}"
        for n in SCHEMES:
            row += f"{100*avg(acc[n][(lo,hi)]['t1']):>13.1f}%"
        print(row)


if __name__ == "__main__":
    main()
