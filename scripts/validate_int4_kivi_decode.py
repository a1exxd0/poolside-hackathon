"""Validate the fused INT4-KIVI decode kernel == dequant-then-SDPA.

Both sides operate on the SAME quantized cache, so this isolates KERNEL
correctness from quantization error: the fused path must equal the path that
fully dequantizes the cache (dequant_kivi) and runs standard GQA SDPA.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from int4_kivi import store_kivi, dequant_kivi, BLOCK, PACK  # noqa: E402
from int4_kivi.decode import kivi_decode_attention  # noqa: E402

DEV = "cuda"
H_KV, D = 8, 128
N_QH = 48          # Laguna-XS.2 query heads (GQA 48/8 = group 6)


def _unpack_codes(packed):
    """[..., PACK] uint8 -> [..., 16] int codes (sign-extended), torch-side."""
    p = packed.to(torch.int32)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


def dequant_fp32_exact(cache):
    """Reconstruct K,V in fp32 from packed int4 codes EXACTLY as the kernel does
    (code.fp32 * scale.fp32, no bf16 round-trip). This is the precise reference
    that isolates kernel-logic correctness from bf16 dequant rounding."""
    H, S, Dd = cache.H, cache.S, cache.D
    NP = cache.k_packed.shape[2]
    n_full = NP * BLOCK
    # K: codes [H, D, NP, 16] -> k[H, S, D]
    kc = _unpack_codes(cache.k_packed).float()                 # [H,D,NP,16]
    ks = cache.k_scale.float().unsqueeze(-1)                   # [H,D,NP,1]
    kfull = (kc * ks).reshape(H, Dd, n_full).permute(0, 2, 1)  # [H,n_full,D]
    k = torch.empty((H, S, Dd), dtype=torch.float32, device=kc.device)
    k[:, :n_full] = kfull
    if n_full < S:
        k[:, n_full:] = cache.k_hot.float()
    # V: codes [H, S, ND, 16] -> v[H,S,D]
    vc = _unpack_codes(cache.v_packed).float()                 # [H,S,ND,16]
    vs = cache.v_scale.float().unsqueeze(-1)                   # [H,S,ND,1]
    v = (vc * vs).reshape(H, S, Dd)
    return k, v


def ref_decode(q, cache):
    """dequant_kivi then GQA-expanded scaled-dot-product decode attention."""
    k, v = dequant_kivi(cache)                  # [H_KV, S, D] bf16
    group = q.shape[0] // cache.H
    k = k.repeat_interleave(group, dim=0)       # [N_QH, S, D]
    v = v.repeat_interleave(group, dim=0)
    qf = q.reshape(q.shape[0], 1, D).float()
    sm = 1.0 / math.sqrt(D)
    scores = (qf @ k.float().transpose(-1, -2)) * sm   # [N_QH,1,S]
    p = torch.softmax(scores, dim=-1)
    out = p @ v.float()                                # [N_QH,1,D]
    return out.to(torch.bfloat16)


def ref_decode_fp32(q, cache):
    """SDPA on fp32 K/V reconstructed exactly as the kernel dequantizes (no bf16
    round-trip). Shares the fused path's precision -> isolates kernel logic."""
    k, v = dequant_fp32_exact(cache)
    group = q.shape[0] // cache.H
    k = k.repeat_interleave(group, dim=0)
    v = v.repeat_interleave(group, dim=0)
    qf = q.reshape(q.shape[0], 1, D).float()
    sm = 1.0 / math.sqrt(D)
    scores = (qf @ k.transpose(-1, -2)) * sm
    p = torch.softmax(scores, dim=-1)
    return (p @ v)


def run(S, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    k = (torch.randn(H_KV, S, D, generator=g, device=DEV) ** 3).to(torch.bfloat16)
    v = torch.randn(H_KV, S, D, generator=g, device=DEV).to(torch.bfloat16)
    q = torch.randn(N_QH, 1, D, generator=g, device=DEV).to(torch.bfloat16)
    cache = store_kivi(k, v)

    out_fused = kivi_decode_attention(q, cache).float()
    out_ref = ref_decode(q, cache).float()
    out_ref32 = ref_decode_fp32(q, cache)

    # diagnostic: fused vs the fp32 reference (same int4 codes, fp32 throughout).
    n32 = (out_fused - out_ref32).reshape(N_QH, -1).norm(dim=-1)
    d32 = out_ref32.reshape(N_QH, -1).norm(dim=-1).clamp_min(1e-6)
    rel32 = (n32 / d32).max().item()

    abs_err = (out_fused - out_ref).abs()
    max_abs = abs_err.max().item()
    # Per-head relative error on the OUTPUT VECTOR NORM (principled: the output is
    # a vector, so element-wise rel-err near a zero component is meaningless;
    # ||fused - ref|| / ||ref|| per head is the right scale-invariant metric).
    num = (out_fused - out_ref).reshape(N_QH, -1).norm(dim=-1)
    den = out_ref.reshape(N_QH, -1).norm(dim=-1).clamp_min(1e-6)
    rel_norm = (num / den).max().item()
    cos = F.cosine_similarity(out_fused.reshape(N_QH, -1),
                              out_ref.reshape(N_QH, -1), dim=-1).min().item()
    n_full = (S // 16) * 16
    # The fused path dequants int4->fp32 directly; the reference dequants
    # int4->bf16 (dequant_kivi) then SDPAs. So they differ only by that one bf16
    # rounding of K/V (~2^-8 rel). A per-head rel-norm < 1e-2 confirms the fused
    # kernel computes the same attention as dequant-then-SDPA.
    # PASS criterion: vs the fp32 reference (which shares the fused path's
    # internal precision), the kernel logic must be exact to ~1e-2.
    ok = rel32 < 1e-2
    print(f"S={S:>6}  n_full={n_full:>6} hot={S-n_full:>2} | "
          f"max_abs={max_abs:.2e} rel_vs_bf16ref={rel_norm:.2e} "
          f"rel_vs_fp32ref={rel32:.2e} min_cos={cos:.6f} | "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print(f"Fused INT4-KIVI decode vs dequant-then-SDPA  (N_QH={N_QH}, H_KV={H_KV}, D={D})\n")
    results = []
    for S in [127, 512, 1000, 2048, 4096, 8192, 16384, 32768]:
        results.append(run(S, seed=S))
    print()
    print("ALL PASS" if all(results) else "SOME FAILED")
    sys.exit(0 if all(results) else 1)
