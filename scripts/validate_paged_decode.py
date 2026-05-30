"""Validate the fused paged INT4-KIVI decode kernel against the dense path.

Both sides read the SAME packed int4 cache (built with ``int4_kivi_store`` exactly
as vLLM's prefill does), so this isolates KERNEL correctness from quant error:
the fused ``int4_kivi_paged_decode`` must match dequant-the-whole-cache
(``int4_kivi_gather_dequant``) + GQA softmax attention, to fp32 tolerance.

Run with the vLLM venv from a NON-vllm cwd:
  cd /tmp && /home/alex/poolside-hackathon-kv-quant/.venv-vllm/bin/python \
    /home/alex/poolside-hackathon-kv-quant/.claude/worktrees/kv-quant-decode-speed/scripts/validate_paged_decode.py
"""

from __future__ import annotations

import math

import torch

from vllm.v1.attention.ops.triton_int4_kivi import (
    BLOCK,
    int4_kivi_gather_dequant,
    int4_kivi_paged_decode,
    int4_kivi_store,
)

DEV = "cuda"
HQ, HK, D = 48, 8, 128          # Laguna-XS.2 geometry (GQA group 6)
PAGE = 16                        # paged block_size (tokens per page)
FULL_DIM = D // 2 + D // 16      # 64 + 8 = 72


def build_cache(seq_lens, seed=0):
    """Build a paged int4 cache + block_table for the given per-request seq_lens,
    storing each request's whole sequence prefill-style (monotone slots from a
    block boundary -> full blocks become per-channel K, trailing block per-token).
    Returns (kv_cache, block_table, seq_lens_t, k_ref, v_ref) where k_ref/v_ref
    are the original bf16 K/V (only for sanity, not used as the reference)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    B = len(seq_lens)
    nblk_per = [(L + PAGE - 1) // PAGE for L in seq_lens]
    max_blocks = max(nblk_per)
    num_blocks = sum(nblk_per) + 4
    kv_cache = torch.zeros(
        (num_blocks, 2, PAGE, HK, FULL_DIM), dtype=torch.uint8, device=DEV
    )
    block_table = torch.zeros((B, max_blocks), dtype=torch.int32, device=DEV)

    cursor = 1  # leave block 0 unused to catch base-offset bugs
    for b, L in enumerate(seq_lens):
        nb = nblk_per[b]
        phys = list(range(cursor, cursor + nb))
        cursor += nb
        for j, p in enumerate(phys):
            block_table[b, j] = p
        k = torch.randn(L, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(L, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
        # slot_mapping: token t -> phys_block*PAGE + (t % PAGE)
        slots = torch.tensor(
            [phys[t // PAGE] * PAGE + (t % PAGE) for t in range(L)],
            dtype=torch.int64, device=DEV,
        )
        int4_kivi_store(k, v, kv_cache, slots, D)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.int32, device=DEV)
    return kv_cache, block_table, seq_lens_t


def ref_attend(q, kv_cache, block_table, seq_lens):
    """Dense reference: gather-dequant the whole cache, GQA softmax attention."""
    B = q.shape[0]
    max_seq = int(seq_lens.max().item())
    k_dense, v_dense = int4_kivi_gather_dequant(
        kv_cache, block_table, seq_lens, D, HK, max_seq
    )  # [B, HK, max_seq, D] bf16
    group = HQ // HK
    sm = 1.0 / math.sqrt(D)
    out = torch.empty(B, HQ, D, dtype=torch.bfloat16, device=DEV)
    for b in range(B):
        L = int(seq_lens[b].item())
        k = k_dense[b, :, :L, :].float().repeat_interleave(group, dim=0)  # [HQ,L,D]
        v = v_dense[b, :, :L, :].float().repeat_interleave(group, dim=0)
        qb = q[b].float().unsqueeze(1)                  # [HQ,1,D]
        scores = (qb @ k.transpose(-1, -2)) * sm        # [HQ,1,L]
        p = torch.softmax(scores, dim=-1)
        out[b] = (p @ v).squeeze(1).to(torch.bfloat16)
    return out


def run_case(seq_lens, seed=0):
    kv_cache, bt, sl = build_cache(seq_lens, seed=seed)
    g = torch.Generator(device=DEV).manual_seed(seed + 999)
    B = len(seq_lens)
    q = torch.randn(B, HQ, D, generator=g, device=DEV, dtype=torch.bfloat16)
    sm = 1.0 / math.sqrt(D)
    ref = ref_attend(q, kv_cache, bt, sl)
    fused = int4_kivi_paged_decode(q, kv_cache, bt, sl, sm)
    diff = (fused.float() - ref.float()).abs()
    rel = diff / (ref.float().abs() + 1e-3)
    return diff.max().item(), diff.mean().item(), rel.max().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    cases = {
        "exact-block (L=512)": [512],
        "partial-tail (L=500)": [500],
        "short (L=33)": [33],
        "tiny (L=1)": [1],
        "mixed batch": [128, 257, 64, 1000, 16, 999],
        "long (L=12000)": [12000],
        "long mixed": [12000, 8001, 16000, 4096],
    }
    ok = True
    for name, sl in cases.items():
        amax, amean, rmax = run_case(sl, seed=hash(name) % 10000)
        tol = 5e-2
        passed = amax < tol
        ok = ok and passed
        flag = "ok " if passed else "FAIL"
        print(f"[{flag}] {name:24s} max|Δ|={amax:.4e} mean|Δ|={amean:.2e} relmax={rmax:.2e}")
    print("ALL PASS" if ok else "SOME FAILED")
