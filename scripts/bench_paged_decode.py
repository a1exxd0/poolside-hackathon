"""Microbenchmark: fused paged INT4-KIVI decode vs the dense gather+attend path.

Same packed cache for both; times one decode step (single query token per
request) over a sweep of context lengths and batch sizes.  This is the decode-
speed future-work item: the fused path must beat (or at least not regress) the
dense whole-context dequant that materializes (B,H,max_seq,D) bf16 every step.

Run with the vLLM venv from /tmp (avoid vllm package shadowing):
  cd /tmp && /home/alex/poolside-hackathon-kv-quant/.venv-vllm/bin/python \
    .../scripts/bench_paged_decode.py
"""

from __future__ import annotations

import math
import time

import torch

from vllm.v1.attention.ops.triton_int4_kivi import (
    int4_kivi_gather_dequant,
    int4_kivi_paged_decode,
    int4_kivi_store,
)

DEV = "cuda"
HQ, HK, D = 48, 8, 128
PAGE = 16
FULL_DIM = D // 2 + D // 16
SM = 1.0 / math.sqrt(D)


def build_cache(B, L, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = (L + PAGE - 1) // PAGE
    num_blocks = B * nb + 4
    kv_cache = torch.zeros(
        (num_blocks, 2, PAGE, HK, FULL_DIM), dtype=torch.uint8, device=DEV
    )
    block_table = torch.zeros((B, nb), dtype=torch.int32, device=DEV)
    cursor = 1
    for b in range(B):
        phys = list(range(cursor, cursor + nb))
        cursor += nb
        for j, p in enumerate(phys):
            block_table[b, j] = p
        k = torch.randn(L, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(L, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
        slots = torch.tensor(
            [phys[t // PAGE] * PAGE + (t % PAGE) for t in range(L)],
            dtype=torch.int64, device=DEV,
        )
        int4_kivi_store(k, v, kv_cache, slots, D)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=DEV)
    return kv_cache, block_table, seq_lens


def dense_decode(q, kv_cache, block_table, seq_lens):
    """Reproduce the backend's dense path: gather-dequant whole cache + SDPA."""
    B = q.shape[0]
    max_seq = int(seq_lens.max().item())
    k_dense, v_dense = int4_kivi_gather_dequant(
        kv_cache, block_table, seq_lens, D, HK, max_seq
    )
    group = HQ // HK
    out = torch.empty(B, HQ, D, dtype=torch.bfloat16, device=DEV)
    for b in range(B):
        L = int(seq_lens[b].item())
        k = k_dense[b, :, :L, :].repeat_interleave(group, dim=0)  # [HQ,L,D]
        v = v_dense[b, :, :L, :].repeat_interleave(group, dim=0)
        qb = q[b].unsqueeze(1)                       # [HQ,1,D]
        scores = (qb.float() @ k.float().transpose(-1, -2)) * SM  # [HQ,1,L]
        p = torch.softmax(scores, dim=-1)
        out[b] = (p @ v.float()).squeeze(1).to(torch.bfloat16)
    return out


def timeit(fn, iters=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/step


if __name__ == "__main__":
    print(f"{'B':>3} {'ctx':>7} | {'dense ms':>9} {'fused ms':>9} {'speedup':>8}")
    for B, L in [(1, 4096), (1, 12000), (1, 32000),
                 (8, 4096), (8, 12000), (16, 8000), (32, 4096)]:
        kv_cache, bt, sl = build_cache(B, L)
        q = torch.randn(B, HQ, D, device=DEV, dtype=torch.bfloat16)
        # correctness sanity at this size
        ref = dense_decode(q, kv_cache, bt, sl)
        fus = int4_kivi_paged_decode(q, kv_cache, bt, sl, SM)
        d = (ref.float() - fus.float()).abs().max().item()
        dense_ms = timeit(lambda: dense_decode(q, kv_cache, bt, sl))
        fused_ms = timeit(lambda: int4_kivi_paged_decode(q, kv_cache, bt, sl, SM))
        print(f"{B:>3} {L:>7} | {dense_ms:9.3f} {fused_ms:9.3f} "
              f"{dense_ms/fused_ms:7.2f}x   max|Δ|={d:.1e}")
