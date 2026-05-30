"""Sweep fused INT4-KIVI decode launch params to pick the best B300 config.

Times only the fused read (int4_kivi_paged_decode) across a small grid of
(BLOCK_N, num_warps, num_stages, split-waves) for a few (B, ctx) shapes and
prints the best per shape.  Mutates the module-level tuning globals between
runs (the launcher reads them per call).  Run from /tmp with the vLLM venv:

  cd /tmp && CUDA_HOME=/usr/local/cuda-12.8 .venv-vllm/bin/python \
    .../scripts/sweep_decode.py
"""

from __future__ import annotations

import math
import time

import torch

import vllm.v1.attention.ops.triton_int4_kivi as K
from vllm.v1.attention.ops.triton_int4_kivi import (
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


def timeit(fn, iters=40, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


SHAPES = [(1, 4096), (1, 12000), (1, 32000), (8, 12000), (16, 8000), (32, 4096)]
BLOCK_NS = [64, 128]
WARPS = [2, 4]
STAGES = [2, 3]
WAVES = [1, 2, 4]

if __name__ == "__main__":
    for B, L in SHAPES:
        kv_cache, bt, sl = build_cache(B, L)
        q = torch.randn(B, HQ, D, device=DEV, dtype=torch.bfloat16)
        best = (1e9, None)
        results = []
        for bn in BLOCK_NS:
            for w in WARPS:
                for st in STAGES:
                    for wv in WAVES:
                        K._DECODE_BLOCK_N = bn
                        K._DECODE_NUM_WARPS = w
                        K._DECODE_NUM_STAGES = st
                        K._DECODE_WAVES = wv
                        try:
                            ms = timeit(
                                lambda: int4_kivi_paged_decode(
                                    q, kv_cache, bt, sl, SM
                                )
                            )
                        except Exception as e:  # noqa: BLE001
                            ms = float("nan")
                        cfg = (bn, w, st, wv)
                        results.append((ms, cfg))
                        if ms < best[0]:
                            best = (ms, cfg)
        results.sort()
        print(f"\n=== B={B} ctx={L} ===  best {best[0]:.3f}ms  cfg(BLOCK_N,warps,stages,waves)={best[1]}")
        for ms, cfg in results[:5]:
            print(f"    {ms:7.3f}ms  {cfg}")
