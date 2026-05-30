"""How much does the INT4-KIVI quantization pipeline cost vs *only* flash decode?

Baseline ("only flash decode"): a bf16 KV cache attended with FlashAttention
(flash_attn_varlen_func) — exactly what vLLM's KVD=auto path runs, no quant.
Quantized path: our fused INT4-KIVI decode (int4_kivi_paged_decode) reading the
packed 4-bit cache, dequant-in-kernel.

We isolate the per-decode-step cost on identical shapes:
  * read/attend: fused int4 decode   vs   bf16 flash decode
  * store: per-token int4 quant (int4_kivi_store of 1 new token) — the other
    half of the quant pipeline that bf16 doesn't pay.

Run from /tmp with the vLLM venv:
  cd /tmp && CUDA_HOME=/usr/local/cuda-12.8 .venv-vllm/bin/python \
    .../scripts/bench_quant_vs_flash.py
"""

from __future__ import annotations

import math
import time

import torch

from vllm.v1.attention.backends.fa_utils import flash_attn_varlen_func
from vllm.v1.attention.ops.triton_int4_kivi import (
    int4_kivi_paged_decode,
    int4_kivi_store,
)

DEV = "cuda"
HQ, HK, D = 48, 8, 128
PAGE = 16
FULL_DIM = D // 2 + D // 16
SM = 1.0 / math.sqrt(D)


def build(B, L, seed=0):
    """Return bf16 dense K/V (varlen-packed) + the equivalent packed int4 cache."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = (L + PAGE - 1) // PAGE
    num_blocks = B * nb + 4
    kv_cache = torch.zeros(
        (num_blocks, 2, PAGE, HK, FULL_DIM), dtype=torch.uint8, device=DEV
    )
    block_table = torch.zeros((B, nb), dtype=torch.int32, device=DEV)
    k_pack = torch.empty(B * L, HK, D, dtype=torch.bfloat16, device=DEV)
    v_pack = torch.empty(B * L, HK, D, dtype=torch.bfloat16, device=DEV)
    cursor = 1
    for b in range(B):
        phys = list(range(cursor, cursor + nb))
        cursor += nb
        for j, p in enumerate(phys):
            block_table[b, j] = p
        k = torch.randn(L, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(L, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
        k_pack[b * L : (b + 1) * L] = k
        v_pack[b * L : (b + 1) * L] = v
        slots = torch.tensor(
            [phys[t // PAGE] * PAGE + (t % PAGE) for t in range(L)],
            dtype=torch.int64, device=DEV,
        )
        int4_kivi_store(k, v, kv_cache, slots, D)
    seq_lens = torch.full((B,), L, dtype=torch.int32, device=DEV)
    cu_q = torch.arange(B + 1, dtype=torch.int32, device=DEV)
    cu_k = torch.arange(0, B * L + 1, L, dtype=torch.int32, device=DEV)
    return kv_cache, block_table, seq_lens, k_pack, v_pack, cu_q, cu_k


def bf16_flash(q, k_pack, v_pack, cu_q, cu_k, L):
    return flash_attn_varlen_func(
        q=q, k=k_pack, v=v_pack,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=1, max_seqlen_k=L,
        softmax_scale=SM, causal=True, fa_version=4,
    )


def timeit(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3  # ms/step


def store_one(kv_cache, B):
    """Cost of quantizing+storing one new decode token per request."""
    g = torch.Generator(device=DEV).manual_seed(7)
    k = torch.randn(B, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
    # one new token per request, each token-0 of its own block -> partial block,
    # per-token K path (exactly how a decode step grows the trailing block).
    slots = (torch.arange(B, dtype=torch.int64, device=DEV)) * PAGE
    return lambda: int4_kivi_store(k, v, kv_cache, slots, D)


if __name__ == "__main__":
    print("Per decode step, B300.  read = attention over the cached context;")
    print("store = quantize+write the 1 new token (int4 only).  'pipeline' = read+store.\n")
    print(f"{'B':>3} {'ctx':>7} | {'bf16 flash':>10} {'int4 read':>10} {'int4 store':>10} "
          f"{'int4 pipe':>10} | {'read x':>7} {'pipe x':>7}")
    for B, L in [(1, 4096), (1, 12000), (1, 32000),
                 (8, 4096), (8, 12000), (16, 8000), (32, 4096)]:
        kv_cache, bt, sl, kp, vp, cu_q, cu_k = build(B, L)
        q = torch.randn(B, HQ, D, device=DEV, dtype=torch.bfloat16)
        qf = q.reshape(B, HQ, D)
        bf16_ms = timeit(lambda: bf16_flash(qf, kp, vp, cu_q, cu_k, L))
        int4_read_ms = timeit(lambda: int4_kivi_paged_decode(q, kv_cache, bt, sl, SM))
        int4_store_ms = timeit(store_one(kv_cache, B))
        pipe = int4_read_ms + int4_store_ms
        print(f"{B:>3} {L:>7} | {bf16_ms:10.3f} {int4_read_ms:10.3f} {int4_store_ms:10.3f} "
              f"{pipe:10.3f} | {int4_read_ms/bf16_ms:6.2f}x {pipe/bf16_ms:6.2f}x")
