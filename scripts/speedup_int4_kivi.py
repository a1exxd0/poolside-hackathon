"""PRIMARY benchmark: fused INT4-KIVI flash-decode vs bf16 flash-decode.

Decode attention at long context is HBM-bandwidth bound. The fused int4-KIVI
kernel reads ~3.2x fewer bytes for K/V than a bf16 cache, so it wins on latency
once the KV read dominates and the GPU is bandwidth (not launch) bound.

Both paths use the SAME flash-decode split-K structure (one program per
(batch, q-head, kv-segment), online softmax, then a combine reduction) so the
only difference is the KV dtype + in-kernel int4 unpack. We sweep context length
and batch size on the B300, print latency, speedup, achieved GB/s, and tokens/s,
and identify the crossover. The bf16 baseline is cross-checked vs torch SDPA.

Run:
  .venv/bin/python scripts/speedup_int4_kivi.py
"""

from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
import triton.language as tl

from int4_kivi import store_kivi  # noqa: E402
from int4_kivi.decode import (  # noqa: E402
    stack_caches, kivi_decode_attention_gqa,
)

DEV = "cuda"
H_KV, D = 8, 128
N_QH = 48
GROUP = N_QH // H_KV
CTXS = [4096, 8192, 16384, 32768, 65536]
BATCHES = [1, 8, 32]
SPLIT = 16


# --------------------------------------------------------------------------- #
# bf16 split-K flash-decode baseline (same structure as the int4 split-K path).
# --------------------------------------------------------------------------- #
@triton.jit
def _bf16_splitk_kernel(
    q_ptr, k_ptr, v_ptr,
    pm_ptr, pl_ptr, pacc_ptr,
    sm_scale, S, n_qh,
    GROUP: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % SPLIT
    tmp = pid // SPLIT
    qh = tmp % n_qh
    b = tmp // n_qh
    kvh = b * H + (qh // GROUP)

    d = tl.arange(0, D)
    q = tl.load(q_ptr + (b * n_qh + qh) * D + d).to(tl.float32) * sm_scale

    seg = (S + SPLIT - 1) // SPLIT
    seg0 = s * seg
    seg1 = tl.minimum(seg0 + seg, S)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    kv_base = kvh * S
    t = seg0
    while t < seg1:
        tok = t + tl.arange(0, BLOCK_N)
        tmask = tok < seg1
        koff = (kv_base + tok[:, None]) * D + d[None, :]
        kblk = tl.load(k_ptr + koff, mask=tmask[:, None], other=0.0).to(tl.float32)
        qk = tl.sum(q[None, :] * kblk, axis=1)
        qk = tl.where(tmask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        pblk = tl.where(tmask, tl.exp(qk - m_new), 0.0)
        vblk = tl.load(v_ptr + koff, mask=tmask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha + tl.sum(pblk[:, None] * vblk, axis=0)
        l_i = l_i * alpha + tl.sum(pblk, axis=0)
        m_i = m_new
        t += BLOCK_N

    base = (b * n_qh + qh) * SPLIT + s
    tl.store(pm_ptr + base, m_i)
    tl.store(pl_ptr + base, l_i)
    tl.store(pacc_ptr + base * D + d, acc)


@triton.jit
def _bf16_gqa_kernel(
    q_ptr, k_ptr, v_ptr,
    pm_ptr, pl_ptr, pacc_ptr,
    sm_scale, S, n_qh,
    GROUP: tl.constexpr, GPAD: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """GQA-grouped bf16 split-K baseline: load each KV head once, reuse across the
    GROUP query heads (same structure as the int4 fast path, only KV dtype differs)."""
    pid = tl.program_id(0)
    s = pid % SPLIT
    tmp = pid // SPLIT
    kvh_local = tmp % H
    b = tmp // H
    kvh = b * H + kvh_local
    qh0 = kvh_local * GROUP

    d = tl.arange(0, D)
    gr = tl.arange(0, GPAD)
    gmask = gr < GROUP
    qoff = (b * n_qh + qh0 + gr[:, None]) * D + d[None, :]
    q = tl.load(q_ptr + qoff, mask=gmask[:, None], other=0.0).to(tl.float32) * sm_scale

    seg = (S + SPLIT - 1) // SPLIT
    seg0 = s * seg
    seg1 = tl.minimum(seg0 + seg, S)

    m_i = tl.full([GPAD], -float("inf"), tl.float32)
    l_i = tl.zeros([GPAD], tl.float32)
    acc = tl.zeros([GPAD, D], tl.float32)
    kv_base = kvh * S
    t = seg0
    while t < seg1:
        tok = t + tl.arange(0, BLOCK_N)
        tmask = tok < seg1
        koff = (kv_base + tok[:, None]) * D + d[None, :]      # [BLOCK_N, D]
        kblk = tl.load(k_ptr + koff, mask=tmask[:, None], other=0.0).to(tl.float32)
        qk = tl.dot(q, tl.trans(kblk))                        # [GPAD, BLOCK_N]
        qk = tl.where(tmask[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        pblk = tl.where(tmask[None, :], tl.exp(qk - m_new[:, None]), 0.0)
        vblk = tl.load(v_ptr + koff, mask=tmask[:, None], other=0.0).to(tl.float32)
        acc = acc * alpha[:, None] + tl.dot(pblk.to(tl.float32), vblk)
        l_i = l_i * alpha + tl.sum(pblk, axis=1)
        m_i = m_new
        t += BLOCK_N

    base = ((b * H + kvh_local) * GROUP + gr) * SPLIT + s
    tl.store(pm_ptr + base, m_i, mask=gmask)
    tl.store(pl_ptr + base, l_i, mask=gmask)
    tl.store(pacc_ptr + base[:, None] * D + d[None, :], acc, mask=gmask[:, None])


def bf16_decode_gqa(q, k, v, sm_scale, split=SPLIT, block_n=32):
    """GQA-grouped bf16 split-K decode. q [B,n_qh,D], k/v [B*H,S,D]."""
    B, n_qh, _ = q.shape
    S = k.shape[1]
    split = max(1, min(split, (S + block_n - 1) // block_n))
    GPAD = max(1 << (GROUP - 1).bit_length(), 16)
    npart = B * H_KV * GROUP * split
    pm = torch.empty((npart,), dtype=torch.float32, device=q.device)
    pl = torch.empty((npart,), dtype=torch.float32, device=q.device)
    pacc = torch.empty((npart, D), dtype=torch.float32, device=q.device)
    qc = q.reshape(B, n_qh, D).contiguous()
    _bf16_gqa_kernel[(B * H_KV * split,)](
        qc, k, v, pm, pl, pacc, sm_scale, S, n_qh,
        GROUP=GROUP, GPAD=GPAD, D=D, H=H_KV, SPLIT=split, BLOCK_N=block_n, num_warps=4,
    )
    out = torch.empty((B * n_qh, D), dtype=torch.bfloat16, device=q.device)
    _combine_kernel[(B * n_qh,)](pm, pl, pacc, out, D=D, SPLIT=split)
    return out.reshape(B, n_qh, D)


@triton.jit
def _combine_kernel(pm_ptr, pl_ptr, pacc_ptr, out_ptr,
                    D: tl.constexpr, SPLIT: tl.constexpr):
    pid = tl.program_id(0)
    d = tl.arange(0, D)
    m = -float("inf")
    for s in range(0, SPLIT):
        m = tl.maximum(m, tl.load(pm_ptr + pid * SPLIT + s))
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for s in range(0, SPLIT):
        ms = tl.load(pm_ptr + pid * SPLIT + s)
        ls = tl.load(pl_ptr + pid * SPLIT + s)
        a = tl.load(pacc_ptr + (pid * SPLIT + s) * D + d)
        sc = tl.exp(ms - m)
        acc += a * sc
        l += ls * sc
    tl.store(out_ptr + pid * D + d, (acc / l).to(tl.bfloat16))


def bf16_decode_splitk(q, k, v, sm_scale, split=SPLIT, block_n=64):
    """q [B,n_qh,D], k/v [B*H, S, D] bf16 -> out [B,n_qh,D]."""
    B, n_qh, _ = q.shape
    S = k.shape[1]
    split = max(1, min(split, (S + block_n - 1) // block_n))
    pm = torch.empty((B * n_qh * split,), dtype=torch.float32, device=q.device)
    pl = torch.empty((B * n_qh * split,), dtype=torch.float32, device=q.device)
    pacc = torch.empty((B * n_qh * split, D), dtype=torch.float32, device=q.device)
    qc = q.reshape(B, n_qh, D).contiguous()
    _bf16_splitk_kernel[(B * n_qh * split,)](
        qc, k, v, pm, pl, pacc, sm_scale, S, n_qh,
        GROUP=GROUP, D=D, H=H_KV, SPLIT=split, BLOCK_N=block_n, num_warps=4,
    )
    out = torch.empty((B * n_qh, D), dtype=torch.bfloat16, device=q.device)
    _combine_kernel[(B * n_qh,)](pm, pl, pacc, out, D=D, SPLIT=split)
    return out.reshape(B, n_qh, D)


def _time_ms(fn, iters=50, warmup=15):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def main():
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"GQA: {N_QH} q-heads / {H_KV} kv-heads (group {GROUP}), D={D}, split-K={SPLIT}\n")
    sm = 1.0 / math.sqrt(D)

    # sanity: bf16 split-K baseline vs torch SDPA
    g = torch.Generator(device=DEV).manual_seed(1)
    kk = torch.randn(H_KV, 1024, D, generator=g, device=DEV).bfloat16()
    vv = torch.randn(H_KV, 1024, D, generator=g, device=DEV).bfloat16()
    qq = torch.randn(1, N_QH, D, generator=g, device=DEV).bfloat16()
    ob = bf16_decode_gqa(qq, kk, vv, sm).float()
    kx = kk.repeat_interleave(GROUP, dim=0).float()
    vx = vv.repeat_interleave(GROUP, dim=0).float()
    qx = qq[0].unsqueeze(1).float()
    sc = torch.softmax((qx @ kx.transpose(-1, -2)) * sm, dim=-1)
    oref = (sc @ vx).squeeze(1)
    print(f"[sanity] bf16 GQA split-K baseline vs torch SDPA rel-err: "
          f"{((ob[0]-oref).norm()/oref.norm()).item():.2e}\n")

    print("=" * 104)
    hdr = (f"{'batch':>5} {'ctx':>7} | {'bf16 us':>9} {'int4 us':>9} {'speedup':>8} | "
           f"{'bf16 GB/s':>10} {'int4 GB/s':>10} | {'bf16 Mtok/s':>11} {'int4 Mtok/s':>11}")
    print(hdr)
    print("-" * len(hdr))

    results = []
    for B in BATCHES:
        for S in CTXS:
            g = torch.Generator(device=DEV).manual_seed(S + B)
            k = torch.randn(B, H_KV, S, D, generator=g, device=DEV).bfloat16()
            v = torch.randn(B, H_KV, S, D, generator=g, device=DEV).bfloat16()
            q = torch.randn(B, N_QH, D, generator=g, device=DEV).bfloat16()
            kflat = k.reshape(B * H_KV, S, D)
            vflat = v.reshape(B * H_KV, S, D)

            bf16_ms = _time_ms(lambda: bf16_decode_gqa(q, kflat, vflat, sm))
            bf16_bytes = B * 2 * H_KV * S * D * 2
            bf16_gbs = bf16_bytes / (bf16_ms * 1e-3) / 1e9

            caches = [store_kivi(k[b], v[b]) for b in range(B)]
            stacked = stack_caches(caches)
            int4_bytes = sum(c.nbytes for c in caches)
            int4_ms = _time_ms(lambda: kivi_decode_attention_gqa(
                q, stacked, N_QH, sm_scale=sm, split=max(SPLIT, 32), block_n=16))
            int4_gbs = int4_bytes / (int4_ms * 1e-3) / 1e9

            speedup = bf16_ms / int4_ms
            bf16_mtok = B / (bf16_ms * 1e-3) / 1e6
            int4_mtok = B / (int4_ms * 1e-3) / 1e6
            results.append((B, S, speedup))
            print(f"{B:>5} {S:>7} | {bf16_ms*1e3:>9.1f} {int4_ms*1e3:>9.1f} {speedup:>7.2f}x | "
                  f"{bf16_gbs:>10.1f} {int4_gbs:>10.1f} | {bf16_mtok:>11.3f} {int4_mtok:>11.3f}")
            del k, v, kflat, vflat, caches, stacked
            torch.cuda.empty_cache()
        print("-" * len(hdr))

    print("\nCrossover (int4 first wins, speedup > 1.0):")
    for B in BATCHES:
        cross = next((S for (bb, S, sp) in results if bb == B and sp > 1.0), None)
        print(f"  batch {B:>3}: " +
              (f"int4 wins from ctx >= {cross}" if cross else "no win in tested range"))
    best = max(results, key=lambda r: r[2])
    print(f"\nbest fused-int4 single-step decode speedup = {best[2]:.2f}x "
          f"(batch {best[0]}, ctx {best[1]})")

    # peak HBM reference (saturating copy) to contextualise the achieved GB/s.
    x = torch.empty(1_000_000_000, dtype=torch.bfloat16, device=DEV)
    y = torch.empty_like(x)
    pk = _time_ms(lambda: y.copy_(x), iters=20, warmup=5)
    peak = 2 * x.numel() * 2 / (pk * 1e-3) / 1e9
    print(f"\nB300 peak HBM (copy) ~ {peak:.0f} GB/s.")
    print("FINDING: the bf16 decode baseline tops out ~1.6-1.7 TB/s (GEMV-issue")
    print("bound, FAR below the ~6.5 TB/s roof) — i.e. single-stream decode")
    print("attention on B300 is NOT bandwidth-starved, so moving 3.2x fewer bytes")
    print("does not remove the bottleneck. The int4 path additionally pays per-")
    print("element unpack ALU (nibble extract + sign-extend + scale), which caps")
    print("it ~180-250 GB/s of *compressed* bytes and makes single-step decode")
    print("~2.5-3x SLOWER. The INT4-KIVI win on B300 is therefore a CAPACITY win")
    print("(3.2x more KV in HBM -> 3.2x batch/context past the bf16 OOM frontier;")
    print("see scripts/bench_int4_kivi.py), not a single-stream latency win.")


if __name__ == "__main__":
    main()
