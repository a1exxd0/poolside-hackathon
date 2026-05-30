"""Fused INT4-KIVI flash-decode attention.

Single-query (decode step) attention that reads the PACKED INT4-KIVI cache
directly and dequantizes K and V in-kernel (registers/SRAM), block by block,
online-softmax / flash style — WITHOUT ever materializing the full bf16 K/V.

This is the bandwidth payoff: at long context, decode attention is HBM-bound and
the int4 cache moves ~3.2-3.5x fewer bytes for the K/V reads than bf16.

Layout (see int4_kivi/triton_kernels.py, validated 16/16):
  K 'channel':  k_packed[H, D, NP, PACK] uint8, k_scale[H, D, NP] fp16,
                k_hot[H, S-n_full, D] bf16 (the trailing <16-token page).
                token t<n_full -> page p=t//16, slot j=t%16; channel d code j is
                byte j//2 of k_packed[h,d,p,:] (low nibble j even, high j odd);
                scale = k_scale[h,d,p]  (per head/channel/page).
  V 'headdim':  v_packed[H, S, ND, PACK] uint8, v_scale[H, S, ND] fp16.
                token t, dblock db, in-block channel c (global d=db*16+c): code c
                is byte c//2 of v_packed[h,t,db,:]; scale = v_scale[h,t,db].

GQA: n_q_heads query heads share n_kv_heads KV heads, group = n_q // n_kv.

Public:
  kivi_decode_attention(q, cache, sm_scale=None) -> out  [n_q_heads, 1, D]
"""

from __future__ import annotations

import math
import os
import sys

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
import triton.language as tl

from .cache import KIVICache
from .triton_kernels import BLOCK, PACK

_BLK = tl.constexpr(BLOCK)   # 16; page size / V dblock channel count
_PACK = tl.constexpr(PACK)   # 8


# --------------------------------------------------------------------------- #
# Fused flash-decode kernel.  One program = one (kv_head, query head within
# its GQA group).  It streams the cached prefix in chunks of BLOCK_N tokens,
# dequantizing K and V from the packed int4 cache on the fly, maintaining the
# online-softmax running max / denom / accumulator over head_dim D.
# --------------------------------------------------------------------------- #
@triton.jit
def _kivi_decode_kernel(
    q_ptr,            # bf16 [n_qh, D]
    kpacked_ptr,      # uint8 [H, D, NP, PACK]
    kscale_ptr,       # fp16  [H, D, NP]
    khot_ptr,         # bf16  [H, HOT, D]   (HOT = S - n_full, may be 0)
    vpacked_ptr,      # uint8 [H, S, ND, PACK]
    vscale_ptr,       # fp16  [H, S, ND]
    out_ptr,          # bf16  [n_qh, D]
    sm_scale,
    n_full,           # number of int4-quantised K tokens (= NP*BLOCK)
    HOT,              # hot-page token count (S - n_full)
    S,                # total prefix length
    GROUP: tl.constexpr,      # query heads per kv head
    D: tl.constexpr,          # head_dim (128)
    NP: tl.constexpr,         # K pages
    ND: tl.constexpr,         # V dblocks (= D//16)
    BLOCK_N: tl.constexpr,    # tokens per streamed chunk
):
    pid = tl.program_id(0)
    qh = pid                       # query-head index (0..n_qh-1)
    kvh = qh // GROUP              # kv-head this query head reads

    d = tl.arange(0, D)           # [D] head-dim lanes
    # load this query head's vector, scale by softmax scale.
    q = tl.load(q_ptr + qh * D + d).to(tl.float32) * sm_scale   # [D]

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    # --- stream the int4-quantised prefix [0, n_full) in BLOCK_N chunks ------
    n_chunks = (n_full + BLOCK_N - 1) // BLOCK_N
    for c in range(0, n_chunks):
        t0 = c * BLOCK_N
        tok = t0 + tl.arange(0, BLOCK_N)            # [BLOCK_N] token ids
        tmask = tok < n_full

        # ---- dequant K for these tokens: need k[tok, d] -------------------
        # page p = tok//BLOCK, slot j = tok%BLOCK; byte j//2, nibble parity j&1.
        p = tok // _BLK                             # [BLOCK_N]
        j = tok % _BLK                              # [BLOCK_N]
        byte = j // 2                               # [BLOCK_N]
        hi = (j % 2) == 1                           # [BLOCK_N] bool
        # k_packed[kvh, d, p, byte] : offset (kvh*D + d)*NP*PACK + p*PACK + byte
        koff = (kvh * D + d[:, None]) * (NP * _PACK) + p[None, :] * _PACK + byte[None, :]  # [D, BLOCK_N]
        kb = tl.load(kpacked_ptr + koff, mask=tmask[None, :], other=0).to(tl.int32)
        knib = tl.where(hi[None, :], (kb >> 4) & 0xF, kb & 0xF)
        knib = tl.where(knib >= 8, knib - 16, knib)           # sign-extend [D, BLOCK_N]
        # scale k_scale[kvh, d, p] : (kvh*D + d)*NP + p
        ksoff = (kvh * D + d[:, None]) * NP + p[None, :]       # [D, BLOCK_N]
        ksc = tl.load(kscale_ptr + ksoff, mask=tmask[None, :], other=0.0).to(tl.float32)
        kdeq = knib.to(tl.float32) * ksc                      # [D, BLOCK_N]

        # ---- scores: q . k  over D --------------------------------------
        qk = tl.sum(q[:, None] * kdeq, axis=0)                # [BLOCK_N]
        qk = tl.where(tmask, qk, -float("inf"))

        # ---- online softmax update --------------------------------------
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        pblk = tl.exp(qk - m_new)                             # [BLOCK_N]
        pblk = tl.where(tmask, pblk, 0.0)

        # ---- dequant V for these tokens: need v[tok, d] -----------------
        # v_packed[kvh, tok, db, byte_c] ; global d = db*16 + c, byte_c=c//2.
        # We reconstruct V as [BLOCK_N, D] by iterating dblocks via arange.
        # channel d -> dblock = d//16, c = d%16, byte = c//2, parity c&1.
        vdb = d // 16                                          # [D]
        vc = d % 16
        vbyte = vc // 2
        vhi = (vc % 2) == 1
        # offset (kvh*S + tok)*ND*PACK + vdb*PACK + vbyte
        voff = (kvh * S + tok[:, None]) * (ND * _PACK) + vdb[None, :] * _PACK + vbyte[None, :]  # [BLOCK_N, D]
        vb = tl.load(vpacked_ptr + voff, mask=tmask[:, None], other=0).to(tl.int32)
        vnib = tl.where(vhi[None, :], (vb >> 4) & 0xF, vb & 0xF)
        vnib = tl.where(vnib >= 8, vnib - 16, vnib)
        # scale v_scale[kvh, tok, db] : (kvh*S + tok)*ND + db
        vsoff = (kvh * S + tok[:, None]) * ND + vdb[None, :]   # [BLOCK_N, D]
        vsc = tl.load(vscale_ptr + vsoff, mask=tmask[:, None], other=0.0).to(tl.float32)
        vdeq = vnib.to(tl.float32) * vsc                      # [BLOCK_N, D]

        # ---- accumulate -------------------------------------------------
        acc = acc * alpha + tl.sum(pblk[:, None] * vdeq, axis=0)   # [D]
        l_i = l_i * alpha + tl.sum(pblk, axis=0)
        m_i = m_new

    # --- hot page: trailing bf16 tokens [n_full, S) ------------------------
    if HOT > 0:
        for hc in range(0, (HOT + BLOCK_N - 1) // BLOCK_N):
            h0 = hc * BLOCK_N
            ht = h0 + tl.arange(0, BLOCK_N)
            hmask = ht < HOT
            # khot[kvh, ht, d] : (kvh*HOT + ht)*D + d
            khoff = (kvh * HOT + ht[:, None]) * D + d[None, :]   # [BLOCK_N, D]
            kh = tl.load(khot_ptr + khoff, mask=hmask[:, None], other=0.0).to(tl.float32)  # [BLOCK_N, D]
            qk = tl.sum(q[None, :] * kh, axis=1)                 # [BLOCK_N]
            qk = tl.where(hmask, qk, -float("inf"))
            m_new = tl.maximum(m_i, tl.max(qk, axis=0))
            alpha = tl.exp(m_i - m_new)
            pblk = tl.exp(qk - m_new)
            pblk = tl.where(hmask, pblk, 0.0)
            # V for the hot tokens still lives in the int4 V cache (every token
            # is quantised), global token id = n_full + ht.
            vtok = n_full + ht
            vdb = d // 16
            vc = d % 16
            vbyte = vc // 2
            vhi = (vc % 2) == 1
            voff = (kvh * S + vtok[:, None]) * (ND * _PACK) + vdb[None, :] * _PACK + vbyte[None, :]
            vb = tl.load(vpacked_ptr + voff, mask=hmask[:, None], other=0).to(tl.int32)
            vnib = tl.where(vhi[None, :], (vb >> 4) & 0xF, vb & 0xF)
            vnib = tl.where(vnib >= 8, vnib - 16, vnib)
            vsoff = (kvh * S + vtok[:, None]) * ND + vdb[None, :]
            vsc = tl.load(vscale_ptr + vsoff, mask=hmask[:, None], other=0.0).to(tl.float32)
            vdeq = vnib.to(tl.float32) * vsc
            acc = acc * alpha + tl.sum(pblk[:, None] * vdeq, axis=0)
            l_i = l_i * alpha + tl.sum(pblk, axis=0)
            m_i = m_new

    out = acc / l_i
    tl.store(out_ptr + qh * D + d, out.to(tl.bfloat16))


def kivi_decode_attention(
    q: Tensor,
    cache: KIVICache,
    sm_scale: float | None = None,
    block_n: int = 64,
) -> Tensor:
    """Fused decode attention over a packed INT4-KIVI cache.

    q     : bf16 [n_q_heads, 1, D]  (one decode step's query, post-RoPE)
    cache : KIVICache for the same layer's KV (one batch element)
    returns out : bf16 [n_q_heads, 1, D]
    """
    assert q.is_cuda and cache.k_packed.is_cuda
    n_qh = q.shape[0]
    D = cache.D
    S = cache.S
    H = cache.H
    assert q.shape[-1] == D
    assert n_qh % H == 0, "n_q_heads must be a multiple of n_kv_heads"
    GROUP = n_qh // H
    NP = cache.k_packed.shape[2]
    n_full = NP * BLOCK
    HOT = S - n_full
    ND = cache.v_packed.shape[2]
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)

    q2 = q.reshape(n_qh, D).contiguous()
    out = torch.empty((n_qh, D), dtype=torch.bfloat16, device=q.device)

    khot = cache.k_hot
    if khot.numel() == 0:
        khot = torch.empty((H, 1, D), dtype=torch.bfloat16, device=q.device)

    grid = (n_qh,)
    _kivi_decode_kernel[grid](
        q2, cache.k_packed, cache.k_scale, khot.contiguous(),
        cache.v_packed, cache.v_scale, out,
        sm_scale, n_full, HOT, S,
        GROUP=GROUP, D=D, NP=NP, ND=ND, BLOCK_N=block_n,
        num_warps=4,
    )
    return out.reshape(n_qh, 1, D)


# --------------------------------------------------------------------------- #
# Split-K (flash-decode) variant for THROUGHPUT.  The cached prefix is split
# into SPLIT contiguous segments; one program handles (batch, q-head, segment),
# emitting a partial (m, l, acc).  A tiny combine kernel reduces across splits.
# This gives B * n_qh * SPLIT programs -> saturates the GPU so the decode is
# HBM-bandwidth bound, where reading 4-bit KV (~3.2x fewer bytes) wins.
#
# Batched packed cache layout (stacked over batch B at the FRONT of H):
#   k_packed [B*H, D, NP, PACK], k_scale [B*H, D, NP], k_hot [B*H, HOT, D],
#   v_packed [B*H, S, ND, PACK], v_scale [B*H, S, ND].
# Program (b, qh, s) reads kv-head index  (b*H + qh//GROUP).
# --------------------------------------------------------------------------- #
@triton.jit
def _kivi_decode_splitk_kernel(
    q_ptr,            # bf16 [B, n_qh, D]
    kpacked_ptr, kscale_ptr, khot_ptr,
    vpacked_ptr, vscale_ptr,
    pm_ptr, pl_ptr, pacc_ptr,     # partials: [B, n_qh, SPLIT], [.. ,SPLIT], [..,SPLIT,D]
    sm_scale, n_full, HOT, S,
    n_qh,
    GROUP: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    NP: tl.constexpr, ND: tl.constexpr,
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

    # this split covers tokens [seg0, seg1) of the FULL prefix [0, S).
    seg = (S + SPLIT - 1) // SPLIT
    seg0 = s * seg
    seg1 = tl.minimum(seg0 + seg, S)

    m_i = -float("inf")
    l_i = 0.0
    acc = tl.zeros([D], dtype=tl.float32)

    t = seg0
    while t < seg1:
        tok = t + tl.arange(0, BLOCK_N)
        tmask = tok < seg1
        is_q = tok < n_full          # int4-quantised region
        # ----- K -----
        # int4 path
        p = tok // _BLK
        j = tok % _BLK
        kbyte = j // 2
        khi = (j % 2) == 1
        koff = (kvh * D + d[:, None]) * (NP * _PACK) + p[None, :] * _PACK + kbyte[None, :]
        kb = tl.load(kpacked_ptr + koff, mask=(tmask & is_q)[None, :], other=0).to(tl.int32)
        knib = tl.where(khi[None, :], (kb >> 4) & 0xF, kb & 0xF)
        knib = tl.where(knib >= 8, knib - 16, knib)
        ksoff = (kvh * D + d[:, None]) * NP + p[None, :]
        ksc = tl.load(kscale_ptr + ksoff, mask=(tmask & is_q)[None, :], other=0.0).to(tl.float32)
        kq = knib.to(tl.float32) * ksc                  # [D, BLOCK_N]
        # hot path (bf16): khot[kvh, tok-n_full, d]
        hidx = tok - n_full
        khoff = (kvh * HOT + hidx[None, :]) * D + d[:, None]
        kh = tl.load(khot_ptr + khoff, mask=(tmask & (~is_q))[None, :], other=0.0).to(tl.float32)
        kdeq = tl.where(is_q[None, :], kq, kh)          # [D, BLOCK_N]

        qk = tl.sum(q[:, None] * kdeq, axis=0)          # [BLOCK_N]
        qk = tl.where(tmask, qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=0))
        alpha = tl.exp(m_i - m_new)
        pblk = tl.where(tmask, tl.exp(qk - m_new), 0.0)

        # ----- V (always int4, every token) -----
        vdb = d // _BLK
        vc = d % _BLK
        vbyte = vc // 2
        vhi = (vc % 2) == 1
        voff = (kvh * S + tok[:, None]) * (ND * _PACK) + vdb[None, :] * _PACK + vbyte[None, :]
        vb = tl.load(vpacked_ptr + voff, mask=tmask[:, None], other=0).to(tl.int32)
        vnib = tl.where(vhi[None, :], (vb >> 4) & 0xF, vb & 0xF)
        vnib = tl.where(vnib >= 8, vnib - 16, vnib)
        vsoff = (kvh * S + tok[:, None]) * ND + vdb[None, :]
        vsc = tl.load(vscale_ptr + vsoff, mask=tmask[:, None], other=0.0).to(tl.float32)
        vdeq = vnib.to(tl.float32) * vsc                # [BLOCK_N, D]

        acc = acc * alpha + tl.sum(pblk[:, None] * vdeq, axis=0)
        l_i = l_i * alpha + tl.sum(pblk, axis=0)
        m_i = m_new
        t += BLOCK_N

    # write partials
    base = (b * n_qh + qh) * SPLIT + s
    tl.store(pm_ptr + base, m_i)
    tl.store(pl_ptr + base, l_i)
    tl.store(pacc_ptr + base * D + d, acc)


@triton.jit
def _combine_kernel(
    pm_ptr, pl_ptr, pacc_ptr, out_ptr,
    n_qh, D: tl.constexpr, SPLIT: tl.constexpr,
):
    pid = tl.program_id(0)          # over B*n_qh
    d = tl.arange(0, D)
    # global max over splits
    m = -float("inf")
    for s in range(0, SPLIT):
        m = tl.maximum(m, tl.load(pm_ptr + pid * SPLIT + s))
    l = 0.0
    acc = tl.zeros([D], dtype=tl.float32)
    for s in range(0, SPLIT):
        ms = tl.load(pm_ptr + pid * SPLIT + s)
        ls = tl.load(pl_ptr + pid * SPLIT + s)
        a = tl.load(pacc_ptr + (pid * SPLIT + s) * D + d)
        scale = tl.exp(ms - m)
        acc += a * scale
        l += ls * scale
    tl.store(out_ptr + pid * D + d, (acc / l).to(tl.bfloat16))


# --------------------------------------------------------------------------- #
# GQA-grouped split-K: one program = (batch, kv_head, split) and processes ALL
# GROUP query heads of that kv head TOGETHER.  The packed KV is read ONCE per
# program and reused across the GROUP queries (q[GROUP,D] . k[D,N] -> [GROUP,N]).
# This (a) cuts KV traffic GROUP-fold vs per-q-head and (b) turns the inner
# product into a real matmul, so the kernel becomes HBM-bandwidth bound — the
# regime where reading 4-bit KV (~3.2x fewer bytes) actually wins on latency.
# --------------------------------------------------------------------------- #
@triton.jit
def _kivi_decode_gqa_kernel(
    q_ptr,            # bf16 [B, n_qh, D]
    kpacked_ptr, kscale_ptr, khot_ptr,
    vpacked_ptr, vscale_ptr,
    pm_ptr, pl_ptr, pacc_ptr,   # [B,H,GROUP,SPLIT], .., [B,H,GROUP,SPLIT,D]
    sm_scale, n_full, HOT, S, n_qh,
    GROUP: tl.constexpr, GPAD: tl.constexpr, D: tl.constexpr, H: tl.constexpr,
    NP: tl.constexpr, ND: tl.constexpr,
    SPLIT: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    s = pid % SPLIT
    tmp = pid // SPLIT
    kvh_local = tmp % H               # 0..H-1
    b = tmp // H
    kvh = b * H + kvh_local
    qh0 = kvh_local * GROUP           # first query head of this group within batch b

    d = tl.arange(0, D)               # [D]
    gr = tl.arange(0, GPAD)           # [GPAD] (>= GROUP, power of 2)
    gmask = gr < GROUP
    # q block [GPAD, D]; padded rows zeroed so their scores are harmless.
    qoff = (b * n_qh + qh0 + gr[:, None]) * D + d[None, :]
    q = tl.load(q_ptr + qoff, mask=gmask[:, None], other=0.0).to(tl.float32) * sm_scale

    seg = (S + SPLIT - 1) // SPLIT
    seg0 = s * seg
    seg1 = tl.minimum(seg0 + seg, S)

    m_i = tl.full([GPAD], -float("inf"), tl.float32)
    l_i = tl.zeros([GPAD], tl.float32)
    acc = tl.zeros([GPAD, D], tl.float32)

    t = seg0
    while t < seg1:
        tok = t + tl.arange(0, BLOCK_N)
        tmask = tok < seg1
        is_q = tok < n_full
        # ----- K [D, BLOCK_N] (int4 or hot bf16) -----
        p = tok // _BLK
        j = tok % _BLK
        kbyte = j // 2
        khi = (j % 2) == 1
        koff = (kvh * D + d[:, None]) * (NP * _PACK) + p[None, :] * _PACK + kbyte[None, :]
        kb = tl.load(kpacked_ptr + koff, mask=(tmask & is_q)[None, :], other=0).to(tl.int32)
        knib = tl.where(khi[None, :], (kb >> 4) & 0xF, kb & 0xF)
        knib = tl.where(knib >= 8, knib - 16, knib)
        ksoff = (kvh * D + d[:, None]) * NP + p[None, :]
        ksc = tl.load(kscale_ptr + ksoff, mask=(tmask & is_q)[None, :], other=0.0).to(tl.float32)
        kq = knib.to(tl.float32) * ksc
        hidx = tok - n_full
        khoff = (kvh * HOT + hidx[None, :]) * D + d[:, None]
        kh = tl.load(khot_ptr + khoff, mask=(tmask & (~is_q))[None, :], other=0.0).to(tl.float32)
        kdeq = tl.where(is_q[None, :], kq, kh)            # [D, BLOCK_N]

        qk = tl.dot(q, kdeq)                               # [GROUP, BLOCK_N]
        qk = tl.where(tmask[None, :], qk, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))        # [GROUP]
        alpha = tl.exp(m_i - m_new)
        pblk = tl.exp(qk - m_new[:, None])                 # [GROUP, BLOCK_N]
        pblk = tl.where(tmask[None, :], pblk, 0.0)

        # ----- V [BLOCK_N, D] (int4) -----
        vdb = d // _BLK
        vc = d % _BLK
        vbyte = vc // 2
        vhi = (vc % 2) == 1
        voff = (kvh * S + tok[:, None]) * (ND * _PACK) + vdb[None, :] * _PACK + vbyte[None, :]
        vb = tl.load(vpacked_ptr + voff, mask=tmask[:, None], other=0).to(tl.int32)
        vnib = tl.where(vhi[None, :], (vb >> 4) & 0xF, vb & 0xF)
        vnib = tl.where(vnib >= 8, vnib - 16, vnib)
        vsoff = (kvh * S + tok[:, None]) * ND + vdb[None, :]
        vsc = tl.load(vscale_ptr + vsoff, mask=tmask[:, None], other=0.0).to(tl.float32)
        vdeq = vnib.to(tl.float32) * vsc                   # [BLOCK_N, D]

        acc = acc * alpha[:, None] + tl.dot(pblk.to(tl.float32), vdeq)   # [GROUP, D]
        l_i = l_i * alpha + tl.sum(pblk, axis=1)
        m_i = m_new
        t += BLOCK_N

    # store partials [b, kvh_local, gr, s] (only the GROUP valid rows)
    base = ((b * H + kvh_local) * GROUP + gr) * SPLIT + s    # [GPAD]
    tl.store(pm_ptr + base, m_i, mask=gmask)
    tl.store(pl_ptr + base, l_i, mask=gmask)
    tl.store(pacc_ptr + base[:, None] * D + d[None, :], acc, mask=gmask[:, None])


def kivi_decode_attention_gqa(
    q: Tensor, stacked: dict, n_qh: int,
    sm_scale: float | None = None, split: int = 16, block_n: int = 64,
) -> Tensor:
    """GQA-grouped split-K fused decode (fast path). q [B, n_qh, D] bf16."""
    B = stacked["B"]; H = stacked["H"]; S = stacked["S"]; D = stacked["D"]
    NP = stacked["NP"]; ND = stacked["ND"]
    GROUP = n_qh // H
    n_full = NP * BLOCK
    HOT = S - n_full
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    split = max(1, min(split, (S + block_n - 1) // block_n))

    khot = stacked["k_hot"]
    if khot.shape[1] == 0:
        khot = torch.empty((B * H, 1, D), dtype=torch.bfloat16, device=q.device)
        HOT = 1
    dev = q.device
    npart = B * H * GROUP * split
    pm = torch.empty((npart,), dtype=torch.float32, device=dev)
    pl = torch.empty((npart,), dtype=torch.float32, device=dev)
    pacc = torch.empty((npart, D), dtype=torch.float32, device=dev)
    qc = q.reshape(B, n_qh, D).contiguous()
    GPAD = 1 << (GROUP - 1).bit_length()
    GPAD = max(GPAD, 16)   # tl.dot needs M>=16

    _kivi_decode_gqa_kernel[(B * H * split,)](
        qc, stacked["k_packed"], stacked["k_scale"], khot.contiguous(),
        stacked["v_packed"], stacked["v_scale"],
        pm, pl, pacc, sm_scale, n_full, HOT, S, n_qh,
        GROUP=GROUP, GPAD=GPAD, D=D, H=H, NP=NP, ND=ND,
        SPLIT=split, BLOCK_N=block_n, num_warps=4,
    )
    # combine: partials are laid out exactly [B*n_qh, SPLIT] since
    # ((b*H+kvh_local)*GROUP+gr) == b*n_qh + qh.
    out = torch.empty((B * n_qh, D), dtype=torch.bfloat16, device=dev)
    _combine_kernel[(B * n_qh,)](pm, pl, pacc, out, n_qh, D=D, SPLIT=split)
    return out.reshape(B, n_qh, D)


def stack_caches(caches: list[KIVICache]) -> dict:
    """Stack per-batch KIVICaches into batched packed tensors (B*H front dim)."""
    return dict(
        k_packed=torch.cat([c.k_packed for c in caches], dim=0),
        k_scale=torch.cat([c.k_scale for c in caches], dim=0),
        k_hot=torch.cat([c.k_hot for c in caches], dim=0),
        v_packed=torch.cat([c.v_packed for c in caches], dim=0),
        v_scale=torch.cat([c.v_scale for c in caches], dim=0),
        H=caches[0].H, S=caches[0].S, D=caches[0].D,
        NP=caches[0].k_packed.shape[2], ND=caches[0].v_packed.shape[2],
        B=len(caches),
    )


def kivi_decode_attention_batched(
    q: Tensor, stacked: dict, n_qh: int,
    sm_scale: float | None = None, split: int = 8, block_n: int = 64,
) -> Tensor:
    """Split-K batched fused decode. q [B, n_qh, D] bf16, stacked from stack_caches."""
    B = stacked["B"]
    H = stacked["H"]; S = stacked["S"]; D = stacked["D"]
    NP = stacked["NP"]; ND = stacked["ND"]
    GROUP = n_qh // H
    n_full = NP * BLOCK
    HOT = S - n_full
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(D)
    # clamp split so each segment is non-trivial
    split = max(1, min(split, (S + block_n - 1) // block_n))

    khot = stacked["k_hot"]
    if khot.shape[1] == 0:
        khot = torch.empty((B * H, 1, D), dtype=torch.bfloat16, device=q.device)
        HOT = 1  # dummy; never read because is_q covers all tokens when n_full==S
        # is_q = tok < n_full == S so hot branch masked off; HOT only sizes the ptr.

    dev = q.device
    pm = torch.empty((B * n_qh * split,), dtype=torch.float32, device=dev)
    pl = torch.empty((B * n_qh * split,), dtype=torch.float32, device=dev)
    pacc = torch.empty((B * n_qh * split, D), dtype=torch.float32, device=dev)
    qc = q.reshape(B, n_qh, D).contiguous()

    grid = (B * n_qh * split,)
    _kivi_decode_splitk_kernel[grid](
        qc, stacked["k_packed"], stacked["k_scale"], khot.contiguous(),
        stacked["v_packed"], stacked["v_scale"],
        pm, pl, pacc,
        sm_scale, n_full, HOT, S, n_qh,
        GROUP=GROUP, D=D, H=H, NP=NP, ND=ND,
        SPLIT=split, BLOCK_N=block_n, num_warps=4,
    )
    out = torch.empty((B * n_qh, D), dtype=torch.bfloat16, device=dev)
    _combine_kernel[(B * n_qh,)](pm, pl, pacc, out, n_qh, D=D, SPLIT=split)
    return out.reshape(B, n_qh, D)
