"""Triton INT4-KIVI store/dequant kernels.

Numerically faithful to ``kv_quant.roundtrip`` (the validated reference):

* INT4 symmetric on [-7, 7]:  code = round(x/scale).clamp(-7, 7); deq = code*scale.
* All quant math in fp32.
* round = round-half-to-even (torch.round) -> libdevice.rint in Triton.
* MSE calibration: alphas = linspace(0.5, 1.0, 32); scale = alpha*absmax/QMAX;
  pick the alpha minimising mean((x - round(x/scale).clamp*scale)^2) over the
  16-element block.  Ties resolve to the *smallest* alpha (matches torch.argmin,
  which returns the lowest index).

Block geometry (BLOCK = 16):

* K -> 'channel' layout: a block is 16 *tokens* of one channel.  One scale per
  (head, channel, page).  Only the first (S//16)*16 tokens are quantised; the
  trailing S%16 tokens are the bf16 "hot page" (handled in cache.py, not here).
* V -> 'headdim' layout: a block is 16 contiguous *channels* of one token.  One
  scale per (head, token, dblock).  Every token is quantised.

Packing: two signed 4-bit codes per uint8 byte.  A code c in [-7, 7] is stored
as the low nibble (c & 0xF, two's complement); unpacked by sign-extending the
nibble.  Within a 16-element block, element 2*j -> low nibble of byte j,
element 2*j+1 -> high nibble of byte j  (8 bytes per block).
"""

from __future__ import annotations

import triton
import triton.language as tl
from triton.language.extra import libdevice

BLOCK = 16
QMAX = 7
N_ALPHAS = 32
PACK = BLOCK // 2  # bytes per quantised block (2 codes/byte)

# Triton 3.7 forbids reading plain module globals inside @jit functions; the
# kernels reference these constexpr aliases instead.  Host code uses the ints.
# NOTE: must be `tl.constexpr(v)`, not an annotation, per Triton's check.
_BLOCK = tl.constexpr(BLOCK)
# float so libdevice.div_rn (no float/int overload) and the [-7,7] clamp typecheck
_QMAX = tl.constexpr(float(QMAX))
_N_ALPHAS = tl.constexpr(N_ALPHAS)
_PACK = tl.constexpr(PACK)


# --------------------------------------------------------------------------- #
# device helpers
# --------------------------------------------------------------------------- #
@triton.jit
def _round_even(x):
    # round-half-to-even, matching torch.round / libdevice rint (IEEE nearbyint).
    return libdevice.rint(x)


@triton.jit
def _quant_codes(x, scale):
    # x, scale fp32 -> signed int4 codes in [-7, 7] as fp32.
    # IEEE round-to-nearest division (libdevice.div_rn) matches torch's CUDA
    # `/` bit-for-bit; the default Triton `/` may use a faster reciprocal that
    # diverges by 1 ULP and flips exact-.5 rounding ties.
    q = _round_even(libdevice.div_rn(x, scale))
    q = tl.minimum(tl.maximum(q, -_QMAX), _QMAX)
    return q


@triton.jit
def _absmax_scale(x_blk, mask):
    # x_blk: [BN, BLOCK] fp32.  Returns [BN] fp32 = absmax/QMAX, clamp_min 1e-9.
    xabs = tl.where(mask, tl.abs(x_blk), 0.0)
    amax = tl.max(xabs, axis=1)
    amax = tl.maximum(amax, 1e-9)
    return libdevice.div_rn(amax, _QMAX)


@triton.jit
def _mse_scale(x_blk, mask, alpha_ptr):
    # x_blk: [BN, BLOCK] fp32.  Grid-search the clip ratio alpha (the exact
    # torch.linspace(0.5, 1.0, N_ALPHAS) values, passed via alpha_ptr) and pick
    # the smallest alpha minimising blockwise MSE.  Returns [BN] fp32.
    #
    # Bit-exact match to kv_quant._calibrate('mse'):
    #   * scale = (alpha * absmax) / QMAX  (this multiply/divide ORDER matters)
    #   * division uses IEEE round-to-nearest (via _quant_codes / div_rn)
    #   * err = mean over the block of (x - q*scale)^2
    #   * argmin resolves ties to the smallest alpha (strict '<').
    xabs = tl.where(mask, tl.abs(x_blk), 0.0)
    amax = tl.max(xabs, axis=1)
    amax = tl.maximum(amax, 1e-9)               # [BN]

    best_err = tl.full(amax.shape, float("inf"), tl.float32)
    best_scale = amax  # placeholder, overwritten on i=0 (inf > err always)
    for i in range(_N_ALPHAS):
        alpha = tl.load(alpha_ptr + i)                       # scalar fp32
        scale = libdevice.div_rn(alpha * amax, _QMAX)        # [BN]; (alpha*absmax)/QMAX
        s = scale[:, None]
        q = _quant_codes(x_blk, s)
        diff = x_blk - q * s
        sq = tl.where(mask, diff * diff, 0.0)
        err = tl.sum(sq, axis=1) / _BLOCK  # mean over the 16-elem block
        # strict '<' keeps the FIRST (smallest) alpha on ties, == torch.argmin.
        take = err < best_err
        best_err = tl.where(take, err, best_err)
        best_scale = tl.where(take, scale, best_scale)
    return best_scale


@triton.jit
def _pack_block(codes, BN: tl.constexpr):
    # codes: [BN, BLOCK] fp32 signed in [-7,7] -> packed [BN, PACK] uint8.
    # element 2*j -> low nibble of byte j ; element 2*j+1 -> high nibble.
    ci = codes.to(tl.int32)
    nib = ci & 0xF  # two's-complement low nibble
    nib3 = tl.reshape(nib, (BN, _PACK, 2))      # [BN, PACK, 2]
    lo, hi = tl.split(nib3)                      # each [BN, PACK]
    packed = (lo | (hi << 4)).to(tl.uint8)
    return packed


@triton.jit
def _unpack_block(packed):
    # packed: [BN, PACK] uint8 -> codes [BN, BLOCK] fp32 signed in [-7,7].
    p = packed.to(tl.int32)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    # sign-extend 4-bit nibbles: values >= 8 are negative.
    lo = tl.where(lo >= 8, lo - 16, lo)
    hi = tl.where(hi >= 8, hi - 16, hi)
    codes = tl.interleave(lo, hi)  # [BN, BLOCK]; (lo0,hi0,lo1,hi1,...)
    return codes.to(tl.float32)


# --------------------------------------------------------------------------- #
# K store: 'channel' layout (16-token blocks of one channel)
# --------------------------------------------------------------------------- #
@triton.jit
def k_store_kernel(
    k_ptr,            # bf16 [H, S, D]
    packed_ptr,       # uint8 [H, D, NP, PACK]
    scale_ptr,        # fp16  [H, D, NP]
    alpha_ptr,        # fp32  [N_ALPHAS]  (linspace(0.5,1.0,N_ALPHAS); MSE only)
    H, S, D, NP,
    sk_h, sk_s, sk_d,         # strides of k (elements)
    USE_MSE: tl.constexpr,
    BLOCK_D: tl.constexpr,    # channels processed per program
):
    pid_h = tl.program_id(0)
    pid_p = tl.program_id(1)      # page index (0..NP-1)
    pid_dg = tl.program_id(2)     # channel-group index

    d_idx = pid_dg * BLOCK_D + tl.arange(0, BLOCK_D)      # [BLOCK_D]
    d_mask = d_idx < D
    tok = pid_p * _BLOCK + tl.arange(0, _BLOCK)           # [BLOCK] token positions

    # load [BLOCK_D channels, BLOCK tokens] for this (head, page).
    # k offset = h*sk_h + tok*sk_s + d*sk_d
    off = pid_h * sk_h + tok[None, :] * sk_s + d_idx[:, None] * sk_d
    blk = tl.load(k_ptr + off, mask=d_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_D, BLOCK]
    mask = tl.broadcast_to(d_mask[:, None], (BLOCK_D, _BLOCK))

    if USE_MSE:
        scale = _mse_scale(blk, mask, alpha_ptr)     # [BLOCK_D]
    else:
        scale = _absmax_scale(blk, mask)

    codes = _quant_codes(blk, scale[:, None])   # [BLOCK_D, BLOCK]
    packed = _pack_block(codes, BLOCK_D)        # [BLOCK_D, PACK]

    # store packed: packed_ptr[h, d, p, :]
    base = (pid_h * D + d_idx) * NP * _PACK + pid_p * _PACK     # [BLOCK_D]
    pcols = tl.arange(0, _PACK)
    p_off = base[:, None] + pcols[None, :]
    tl.store(packed_ptr + p_off, packed, mask=d_mask[:, None])

    # store scale: scale_ptr[h, d, p]
    s_off = (pid_h * D + d_idx) * NP + pid_p
    tl.store(scale_ptr + s_off, scale.to(tl.float16), mask=d_mask)


@triton.jit
def k_dequant_kernel(
    packed_ptr,       # uint8 [H, D, NP, PACK]
    scale_ptr,        # fp16  [H, D, NP]
    out_ptr,          # bf16  [H, S, D]   (only the first NP*BLOCK tokens written)
    H, S, D, NP,
    so_h, so_s, so_d,        # strides of out (elements)
    BLOCK_D: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_dg = tl.program_id(2)

    d_idx = pid_dg * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_idx < D
    pcols = tl.arange(0, _PACK)

    p_base = (pid_h * D + d_idx) * NP * _PACK + pid_p * _PACK
    p_off = p_base[:, None] + pcols[None, :]
    packed = tl.load(packed_ptr + p_off, mask=d_mask[:, None], other=0).to(tl.uint8)
    codes = _unpack_block(packed)                 # [BLOCK_D, BLOCK] fp32

    s_off = (pid_h * D + d_idx) * NP + pid_p
    scale = tl.load(scale_ptr + s_off, mask=d_mask, other=0.0).to(tl.float32)

    deq = codes * scale[:, None]                  # [BLOCK_D, BLOCK]

    tok = pid_p * _BLOCK + tl.arange(0, _BLOCK)
    off = pid_h * so_h + tok[None, :] * so_s + d_idx[:, None] * so_d
    tl.store(out_ptr + off, deq.to(tl.bfloat16), mask=d_mask[:, None])


# --------------------------------------------------------------------------- #
# V store: 'headdim' layout (16 contiguous channels of one token)
# --------------------------------------------------------------------------- #
@triton.jit
def v_store_kernel(
    v_ptr,            # bf16 [H, S, D]
    packed_ptr,       # uint8 [H, S, ND, PACK]
    scale_ptr,        # fp16  [H, S, ND]
    alpha_ptr,        # fp32  [N_ALPHAS]  (linspace(0.5,1.0,N_ALPHAS); MSE only)
    H, S, D, ND,
    sv_h, sv_s, sv_d,
    USE_MSE: tl.constexpr,
    BLOCK_T: tl.constexpr,    # tokens processed per program
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)      # token-group index
    pid_db = tl.program_id(2)     # dblock index (0..ND-1)

    t_idx = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)       # [BLOCK_T] tokens
    t_mask = t_idx < S
    ch = pid_db * _BLOCK + tl.arange(0, _BLOCK)           # [BLOCK] channels

    # load [BLOCK_T tokens, BLOCK channels] for this (head, dblock).
    off = pid_h * sv_h + t_idx[:, None] * sv_s + ch[None, :] * sv_d
    blk = tl.load(v_ptr + off, mask=t_mask[:, None], other=0.0).to(tl.float32)  # [BLOCK_T, BLOCK]
    mask = tl.broadcast_to(t_mask[:, None], (BLOCK_T, _BLOCK))

    if USE_MSE:
        scale = _mse_scale(blk, mask, alpha_ptr)     # [BLOCK_T]
    else:
        scale = _absmax_scale(blk, mask)

    codes = _quant_codes(blk, scale[:, None])
    packed = _pack_block(codes, BLOCK_T)        # [BLOCK_T, PACK]

    base = (pid_h * S + t_idx) * ND * _PACK + pid_db * _PACK
    pcols = tl.arange(0, _PACK)
    p_off = base[:, None] + pcols[None, :]
    tl.store(packed_ptr + p_off, packed, mask=t_mask[:, None])

    s_off = (pid_h * S + t_idx) * ND + pid_db
    tl.store(scale_ptr + s_off, scale.to(tl.float16), mask=t_mask)


@triton.jit
def v_dequant_kernel(
    packed_ptr,       # uint8 [H, S, ND, PACK]
    scale_ptr,        # fp16  [H, S, ND]
    out_ptr,          # bf16  [H, S, D]
    H, S, D, ND,
    so_h, so_s, so_d,
    BLOCK_T: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_t = tl.program_id(1)
    pid_db = tl.program_id(2)

    t_idx = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    t_mask = t_idx < S
    pcols = tl.arange(0, _PACK)

    p_base = (pid_h * S + t_idx) * ND * _PACK + pid_db * _PACK
    p_off = p_base[:, None] + pcols[None, :]
    packed = tl.load(packed_ptr + p_off, mask=t_mask[:, None], other=0).to(tl.uint8)
    codes = _unpack_block(packed)                 # [BLOCK_T, BLOCK]

    s_off = (pid_h * S + t_idx) * ND + pid_db
    scale = tl.load(scale_ptr + s_off, mask=t_mask, other=0.0).to(tl.float32)

    deq = codes * scale[:, None]

    ch = pid_db * _BLOCK + tl.arange(0, _BLOCK)
    off = pid_h * so_h + t_idx[:, None] * so_s + ch[None, :] * so_d
    tl.store(out_ptr + off, deq.to(tl.bfloat16), mask=t_mask[:, None])
