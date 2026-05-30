"""Parity of the Triton INT4-KIVI kernels against kv_quant.py (the validated ref).

Strong check: the INT4 *codes* the Triton kernels write are compared bit-for-bit
against ``kv_quant`` reference codes (same fp32 scale math), so this is layout +
quantization parity, not just "close numbers". A second check confirms the
dequantized BF16 output tracks ``kv_quant.roundtrip`` (the only intended
divergence is fp16 vs fp32 scale storage).

Run: ``.venv/bin/python -m pytest tests/test_int4_kivi.py -v``
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import kv_quant  # noqa: E402
from kv_quant import _FORMATS, _calibrate, _to_blocks, roundtrip  # noqa: E402
from int4_kivi import BLOCK, PACK, dequant_kivi, store_kivi  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
H, D = 8, 128  # Laguna-XS.2 KV heads / head_dim


def _unpack(packed: torch.Tensor) -> torch.Tensor:
    """[..., PACK] uint8 -> [..., 2*PACK] int codes, inverse of the kernel pack.

    byte j: low nibble = code[2j], high nibble = code[2j+1] (two's complement)."""
    p = packed.to(torch.int16)
    lo = p & 0xF
    hi = (p >> 4) & 0xF
    lo = torch.where(lo >= 8, lo - 16, lo)
    hi = torch.where(hi >= 8, hi - 16, hi)
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


def _ref_codes(x: torch.Tensor, layout: str, calib: str) -> torch.Tensor:
    """kv_quant reference INT4 codes for x [H,S,D] under (layout, calib)."""
    qmax, qfn = _FORMATS["int4"]
    xb = _to_blocks(x, layout).float()
    scale = _calibrate(xb, qmax, qfn, calib)
    return qfn(xb / scale).to(torch.int16)  # [..., BLOCK], in [-7,7]


def _rand_kv(S: int, seed: int = 0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    # heavy-tailed so per-channel K outliers actually exist (the whole point)
    k = (torch.randn(H, S, D, generator=g, device=DEV) ** 3).to(torch.bfloat16)
    v = torch.randn(H, S, D, generator=g, device=DEV).to(torch.bfloat16)
    return k, v


# ─────────────────────────────── code parity ───────────────────────────────
def _assert_code_parity(got, ref, tag):
    """Correctness invariant for a quantizer vs the reference:

    a correct implementation can differ from kv_quant ONLY by rounding-boundary
    flips (a value sitting on a k+0.5 edge that Triton's libdevice div/rint and
    torch resolve 1 ULP apart). Those are necessarily off-by-EXACTLY-1 and rare.
    A layout/logic bug instead produces large and/or frequent diffs — so we
    assert both bounds, which is far stronger than a loose match threshold."""
    diff = (got.to(torch.int32) - ref.to(torch.int32)).abs()
    maxdiff = diff.max().item()
    frac = (diff > 0).float().mean().item()
    assert maxdiff <= 1, f"{tag}: max|code-diff|={maxdiff} (>1 => layout/logic bug)"
    assert frac <= 0.005, f"{tag}: mismatch frac {frac:.5f} (boundary flips should be <0.5%)"


@cuda
@pytest.mark.parametrize("calib", ["absmax", "mse"])
@pytest.mark.parametrize("S", [256, 512, 2048])
def test_k_code_parity(S, calib):
    """Per-channel K: Triton codes == kv_quant 'channel' codes (mod boundary flips)."""
    k, v = _rand_kv(S)
    cache = store_kivi(k, v, k_calib=calib, v_calib=calib)
    n_full = (S // BLOCK) * BLOCK
    got = _unpack(cache.k_packed)                       # [H, D, NP, BLOCK]
    ref = _ref_codes(k[:, :n_full], "channel", calib)   # [H, D, NP, BLOCK]
    _assert_code_parity(got, ref, f"K {calib} S={S}")


@cuda
@pytest.mark.parametrize("calib", ["absmax", "mse"])
@pytest.mark.parametrize("S", [256, 512, 2048])
def test_v_code_parity(S, calib):
    """Per-token V: Triton codes == kv_quant 'headdim' codes (mod boundary flips)."""
    k, v = _rand_kv(S)
    cache = store_kivi(k, v, k_calib=calib, v_calib=calib)
    got = _unpack(cache.v_packed)                       # [H, S, ND, BLOCK]
    ref = _ref_codes(v, "headdim", calib)               # [H, S, ND, BLOCK]
    _assert_code_parity(got, ref, f"V {calib} S={S}")


# ───────────────────────────── dequant parity ──────────────────────────────
@cuda
@pytest.mark.parametrize("calib", ["absmax", "mse"])
def test_dequant_matches_roundtrip(calib):
    """dequant_kivi tracks kv_quant.roundtrip in aggregate.

    Two principled sources of divergence: fp16 scale storage (~2^-11 rel) and the
    rare off-by-1 boundary flips. On heavy-tailed K a single flip in a
    large-scale channel is a big *absolute* diff, so we measure *relative RMSE*,
    which a real bug (wrong layout/scale) would blow past."""
    S = 2048
    k, v = _rand_kv(S, seed=1)
    cache = store_kivi(k, v, k_calib=calib, v_calib=calib)
    kq, vq = dequant_kivi(cache)
    n_full = (S // BLOCK) * BLOCK
    ref_k = roundtrip(k, "int4", "channel", calib)
    ref_v = roundtrip(v, "int4", "headdim", calib)

    def rel_rmse(a, b):
        rmse = ((a.float() - b.float()) ** 2).mean().sqrt()
        return (rmse / b.float().abs().mean()).item()

    rk = rel_rmse(kq[:, :n_full], ref_k[:, :n_full])
    rv = rel_rmse(vq, ref_v)
    # noise floor (proven): off-by-1 boundary flips + fp16 scale. absmax is the
    # worst case (~1.8% K, ~1.5% V); a wrong layout/scale would be >>10%.
    assert rk < 0.03, f"K dequant rel-RMSE {rk:.4f}"
    assert rv < 0.02, f"V dequant rel-RMSE {rv:.4f}"


# ─────────────────────────────── hot page ──────────────────────────────────
@cuda
def test_k_hot_page_tail_bf16():
    """S % 16 != 0: the trailing tokens stay BF16, exactly like roundtrip."""
    S = 40  # n_full=32, tail=8
    k, v = _rand_kv(S, seed=2)
    cache = store_kivi(k, v, k_calib="mse", v_calib="mse")
    assert cache.k_hot.shape == (H, 8, D)
    kq, _ = dequant_kivi(cache)
    ref_k = roundtrip(k, "int4", "channel", "mse")
    # tail is verbatim bf16 in both
    assert torch.equal(kq[:, 32:], k[:, 32:])
    assert torch.equal(kq[:, 32:], ref_k[:, 32:])


# ─────────────────────────────── memory ────────────────────────────────────
@cuda
def test_memory_ratio():
    S = 2048
    k, v = _rand_kv(S)
    cache = store_kivi(k, v)
    ratio = cache.compression_ratio_vs_bf16()
    assert 3.0 <= ratio <= 3.4, f"fp16-scale ratio {ratio:.3f}"

    # Projected 1-byte (e4m3) scales == NVFP4's choice == PROBLEM.md's 3.56x.
    data = cache.k_packed.numel() + cache.v_packed.numel()
    n_scales = cache.k_scale.numel() + cache.v_scale.numel()
    hot = cache.k_hot.numel() * 2
    bf16 = cache.bf16_nbytes()
    ratio_1b = bf16 / (data + n_scales * 1 + hot)
    assert 3.45 <= ratio_1b <= 3.65, f"1-byte-scale ratio {ratio_1b:.3f}"
