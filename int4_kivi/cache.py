"""Host-side INT4-KIVI cache + store/dequant wrappers.

Scheme (validated KIVI, see PROBLEM.md / scripts/quant_ab.py):
    K = int4 / 'channel' (per-channel, 16-token page blocks) / mse
    V = int4 / 'headdim' (per-token, 16-channel blocks)       / mse

Operates page-granular on full tensors [H, S, D] (no slot_mapping / streaming;
that is the vLLM integration layer's concern).  For K, only the first
(S // 16) * 16 tokens are quantised — the trailing S % 16 tokens are kept in
bf16 (the "hot page"), exactly as kv_quant.roundtrip(..., 'channel', ...) does.

Conceptual page layout (per 16-token K page / per token for V):
    [ K_data | K_scale | V_data | V_scale ]
with INT4 data at 0.5 B/code (two codes per uint8) and scales in fp16.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import Tensor

from .triton_kernels import (
    BLOCK,
    PACK,
    N_ALPHAS,
    k_store_kernel,
    k_dequant_kernel,
    v_store_kernel,
    v_dequant_kernel,
)


def _grid_blocks(n: int, b: int) -> int:
    return (n + b - 1) // b


def _alphas(device) -> Tensor:
    """The exact MSE clip-ratio grid: torch.linspace(0.5, 1.0, N_ALPHAS) fp32.

    Computed with torch.linspace (not a manual formula) so the in-kernel values
    are bit-identical to kv_quant._calibrate, which also uses torch.linspace.
    """
    return torch.linspace(0.5, 1.0, N_ALPHAS, device=device, dtype=torch.float32)


@dataclasses.dataclass
class KIVICache:
    """Packed INT4-KIVI K/V cache for one layer's [H, S, D] tensors."""

    # K: 'channel' layout, only the first n_full = (S//16)*16 tokens quantised.
    k_packed: Tensor          # uint8 [H, D, NP, PACK]      (NP = n_full // BLOCK)
    k_scale: Tensor           # fp16  [H, D, NP]
    k_hot: Tensor             # bf16  [H, S - n_full, D]    (the bf16 hot-page tail)

    # V: 'headdim' layout, every token quantised.
    v_packed: Tensor          # uint8 [H, S, ND, PACK]      (ND = D // BLOCK)
    v_scale: Tensor           # fp16  [H, S, ND]

    H: int
    S: int
    D: int

    @property
    def n_full(self) -> int:
        return (self.S // BLOCK) * BLOCK

    @property
    def nbytes(self) -> int:
        return (
            self.k_packed.numel() * self.k_packed.element_size()
            + self.k_scale.numel() * self.k_scale.element_size()
            + self.k_hot.numel() * self.k_hot.element_size()
            + self.v_packed.numel() * self.v_packed.element_size()
            + self.v_scale.numel() * self.v_scale.element_size()
        )

    def bf16_nbytes(self) -> int:
        # full K + full V in bf16.
        return 2 * (self.H * self.S * self.D) * 2

    def compression_ratio_vs_bf16(self) -> float:
        return self.bf16_nbytes() / max(self.nbytes, 1)


# --------------------------------------------------------------------------- #
# store / quantize
# --------------------------------------------------------------------------- #
def _store_k(k: Tensor, calib: str) -> tuple[Tensor, Tensor, Tensor]:
    """k bf16 [H, S, D] -> (k_packed [H,D,NP,PACK] uint8, k_scale [H,D,NP] fp16,
    k_hot [H, S-n_full, D] bf16)."""
    H, S, D = k.shape
    assert D % BLOCK == 0, "head_dim must be a multiple of 16"
    n_full = (S // BLOCK) * BLOCK
    NP = n_full // BLOCK

    k_hot = k[:, n_full:].contiguous() if n_full < S else k.new_empty((H, 0, D), dtype=torch.bfloat16)
    if NP == 0:
        k_packed = k.new_empty((H, D, 0, PACK), dtype=torch.uint8)
        k_scale = k.new_empty((H, D, 0), dtype=torch.float16)
        return k_packed, k_scale, k_hot.to(torch.bfloat16)

    k = k.contiguous()
    k_packed = torch.empty((H, D, NP, PACK), dtype=torch.uint8, device=k.device)
    k_scale = torch.empty((H, D, NP), dtype=torch.float16, device=k.device)
    alphas = _alphas(k.device)

    BLOCK_D = 16
    grid = (H, NP, _grid_blocks(D, BLOCK_D))
    k_store_kernel[grid](
        k, k_packed, k_scale, alphas,
        H, S, D, NP,
        k.stride(0), k.stride(1), k.stride(2),
        USE_MSE=(calib == "mse"),
        BLOCK_D=BLOCK_D,
    )
    return k_packed, k_scale, k_hot.to(torch.bfloat16)


def _store_v(v: Tensor, calib: str) -> tuple[Tensor, Tensor]:
    """v bf16 [H, S, D] -> (v_packed [H,S,ND,PACK] uint8, v_scale [H,S,ND] fp16)."""
    H, S, D = v.shape
    assert D % BLOCK == 0, "head_dim must be a multiple of 16"
    ND = D // BLOCK

    v = v.contiguous()
    v_packed = torch.empty((H, S, ND, PACK), dtype=torch.uint8, device=v.device)
    v_scale = torch.empty((H, S, ND), dtype=torch.float16, device=v.device)
    alphas = _alphas(v.device)

    BLOCK_T = 16
    grid = (H, _grid_blocks(S, BLOCK_T), ND)
    v_store_kernel[grid](
        v, v_packed, v_scale, alphas,
        H, S, D, ND,
        v.stride(0), v.stride(1), v.stride(2),
        USE_MSE=(calib == "mse"),
        BLOCK_T=BLOCK_T,
    )
    return v_packed, v_scale


def store_kivi(k: Tensor, v: Tensor, k_calib: str = "mse", v_calib: str = "mse") -> KIVICache:
    """Quantize K (per-channel) and V (per-token) into a KIVICache.

    k, v: bf16 [H, S, D] (S a multiple of 16 for full K coverage; D a multiple
    of 16).  calib in {"absmax", "mse"}.
    """
    assert k.shape == v.shape, "k and v must share shape [H, S, D]"
    assert k.is_cuda and v.is_cuda, "inputs must be on CUDA"
    H, S, D = k.shape
    k_packed, k_scale, k_hot = _store_k(k.to(torch.bfloat16), k_calib)
    v_packed, v_scale = _store_v(v.to(torch.bfloat16), v_calib)
    return KIVICache(
        k_packed=k_packed, k_scale=k_scale, k_hot=k_hot,
        v_packed=v_packed, v_scale=v_scale,
        H=H, S=S, D=D,
    )


# --------------------------------------------------------------------------- #
# dequant
# --------------------------------------------------------------------------- #
def _dequant_k(cache: KIVICache) -> Tensor:
    H, S, D = cache.H, cache.S, cache.D
    NP = cache.k_packed.shape[2]
    out = torch.empty((H, S, D), dtype=torch.bfloat16, device=cache.k_packed.device)
    if NP > 0:
        BLOCK_D = 16
        grid = (H, NP, _grid_blocks(D, BLOCK_D))
        k_dequant_kernel[grid](
            cache.k_packed, cache.k_scale, out,
            H, S, D, NP,
            out.stride(0), out.stride(1), out.stride(2),
            BLOCK_D=BLOCK_D,
        )
    n_full = NP * BLOCK
    if n_full < S:
        out[:, n_full:] = cache.k_hot.to(torch.bfloat16)
    return out


def _dequant_v(cache: KIVICache) -> Tensor:
    H, S, D = cache.H, cache.S, cache.D
    ND = cache.v_packed.shape[2]
    out = torch.empty((H, S, D), dtype=torch.bfloat16, device=cache.v_packed.device)
    BLOCK_T = 16
    grid = (H, _grid_blocks(S, BLOCK_T), ND)
    v_dequant_kernel[grid](
        cache.v_packed, cache.v_scale, out,
        H, S, D, ND,
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_T=BLOCK_T,
    )
    return out


def dequant_kivi(cache: KIVICache) -> tuple[Tensor, Tensor]:
    """Dequantize a KIVICache back to (k_bf16, v_bf16), both [H, S, D]."""
    return _dequant_k(cache), _dequant_v(cache)
