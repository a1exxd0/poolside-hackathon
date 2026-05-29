"""Page-aligned 4-bit KV cache with MSE-optimal blockwise scaling."""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from torch import Tensor

BLOCK = 16   # elements per quant block (matches NVFP4 1×16 along head_dim)
PAGE = 16    # tokens per page (matches vLLM default)
QMAX = 7     # symmetric INT4 range [-7, 7]


def _absmax_scale(x: Tensor) -> Tensor:
    """Per-block scale: absmax / QMAX. Input [..., BLOCK], returns [..., 1]."""
    return x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9) / QMAX


def _mse_optimal_scale(x: Tensor, n_alphas: int = 32) -> Tensor:
    """Grid-search clip ratio to minimise reconstruction MSE. Input [..., BLOCK], returns [..., 1]."""
    xf = x.float()
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)   # [..., 1]
    alphas = torch.linspace(0.5, 1.0, n_alphas, device=x.device, dtype=torch.float32)
    alphas_view = alphas.view((n_alphas,) + (1,) * xf.dim())
    scales = alphas_view * absmax.unsqueeze(0) / QMAX               # [n_alphas, ..., 1]
    xf_exp = xf.unsqueeze(0)
    q = (xf_exp / scales).round().clamp(-QMAX, QMAX)
    mse = ((xf_exp - q * scales) ** 2).mean(dim=-1, keepdim=True)  # [n_alphas, ..., 1]
    best = mse.argmin(dim=0)                                        # [..., 1]
    s_flat = scales.reshape(n_alphas, -1).T                         # [num_blocks, n_alphas]
    opt_flat = s_flat.gather(1, best.reshape(-1, 1))                # [num_blocks, 1]
    return opt_flat.reshape(*x.shape[:-1], 1).to(x.dtype)


def quantize_block(x: Tensor, scale: Tensor) -> Tensor:
    return (x / scale).round().clamp(-QMAX, QMAX).to(torch.int8)


def dequantize_block(q: Tensor, scale: Tensor) -> Tensor:
    return q.to(scale.dtype) * scale


@dataclasses.dataclass
class QuantizedPage:
    k_int4: Tensor      # int8  [n_heads, page_tokens, head_dim]
    v_int4: Tensor
    k_scale: Tensor     # fp16  [n_heads, page_tokens, head_dim // BLOCK]
    v_scale: Tensor
    start_pos: int

    def dequantize(self) -> tuple[Tensor, Tensor]:
        def _deq(q: Tensor, scale: Tensor) -> Tensor:
            sf = scale.unsqueeze(-1).expand(*scale.shape, BLOCK).reshape(*q.shape)
            return dequantize_block(q, sf).to(torch.bfloat16)
        return _deq(self.k_int4, self.k_scale), _deq(self.v_int4, self.v_scale)

    def mem_bytes(self) -> int:
        # INT4 packed at 0.5 B/elem; FP16 scales at 2 B/elem; k+v both
        return 2 * (self.k_int4.numel() // 2 + self.k_scale.numel() * 2)

    def bf16_bytes(self) -> int:
        return self.k_int4.numel() * 2 * 2   # 2 bytes × k+v


def quantize_page(
    k: Tensor, v: Tensor, start_pos: int, use_mse: bool = True, n_alphas: int = 32,
) -> QuantizedPage:
    """Quantize one page. k, v: [n_heads, page_tokens, head_dim]."""
    n_heads, page_tokens, head_dim = k.shape
    kf = k.float().reshape(n_heads, page_tokens, head_dim // BLOCK, BLOCK)
    vf = v.float().reshape(n_heads, page_tokens, head_dim // BLOCK, BLOCK)
    scale_fn = _mse_optimal_scale if use_mse else _absmax_scale
    ks = scale_fn(kf, n_alphas) if use_mse else scale_fn(kf)  # type: ignore[call-arg]
    vs = scale_fn(vf, n_alphas) if use_mse else scale_fn(vf)
    k_q = quantize_block(kf, ks).reshape(n_heads, page_tokens, head_dim)
    v_q = quantize_block(vf, vs).reshape(n_heads, page_tokens, head_dim)
    return QuantizedPage(
        k_q, v_q,
        ks.squeeze(-1).to(torch.float16),
        vs.squeeze(-1).to(torch.float16),
        start_pos,
    )


class QuantizedKVLayer:
    def __init__(self, page_size: int = PAGE, use_mse: bool = True, n_alphas: int = 32) -> None:
        self.page_size = page_size
        self.use_mse = use_mse
        self.n_alphas = n_alphas
        self.pages: list[QuantizedPage] = []
        self.hot_k: Optional[Tensor] = None
        self.hot_v: Optional[Tensor] = None

    def append(self, k: Tensor, v: Tensor) -> None:
        """k, v: [n_kv_heads, new_tokens, head_dim]."""
        self.hot_k = torch.cat([self.hot_k, k], dim=1) if self.hot_k is not None else k
        self.hot_v = torch.cat([self.hot_v, v], dim=1) if self.hot_v is not None else v
        while self.hot_k.shape[1] >= self.page_size:
            start = len(self.pages) * self.page_size
            self.pages.append(
                quantize_page(
                    self.hot_k[:, :self.page_size],
                    self.hot_v[:, :self.page_size],
                    start, self.use_mse, self.n_alphas,
                )
            )
            self.hot_k = self.hot_k[:, self.page_size:]
            self.hot_v = self.hot_v[:, self.page_size:]

    def get_kv(self) -> tuple[Tensor, Tensor]:
        if not self.pages:
            if self.hot_k is None:
                raise ValueError("Cache is empty")
            return self.hot_k.bfloat16(), self.hot_v.bfloat16()   # type: ignore[union-attr]
        parts_k, parts_v = zip(*(p.dequantize() for p in self.pages))
        k = torch.cat(list(parts_k), dim=1)
        v = torch.cat(list(parts_v), dim=1)
        if self.hot_k is not None and self.hot_k.shape[1] > 0:
            k = torch.cat([k, self.hot_k.bfloat16()], dim=1)
            v = torch.cat([v, self.hot_v.bfloat16()], dim=1)      # type: ignore[union-attr]
        return k, v

    def seq_len(self) -> int:
        hot = self.hot_k.shape[1] if self.hot_k is not None else 0
        return len(self.pages) * self.page_size + hot

    def mem_bytes(self) -> int:
        hot_cost = self.hot_k.numel() * 2 * 2 if self.hot_k is not None else 0
        return sum(p.mem_bytes() for p in self.pages) + hot_cost


class QuantizedKVCache:
    """Drop-in replacement for DynamicCache in generate.py layer-access pattern."""

    def __init__(self, page_size: int = PAGE, use_mse: bool = True, n_alphas: int = 32) -> None:
        self.page_size = page_size
        self.use_mse = use_mse
        self.n_alphas = n_alphas
        self.layers: list[QuantizedKVLayer] = []

    def _ensure(self, layer_idx: int) -> None:
        while len(self.layers) <= layer_idx:
            self.layers.append(QuantizedKVLayer(self.page_size, self.use_mse, self.n_alphas))

    def update(self, layer_idx: int, k: Tensor, v: Tensor) -> None:
        """k, v: [batch, n_kv_heads, new_tokens, head_dim]."""
        self._ensure(layer_idx)
        self.layers[layer_idx].append(k.squeeze(0), v.squeeze(0))

    def get_kv(self, layer_idx: int) -> tuple[Tensor, Tensor]:
        """Returns [1, n_kv_heads, seq_len, head_dim] bfloat16."""
        k, v = self.layers[layer_idx].get_kv()
        return k.unsqueeze(0), v.unsqueeze(0)

    def mem_bytes(self) -> int:
        return sum(l.mem_bytes() for l in self.layers)

    def bf16_bytes(self) -> int:
        total = 0
        for layer in self.layers:
            total += sum(p.bf16_bytes() for p in layer.pages)
            if layer.hot_k is not None:
                total += layer.hot_k.numel() * 2 * 2
        return total

    def compression_ratio(self) -> float:
        return self.bf16_bytes() / max(self.mem_bytes(), 1)


def measure_page_error(k: Tensor, page_size: int = PAGE, n_alphas: int = 32) -> dict:
    """Absmax vs MSE-optimal reconstruction error over complete pages.

    Args:
        k: [n_kv_heads, seq_len, head_dim]
    """
    n_heads, seq_len, head_dim = k.shape
    abs_errs, opt_errs = [], []
    for i in range(seq_len // page_size):
        page = k[:, i * page_size:(i + 1) * page_size].float()
        pg = page.reshape(n_heads, page_size, head_dim // BLOCK, BLOCK)
        for scale_fn, errs in ((_absmax_scale, abs_errs), (_mse_optimal_scale, opt_errs)):
            s = scale_fn(pg, n_alphas) if scale_fn is _mse_optimal_scale else scale_fn(pg)  # type: ignore[call-arg]
            mse = ((pg - dequantize_block(quantize_block(pg, s), s)) ** 2).mean().item()
            errs.append(mse)
    absmax_mse = float(sum(abs_errs) / max(len(abs_errs), 1))
    optimal_mse = float(sum(opt_errs) / max(len(opt_errs), 1))
    return {
        "absmax_mse": absmax_mse,
        "optimal_mse": optimal_mse,
        "reduction_pct": 100.0 * (absmax_mse - optimal_mse) / max(absmax_mse, 1e-12),
    }


# ════════════════════════════════════════════════════════════════════════════
# INT4-vs-NVFP4 sweep:  {format} × {layout} × {calibration}
#
# Every cell costs the same memory (4-bit data + one scale per 16 elements =
# 0.5625 B/elem), so this isolates reconstruction *quality*. Scales are kept in
# fp32 here so neither format gets a scale-precision edge; realistic 1-byte
# block scales would add the same small penalty to every cell.
# ════════════════════════════════════════════════════════════════════════════

# e2m1 (NVFP4) representable magnitudes, and midpoints used to round onto them.
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_BOUNDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def _q_int4(y: Tensor) -> Tensor:
    """Round scale-normalised values onto the symmetric INT4 grid [-7, 7]."""
    return y.round().clamp(-7.0, 7.0)


def _q_int3(y: Tensor) -> Tensor:
    """Round scale-normalised values onto the symmetric INT3 grid [-3, 3]."""
    return y.round().clamp(-3.0, 3.0)


def _q_e2m1(y: Tensor) -> Tensor:
    """Round scale-normalised values onto the e2m1 (NVFP4) grid, max magnitude 6."""
    levels = _E2M1_LEVELS.to(y.device, y.dtype)
    bounds = _E2M1_BOUNDS.to(y.device, y.dtype)
    idx = torch.bucketize(y.abs(), bounds, right=False)
    return torch.sign(y) * levels[idx]


# format -> (qmax used to set the scale, rounding fn onto the grid)
_FORMATS = {"int4": (7.0, _q_int4), "int3": (3.0, _q_int3), "nvfp4": (6.0, _q_e2m1)}


def _to_blocks(x: Tensor, layout: str) -> Tensor:
    """x [H, S, D] -> blocks [..., BLOCK].

    'headdim' groups 16 channels of one token (what the NVFP4 kernel does);
    'channel' groups 16 tokens of one channel (per-channel, KIVI-style).
    'channel' requires S % BLOCK == 0.
    """
    H, S, D = x.shape
    if layout == "headdim":
        return x.reshape(H, S, D // BLOCK, BLOCK)
    if layout == "channel":
        return x.transpose(1, 2).contiguous().reshape(H, D, S // BLOCK, BLOCK)
    raise ValueError(f"unknown layout {layout!r}")


def _calibrate(xb: Tensor, qmax: float, qfn, calib: str, n_alphas: int = 32) -> Tensor:
    """Per-block scale for blocks xb [..., BLOCK]. Returns [..., 1].

    'mse' grid-searches the clip ratio alpha in [0.5, 1.0]; alpha=1.0 reproduces
    absmax, so mse is <= absmax by construction.
    """
    xf = xb.float()
    absmax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    if calib == "absmax":
        return absmax / qmax
    if calib != "mse":
        raise ValueError(f"unknown calib {calib!r}")
    alphas = torch.linspace(0.5, 1.0, n_alphas, device=xb.device).view(
        (n_alphas,) + (1,) * xf.dim())
    scales = alphas * absmax.unsqueeze(0) / qmax              # [n_alphas, ..., 1]
    xexp = xf.unsqueeze(0)
    err = ((xexp - qfn(xexp / scales) * scales) ** 2).mean(dim=-1, keepdim=True)
    best = err.argmin(dim=0)                                  # [..., 1]
    sflat = scales.reshape(n_alphas, -1).T                    # [num_blocks, n_alphas]
    return sflat.gather(1, best.reshape(-1, 1)).reshape(*xb.shape[:-1], 1)


def rmse_cell(x: Tensor, fmt: str, layout: str, calib: str, n_alphas: int = 32) -> float:
    """Reconstruction RMSE for one (format, layout, calib) cell. x: [H, S, D]."""
    qmax, qfn = _FORMATS[fmt]
    xb = _to_blocks(x, layout).float()
    scale = _calibrate(xb, qmax, qfn, calib, n_alphas)
    xhat = qfn(xb / scale) * scale
    return ((xb - xhat) ** 2).mean().sqrt().item()


SWEEP_CELLS = [
    (fmt, layout, calib)
    for fmt in ("nvfp4", "int4")
    for layout in ("headdim", "channel")
    for calib in ("absmax", "mse")
]
# What vLLM's NVFP4 KV kernel ships today: NVFP4, head_dim blocks, absmax scale.
SWEEP_BASELINE = ("nvfp4", "headdim", "absmax")


def sweep_tensor(x: Tensor, n_alphas: int = 32) -> dict:
    """Full {format}×{layout}×{calib} grid on x [H, S, D]; RMSE per cell.

    Crops the sequence to a multiple of BLOCK so head_dim and channel layouts
    are scored on identical data.
    """
    S = x.shape[1]
    x = x[:, : (S // BLOCK) * BLOCK]
    return {cell: rmse_cell(x, *cell, n_alphas=n_alphas) for cell in SWEEP_CELLS}


def roundtrip(x: Tensor, fmt: str, layout: str, calib: str, n_alphas: int = 32) -> Tensor:
    """Quantize+dequantize x [H, S, D] under one scheme; same shape and dtype out.

    'headdim' quantizes every token immediately. 'channel' needs a full 16-token
    page per scale, so the trailing S % BLOCK tokens are returned unquantized —
    the realistic bf16 hot-page residual that per-channel blocking always carries.
    """
    qmax, qfn = _FORMATS[fmt]
    H, S, D = x.shape
    if layout == "headdim":
        xb = _to_blocks(x, "headdim").float()
        scale = _calibrate(xb, qmax, qfn, calib, n_alphas)
        return (qfn(xb / scale) * scale).reshape(H, S, D).to(x.dtype)
    if layout == "channel":
        n_full = (S // BLOCK) * BLOCK
        if n_full == 0:
            return x
        xb = _to_blocks(x[:, :n_full], "channel").float()     # [H, D, n_full//B, B]
        scale = _calibrate(xb, qmax, qfn, calib, n_alphas)
        head = (qfn(xb / scale) * scale).reshape(H, D, n_full).transpose(1, 2)
        head = head.contiguous().to(x.dtype)
        return torch.cat([head, x[:, n_full:]], dim=1) if n_full < S else head
    raise ValueError(f"unknown layout {layout!r}")
