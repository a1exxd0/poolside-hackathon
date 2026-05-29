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
