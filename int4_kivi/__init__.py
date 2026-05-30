"""int4_kivi: software INT4-KIVI KV-cache path (Triton).

Validated scheme (PROBLEM.md): K = INT4 per-channel (16-token page blocks),
V = INT4 per-token (16-channel head_dim blocks), symmetric INT4 [-7,7] with
fp16 blockwise scales.  Numerically faithful to kv_quant.roundtrip.

Public API:
    store_kivi(k, v, k_calib="mse", v_calib="mse") -> KIVICache
    dequant_kivi(cache) -> (k_bf16, v_bf16)
    KIVICache  (holds packed tensors + scales + shapes; .nbytes,
                .compression_ratio_vs_bf16())
"""

from .cache import KIVICache, store_kivi, dequant_kivi
from .triton_kernels import BLOCK, QMAX, N_ALPHAS, PACK

__all__ = [
    "KIVICache",
    "store_kivi",
    "dequant_kivi",
    "BLOCK",
    "QMAX",
    "N_ALPHAS",
    "PACK",
]
