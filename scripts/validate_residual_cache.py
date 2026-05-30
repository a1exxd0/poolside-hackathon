"""Correctness check for the bf16 residual-window int4 cache (no model load).

Exercises `_Int4KiviLayer` (residual=0 base) vs `_ResidualInt4KiviLayer` directly
on random GPU tensors, simulating prefill + incremental decode `update` calls:

  1. residual=0 reconstruction is BIT-IDENTICAL to the base layer.
  2. residual=R keeps the trailing R tokens of K and V EXACT bf16 (zero error),
     while older tokens carry the usual int4 quant error.
  3. memory grows by ~R*H*D*2 bytes/K,V vs residual=0 (bounded, seq-independent).
"""
import sys

WORKTREE = "/home/alex/poolside-hackathon-kv-quant/.claude/worktrees/kv-quant-long-context"
sys.path.insert(0, WORKTREE)

import torch

from int4_kivi.hf_cache import _Int4KiviLayer
from int4_kivi.residual_hf_cache import _ResidualInt4KiviLayer

torch.manual_seed(0)
dev = "cuda"
H, D = 4, 64
PREFILL = 500
DECODE = 40
RESIDUAL = 128


def feed(layer, k_all, v_all):
    """Replay prefill (one update) + DECODE single-token updates; return last full."""
    out_k = out_v = None
    # prefill
    out_k, out_v = layer.update(k_all[:, :, :PREFILL], v_all[:, :, :PREFILL])
    for t in range(PREFILL, PREFILL + DECODE):
        out_k, out_v = layer.update(k_all[:, :, t:t + 1], v_all[:, :, t:t + 1])
    return out_k, out_v


S = PREFILL + DECODE
k_all = torch.randn(1, H, S, D, dtype=torch.bfloat16, device=dev)
v_all = torch.randn(1, H, S, D, dtype=torch.bfloat16, device=dev)

base = _Int4KiviLayer()
res0 = _ResidualInt4KiviLayer(residual=0)
resR = _ResidualInt4KiviLayer(residual=RESIDUAL)

bk, bv = feed(base, k_all, v_all)
zk, zv = feed(res0, k_all, v_all)
rk, rv = feed(resR, k_all, v_all)

# (1) residual=0 == base, bit-identical
d0k = (bk - zk).abs().max().item()
d0v = (bv - zv).abs().max().item()
print(f"[1] residual=0 vs base   max|dK|={d0k:.2e}  max|dV|={d0v:.2e}")
assert d0k == 0.0 and d0v == 0.0, "residual=0 must reproduce base bit-exactly"

# (2) residual=R keeps trailing R tokens exact; older tokens quantized
true_k = k_all[0].transpose(0, 1) if False else k_all[0]  # [H,S,D]
true_v = v_all[0]
rk0, rv0 = rk[0], rv[0]  # [H,S,D]
tail_err_k = (rk0[:, -RESIDUAL:] - true_k[:, -RESIDUAL:]).abs().max().item()
tail_err_v = (rv0[:, -RESIDUAL:] - true_v[:, -RESIDUAL:]).abs().max().item()
head_err_k = (rk0[:, :PREFILL - RESIDUAL] - true_k[:, :PREFILL - RESIDUAL]).abs().max().item()
print(f"[2] residual={RESIDUAL}: trailing-window  max|dK|={tail_err_k:.2e} "
      f"max|dV|={tail_err_v:.2e}  (expect 0)")
print(f"    older (quantized) tokens max|dK|={head_err_k:.2e}  (expect >0)")
assert tail_err_k == 0.0 and tail_err_v == 0.0, "residual window must be exact bf16"
assert head_err_k > 0.0, "older tokens should be int4-quantized (nonzero error)"

# base path quantizes everything but the <16 partial tail -> recent tokens lossy
base_recent_err = (bk[0][:, -RESIDUAL:] - true_k[:, -RESIDUAL:]).abs().max().item()
print(f"    base path recent-token max|dK|={base_recent_err:.2e}  (lossy, >0)")

# (3) memory: residual costs a bounded extra bf16 window
mb = base.nbytes()
mr = resR.nbytes()
print(f"[3] nbytes base={mb}  residual={mr}  extra={mr - mb} "
      f"(~{2 * RESIDUAL * H * D * 2} expected for K+V bf16)")
assert mr > mb

print("RESIDUAL CACHE VALIDATION: OK")
