"""Decisive kernel diagnostic: does vLLM per-channel-K actually isolate
per-channel outliers (KIVI's whole premise), and does it match an exact torch
reference?

The existing kernel test uses torch.randn (iid N(0,1)) which has NO per-channel
structure, so per-channel and per-token K give the SAME error there — it cannot
detect a per-channel regression.  Here we build K WITH per-channel outliers (a
few channels persistently ~6x larger, like real attention K) and compare:

  * vLLM per-channel-K  (full 16-token block -> _store_k_channel_kernel)
  * torch per-channel-K reference (exact mirror of the kernel math)
  * torch per-token-K   reference (what we'd get WITHOUT the KIVI layout)

If per-channel works:  vLLM-per-channel ~= torch-per-channel  <<  torch-per-token.
If per-channel is silently behaving like per-token:  all three ~equal.
"""
import torch
from vllm.v1.attention.ops.triton_int4_kivi import (
    int4_kivi_store, int4_kivi_gather_dequant, BLOCK, QMAX,
)
from vllm.utils.torch_utils import int4_kivi_kv_cache_full_dim

torch.manual_seed(0)
dev = "cuda"
H, D = 8, 128
block_size = 16
L = 16  # exactly one full block -> per-channel K path fires


def fp8(s):  # round a scale to e4m3 exactly like the kernel stores it
    return s.to(torch.float8_e4m3fn).float()


def mse_scale(x, axis):
    """MSE-optimal clip scale, reduced over `axis`. Mirrors the kernel's 16-pt
    grid (alpha in [0.5,1.0]) with fp8-rounded scales and strict-< argmin."""
    amax = x.abs().amax(axis, keepdim=True).clamp_min(1e-9)
    best_err = torch.full_like(amax, 1e38)
    best_s = fp8(amax / QMAX)
    for i in range(16):
        a = 0.5 + i * (0.5 / 15)
        s = fp8(a * amax / QMAX)
        code = torch.round(x / s).clamp(-QMAX, QMAX)
        err = ((x - code * s) ** 2).sum(axis, keepdim=True)
        take = err < best_err
        best_err = torch.where(take, err, best_err)
        best_s = torch.where(take, s, best_s)
    return best_s


def rt(x, scale):
    return torch.round(x / scale).clamp(-QMAX, QMAX) * scale


def relrmse(a, b):
    return (a - b).pow(2).mean().sqrt() / b.pow(2).mean().sqrt()


# ---- build K with persistent per-channel outlier structure -----------------
base = torch.randn(L, H, D, device=dev)
chan_mag = (torch.rand(H, D, device=dev) ** 4) * 8.0 + 0.3   # heavy-tailed
outlier = torch.rand(H, D, device=dev) < 0.06
chan_mag = torch.where(outlier, chan_mag * 6.0, chan_mag)     # strong outliers
key = (base * chan_mag[None]).bfloat16()                      # [L,H,D]
value = torch.randn(L, H, D, device=dev).bfloat16()
key_f = key.float()

# ---- vLLM kernel: store (per-channel K for the full block) + dequant -------
full_dim = int4_kivi_kv_cache_full_dim(D)
kv_cache = torch.zeros((4, 2, block_size, H, full_dim), dtype=torch.uint8, device=dev)
block_table = torch.zeros((1, 4), dtype=torch.int32, device=dev)  # req0 -> block 0
seq_lens = torch.tensor([L], dtype=torch.int32, device=dev)
slot_mapping = torch.arange(L, dtype=torch.int64, device=dev)     # block 0, slots 0..15
int4_kivi_store(key, value, kv_cache, slot_mapping, D)
k_out, _ = int4_kivi_gather_dequant(kv_cache, block_table, seq_lens, D, H, L)
k_vllm = k_out[0, :, :L, :].permute(1, 0, 2).float()             # [L,H,D]

# ---- torch per-channel reference (scale per channel over the 16 tokens) -----
s_pc = mse_scale(key_f, axis=0)                                  # [1,H,D]
k_ref_pc = rt(key_f, s_pc)

# ---- torch per-token reference (scale per token over each 16-elem block) -----
xt = key_f.reshape(L, H, D // BLOCK, BLOCK)
s_pt = mse_scale(xt, axis=3)                                     # [L,H,ND,1]
k_ref_pt = rt(xt, s_pt).reshape(L, H, D)

print("=== per-channel-K diagnostic (K has injected per-channel outliers) ===")
print(f"  vLLM  per-channel  relRMSE vs orig : {relrmse(k_vllm,   key_f):.4f}")
print(f"  torch per-channel  relRMSE vs orig : {relrmse(k_ref_pc, key_f):.4f}")
print(f"  torch per-token    relRMSE vs orig : {relrmse(k_ref_pt, key_f):.4f}")
print(f"  vLLM vs torch per-channel (agree)  : {relrmse(k_vllm, k_ref_pc):.4f}")
ratio = relrmse(k_ref_pt, key_f) / relrmse(k_ref_pc, key_f)
print(f"  per-token / per-channel error ratio: {ratio:.2f}x  "
      f"(>1 => per-channel genuinely helps on this data)")

ok_match = relrmse(k_vllm, k_ref_pc) < 0.05
ok_helps = relrmse(k_vllm, key_f) < 0.8 * relrmse(k_ref_pt, key_f)
print()
print(f"  [{'PASS' if ok_match else 'FAIL'}] vLLM per-channel matches torch reference")
print(f"  [{'PASS' if ok_helps else 'FAIL'}] vLLM per-channel beats per-token by >20%")
