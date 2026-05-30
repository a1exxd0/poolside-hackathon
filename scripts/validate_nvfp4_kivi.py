"""Standalone correctness + accuracy A/B for the NVFP4-KIVI kernels.

Imports the two kernel modules *directly by file path* (only torch + triton
needed -- no full vLLM import), so it runs fast and in isolation.

It checks two things:

1. CORRECTNESS of the NVFP4 fused paged decode: build a packed cache, store K/V
   as NVFP4, then compare ``nvfp4_kivi_paged_decode`` against a dense reference
   (SDPA over the gather-dequantized K/V).  The reference uses the SAME quantized
   cache, so this isolates the fused kernel's math from quant error -> must match
   to ~1e-2 (bf16 tensor-core tolerance).

2. ACCURACY A/B (the actual question): for several synthetic K/V distributions
   (Gaussian, and per-channel-outlier which mimics real K), store with BOTH the
   INT4-KIVI and NVFP4-KIVI kernels, gather-dequant, and report reconstruction
   error vs the original bf16.  Same paged layout, same MSE alpha-clip, same
   per-channel-K / per-token-V geometry -- the ONLY difference is the 4-bit grid
   (uniform int4 vs non-uniform E2M1), so the delta is attributable to the format.

Run (under the vLLM venv, from anywhere):
    CUDA_HOME=/usr/local/cuda-12.8 \
      /home/alex/poolside-hackathon-kv-quant/.venv-vllm/bin/python \
      <this file>
"""
import importlib.util
import math
import os

import torch
import torch.nn.functional as F

ROOT = "/home/alex/poolside-hackathon-kv-quant/.claude/worktrees/kv-quant-nvfp4"
OPS = f"{ROOT}/vllm/vllm/v1/attention/ops"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INT4 = _load("triton_int4_kivi", f"{OPS}/triton_int4_kivi.py")
NV = _load("triton_nvfp4_kivi", f"{OPS}/triton_nvfp4_kivi.py")

DEV = "cuda"
torch.manual_seed(0)


def full_dim(head_size: int) -> int:
    return head_size // 2 + head_size // 16


def make_cache(num_blocks, block_size, H, D):
    return torch.zeros(
        (num_blocks, 2, block_size, H, full_dim(D)), dtype=torch.uint8, device=DEV
    )


def store_and_gather(mod, prefix, k, v, block_size):
    """k,v: [N,H,D] bf16. Returns dense (k_hat, v_hat) [H, N, D] bf16."""
    N, H, D = k.shape
    num_blocks = (N + block_size - 1) // block_size + 1
    cache = make_cache(num_blocks, block_size, H, D)
    slot = torch.arange(N, device=DEV, dtype=torch.int64)
    store = getattr(mod, f"{prefix}_kivi_store")
    store(k, v, cache, slot, D)
    # one request, full context
    block_table = (
        torch.arange(num_blocks, device=DEV, dtype=torch.int32)
        .view(1, num_blocks)
    )
    seq_lens = torch.tensor([N], device=DEV, dtype=torch.int32)
    gather = getattr(mod, f"{prefix}_kivi_gather_dequant")
    k_hat, v_hat = gather(cache, block_table, seq_lens, D, H, N)
    return cache, k_hat[0], v_hat[0]  # [H, N, D]


def rel_err(orig, hat):
    o = orig.float()
    h = hat.float()
    num = (o - h).pow(2).sum().sqrt()
    den = o.pow(2).sum().sqrt().clamp_min(1e-12)
    return (num / den).item()


def gen(dist, N, H, D):
    if dist == "gaussian":
        return torch.randn(N, H, D, device=DEV, dtype=torch.bfloat16)
    if dist == "outlier-channel":
        # ~3% of channels carry 10x-larger values (mimics K's persistent
        # per-channel outliers -- the regime KIVI per-channel-K targets).
        x = torch.randn(N, H, D, device=DEV)
        nout = max(1, int(0.03 * D))
        idx = torch.randperm(D, device=DEV)[:nout]
        x[:, :, idx] *= 10.0
        return x.to(torch.bfloat16)
    if dist == "heavy-tail":
        x = torch.randn(N, H, D, device=DEV)
        x = x.sign() * x.abs().pow(1.7)  # leptokurtic
        return x.to(torch.bfloat16)
    raise ValueError(dist)


def accuracy_ab():
    print("=== Accuracy A/B: reconstruction rel-error (lower is better) ===")
    H, D, block_size = 8, 128, 16
    N = 512  # all full blocks (K -> per-channel)
    hdr = f"{'dist':>16} | {'int4 K':>9} {'nvfp4 K':>9} | {'int4 V':>9} {'nvfp4 V':>9}"
    print(hdr)
    print("-" * len(hdr))
    for dist in ("gaussian", "heavy-tail", "outlier-channel"):
        k = gen(dist, N, H, D)
        v = gen(dist, N, H, D)
        _, k_i, v_i = store_and_gather(INT4, "int4", k, v, block_size)
        _, k_n, v_n = store_and_gather(NV, "nvfp4", k, v, block_size)
        ko = k.transpose(0, 1)  # [H, N, D]
        vo = v.transpose(0, 1)
        print(
            f"{dist:>16} | {rel_err(ko, k_i):>9.4f} {rel_err(ko, k_n):>9.4f} | "
            f"{rel_err(vo, v_i):>9.4f} {rel_err(vo, v_n):>9.4f}"
        )


def decode_correctness():
    print("\n=== NVFP4 fused decode vs dense reference (kernel correctness) ===")
    H, Hq, D, block_size = 8, 48, 128, 16
    GROUP = Hq // H
    for N in (512, 500, 33):  # full-only, partial tail, short
        k = torch.randn(N, H, D, device=DEV, dtype=torch.bfloat16)
        v = torch.randn(N, H, D, device=DEV, dtype=torch.bfloat16)
        num_blocks = (N + block_size - 1) // block_size + 1
        cache = make_cache(num_blocks, block_size, H, D)
        slot = torch.arange(N, device=DEV, dtype=torch.int64)
        NV.nvfp4_kivi_store(k, v, cache, slot, D)

        block_table = torch.arange(
            num_blocks, device=DEV, dtype=torch.int32
        ).view(1, num_blocks)
        seq_lens = torch.tensor([N], device=DEV, dtype=torch.int32)
        q = torch.randn(1, Hq, D, device=DEV, dtype=torch.bfloat16)
        sm = 1.0 / math.sqrt(D)

        out = NV.nvfp4_kivi_paged_decode(q, cache, block_table, seq_lens, sm)[0]

        # reference: dense SDPA over the SAME quantized cache (gather-dequant).
        k_hat, v_hat = NV.nvfp4_kivi_gather_dequant(
            cache, block_table, seq_lens, D, H, N
        )
        k_d = k_hat[0]  # [H, N, D]
        v_d = v_hat[0]
        qg = q[0].view(Hq, 1, D).transpose(0, 1)  # [1, Hq, D] -> per-head
        # expand kv heads to query heads (GQA)
        kk = k_d.repeat_interleave(GROUP, dim=0)  # [Hq, N, D]
        vv = v_d.repeat_interleave(GROUP, dim=0)
        ref = F.scaled_dot_product_attention(
            q[0].unsqueeze(1).float(),  # [Hq,1,D]
            kk.float(),                 # [Hq,N,D]
            vv.float(),
            scale=sm,
        ).squeeze(1)  # [Hq, D]
        md = (out.float() - ref).abs().max().item()
        re = rel_err(ref, out)
        ok = "OK" if md < 5e-2 else "FAIL"
        print(f"  N={N:>4}  max|d|={md:.4e}  rel={re:.4e}  [{ok}]")


if __name__ == "__main__":
    print(f"torch {torch.__version__}  device {torch.cuda.get_device_name()}")
    decode_correctness()
    accuracy_ab()
    print("\nNVFP4_KIVI VALIDATE DONE")
