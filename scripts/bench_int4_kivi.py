"""Backup deliverable: raw store_kivi/dequant_kivi throughput + memory math.

Measures, on the B300, at Laguna-XS.2 KV shapes (H=8 KV heads, D=128 head_dim):
  1. store_kivi (quantize) and dequant_kivi (reconstruct) throughput in GB/s and
     latency, where "bytes moved" is the dominant traffic (bf16 in/out).
  2. The measured memory-compression ratio: fp16-scale actual (~3.2x) AND the
     1-byte/e4m3-scale projection (~3.56x, matching NVFP4 / PROBLEM.md).
  3. A capacity->throughput argument: max sequence length / batch whose KV cache
     fits in HBM for INT4-KIVI vs BF16 for the full Laguna model (40 layers,
     8 KV heads, 128 head_dim), and the decode-throughput headroom that unlocks.

Run:
  .venv/bin/python scripts/bench_int4_kivi.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from int4_kivi import store_kivi, dequant_kivi  # noqa: E402

DEV = "cuda"
H, D = 8, 128            # Laguna-XS.2 KV heads, head_dim
N_LAYERS = 40           # full Laguna stack
SEQS = [512, 2048, 8192, 32768]


def _sync():
    torch.cuda.synchronize()


def _time_ms(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    _sync()
    return start.elapsed_time(end) / iters


def bench_store_dequant():
    print("=" * 88)
    print("RAW store_kivi / dequant_kivi THROUGHPUT  (H=8, D=128, B300)")
    print("=" * 88)
    hdr = f"{'S':>7} | {'store ms':>9} {'store GB/s':>11} | {'deq ms':>8} {'deq GB/s':>10} | {'ratio':>6} {'ratio_1B':>8}"
    print(hdr)
    print("-" * len(hdr))
    g = torch.Generator(device=DEV).manual_seed(0)
    for S in SEQS:
        k = (torch.randn(H, S, D, generator=g, device=DEV) ** 3).to(torch.bfloat16)
        v = torch.randn(H, S, D, generator=g, device=DEV).to(torch.bfloat16)

        store_ms = _time_ms(lambda: store_kivi(k, v))
        cache = store_kivi(k, v)
        deq_ms = _time_ms(lambda: dequant_kivi(cache))

        # bytes moved:
        #  store reads 2 bf16 tensors (k,v) and writes ~0.5B/elem int4 + scales.
        #  dequant reads packed int4 + scales and writes 2 bf16 tensors.
        bf16_bytes = 2 * H * S * D * 2  # k + v in bf16
        store_bytes = bf16_bytes + cache.nbytes          # read bf16, write packed
        deq_bytes = cache.nbytes + bf16_bytes            # read packed, write bf16
        store_gbs = store_bytes / (store_ms * 1e-3) / 1e9
        deq_gbs = deq_bytes / (deq_ms * 1e-3) / 1e9

        ratio = cache.compression_ratio_vs_bf16()
        data = cache.k_packed.numel() + cache.v_packed.numel()
        n_scales = cache.k_scale.numel() + cache.v_scale.numel()
        hot = cache.k_hot.numel() * 2
        ratio_1b = cache.bf16_nbytes() / (data + n_scales * 1 + hot)

        print(f"{S:>7} | {store_ms:>9.3f} {store_gbs:>11.1f} | {deq_ms:>8.3f} "
              f"{deq_gbs:>10.1f} | {ratio:>6.3f} {ratio_1b:>8.3f}")
    print()


def capacity_argument():
    print("=" * 88)
    print("MEMORY CAPACITY -> THROUGHPUT HEADROOM  (full Laguna: 40 layers, 8 KV heads, D=128)")
    print("=" * 88)

    total_hbm = torch.cuda.get_device_properties(0).total_memory
    # Reserve headroom for weights/activations; report KV-only frontier too.
    print(f"HBM total: {total_hbm/1e9:.1f} GB")

    # bytes per token of KV cache across the whole model:
    #   bf16:  2 (K,V) * N_LAYERS * H * D * 2 bytes
    #   int4:  ~ (2 * N_LAYERS * H * D * 0.5) data + scales (per 16 block: 2B fp16 / 1B e4m3)
    bf16_per_tok = 2 * N_LAYERS * H * D * 2
    # int4 data: 0.5 B/elem; scales: per 16-elem block one scale.
    # K: 1 scale per (channel, 16 tokens) -> per token = D/16 scales; V: per token D/16 scales.
    int4_data_per_tok = 2 * N_LAYERS * H * D * 0.5
    scales_per_tok = 2 * N_LAYERS * H * (D / 16)  # count of scales per token (K+V)
    int4_fp16_per_tok = int4_data_per_tok + scales_per_tok * 2
    int4_1b_per_tok = int4_data_per_tok + scales_per_tok * 1

    print(f"\nKV bytes / token (whole model):")
    print(f"  bf16            : {bf16_per_tok/1e3:8.2f} KB/token")
    print(f"  int4 (fp16 scl) : {int4_fp16_per_tok/1e3:8.2f} KB/token  "
          f"({bf16_per_tok/int4_fp16_per_tok:.2f}x smaller)")
    print(f"  int4 (1B  scl)  : {int4_1b_per_tok/1e3:8.2f} KB/token  "
          f"({bf16_per_tok/int4_1b_per_tok:.2f}x smaller)")

    for frac, label in [(1.0, "100% HBM (KV-only frontier)"), (0.5, "50% HBM (weights+acts reserved)")]:
        budget = total_hbm * frac
        bf16_tok = budget / bf16_per_tok
        int4_tok = budget / int4_fp16_per_tok
        print(f"\n  Budget = {label} = {budget/1e9:.1f} GB:")
        print(f"    max tokens bf16            : {bf16_tok/1e6:8.3f} M  "
              f"(= batch {int(bf16_tok//32768):>4} @ 32k ctx)")
        print(f"    max tokens int4-KIVI       : {int4_tok/1e6:8.3f} M  "
              f"(= batch {int(int4_tok//32768):>4} @ 32k ctx)")
        print(f"    capacity gain              : {int4_tok/bf16_tok:.2f}x more tokens in-flight")
    print()
    print("Decode is memory-BW bound: throughput scales ~linearly with concurrent")
    print("tokens (batch) up to the BW roof. INT4-KIVI fits ~3.2x more KV in HBM, so")
    print("past the BF16 capacity frontier it sustains ~3.2x the batch -> ~3.2x the")
    print("aggregate decode tokens/s the BF16 cache simply cannot reach (OOM).")
    print()


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB)\n")
    bench_store_dequant()
    capacity_argument()
