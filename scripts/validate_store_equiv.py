"""Golden-equivalence test for int4_kivi_store (guards the per-channel-K path).

validate_paged_decode CANNOT catch store bugs: it compares fused-decode vs
dense-gather, but BOTH read the same cache, so a wrong store agrees with itself.
This test instead pins the *store output*: it quantizes K/V with int4_kivi_store
and dequantizes the cache with int4_kivi_gather_dequant, then compares against a
saved golden (captured from the known-good store).  Any change to the store that
alters the written bytes (e.g. the sync-free full-block detection) must reproduce
the golden bit-exactly.

  cd /tmp && CUDA_HOME=/usr/local/cuda-12.8 .venv-vllm/bin/python \
    .../scripts/validate_store_equiv.py            # compare (or save if missing)
  ... validate_store_equiv.py --save               # force (re)capture golden
"""

from __future__ import annotations

import sys

import torch

from vllm.v1.attention.ops.triton_int4_kivi import (
    int4_kivi_gather_dequant,
    int4_kivi_store,
)

DEV = "cuda"
HK, D = 8, 128
PAGE = 16
FULL_DIM = D // 2 + D // 16
GOLDEN = "/tmp/int4_store_golden.pt"


def store_prefill(L, seed):
    """Store one length-L sequence prefill-style (contiguous slots from a block
    boundary -> full blocks per-channel, trailing partial block per-token)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    nb = (L + PAGE - 1) // PAGE
    kv = torch.zeros((nb + 2, 2, PAGE, HK, FULL_DIM), dtype=torch.uint8, device=DEV)
    bt = torch.zeros((1, nb), dtype=torch.int32, device=DEV)
    phys = list(range(1, 1 + nb))
    for j, p in enumerate(phys):
        bt[0, j] = p
    k = torch.randn(L, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(L, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
    slots = torch.tensor(
        [phys[t // PAGE] * PAGE + (t % PAGE) for t in range(L)],
        dtype=torch.int64, device=DEV,
    )
    int4_kivi_store(k, v, kv, slots, D)
    sl = torch.tensor([L], dtype=torch.int32, device=DEV)
    kd, vd = int4_kivi_gather_dequant(kv, bt, sl, D, HK, L)
    return kd.float().cpu(), vd.float().cpu()


def store_scattered(B, seed):
    """Decode-like store: B tokens at scattered partial-block slots (token 3 of
    B distinct blocks) -> no full blocks, all per-token K (exercises the masks)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    kv = torch.zeros((B + 2, 2, PAGE, HK, FULL_DIM), dtype=torch.uint8, device=DEV)
    k = torch.randn(B, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, HK, D, generator=g, device=DEV, dtype=torch.bfloat16)
    slots = (torch.arange(B, dtype=torch.int64, device=DEV) + 1) * PAGE + 3
    int4_kivi_store(k, v, kv, slots, D)
    return kv.clone().cpu()


CASES = {
    "prefill L=512": ("p", 512),
    "prefill L=500": ("p", 500),
    "prefill L=257": ("p", 257),
    "prefill L=33": ("p", 33),
    "prefill L=16": ("p", 16),
    "prefill L=15": ("p", 15),
    "scattered B=16": ("s", 16),
    "scattered B=40": ("s", 40),
}


def run():
    out = {}
    for i, (name, (kind, n)) in enumerate(CASES.items()):
        seed = 1234 + i  # deterministic across processes (hash() is not)
        if kind == "p":
            kd, vd = store_prefill(n, seed)
            out[name] = (kd, vd)
        else:
            out[name] = store_scattered(n, seed)
    return out


if __name__ == "__main__":
    import os

    res = run()
    if "--save" in sys.argv or not os.path.exists(GOLDEN):
        torch.save(res, GOLDEN)
        print(f"saved golden -> {GOLDEN} ({len(res)} cases)")
        sys.exit(0)
    gold = torch.load(GOLDEN)
    ok = True
    for name in CASES:
        a, b = res[name], gold[name]
        if isinstance(a, tuple):
            dk = (a[0] - b[0]).abs().max().item()
            dv = (a[1] - b[1]).abs().max().item()
            d = max(dk, dv)
        else:
            d = (a.float() - b.float()).abs().max().item()
        passed = d == 0.0
        ok = ok and passed
        print(f"[{'ok ' if passed else 'FAIL'}] {name:18s} max|Δ vs golden|={d:.3e}")
    print("STORE BIT-IDENTICAL" if ok else "STORE CHANGED — REVIEW")
