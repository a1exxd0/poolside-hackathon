# Page-Aligned 4-bit KV Cache: INT4 with Per-Channel-K (KIVI) Scaling

*Validated on Laguna-XS.2. This doc supersedes an earlier draft that proposed an
MSE-optimal clip search on vLLM's NVFP4 kernel; the sweep below shows that was
the smallest of three levers. The real lever is the block **layout**.*

## TL;DR

vLLM's production 4-bit KV path is **NVFP4 (e2m1)** with a per-16-element block
scale along `head_dim`, set by absmax. We swept *format × block-layout ×
calibration* on real Laguna-XS.2 KV activations. The dominant lever is not the
number format and not the calibration — it is the **block layout**:

- Quantizing **K per-channel** (a 16-*token* block of one channel, à la KIVI)
  with plain **uniform INT4** cuts K reconstruction error **~25%** vs the NVFP4
  baseline, at identical memory.
- That gain **grows with context**: neutral below 512 tokens, rising to a stable
  **~20–25% KL reduction** (and ~1 pt top-1) beyond 1k tokens — the long-context
  regime KV quantization exists for.
- **INT3 is below the floor**: ~2–2.7× the distortion, getting worse with length.
- **The catch**: per-channel blocking cannot use NVFP4's hardware microscale
  (which decodes 16 *contiguous* = `head_dim` elements). Capturing this needs a
  **software INT4 KV path**, trading Blackwell's native FP4 throughput for the
  layout freedom that the quality win depends on.

## Problem

The KV cache dominates memory at long context / high batch, and decode attention
is memory-bandwidth bound. vLLM's mature low-bit path is FP8 (1 B/elem). Going to
**4-bit** (~0.56 B/elem incl. an 8-bit block scale, **3.56× vs BF16**) roughly
doubles concurrent sequences or context per HBM byte and moves fewer bytes per
attention read. A single global scale at 4-bit is a non-starter — K has known
per-channel outliers that drag a global scale huge and collapse everything else —
so any 4-bit KV scheme needs **blockwise** scaling. The question is *which* block
geometry and *which* 4-bit format.

## What vLLM already has (verified against upstream)

vLLM already ships a paged NVFP4 KV cache on Blackwell (SM100):
`csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu`, page layout
`[K_data | K_scale | V_data | V_scale]`, **16-element blocks along `head_dim`**,
scales stored as **uint8** (e4m3), quantized via `cvt_warp_fp16_to_fp4` with
`scale = absmax / 6.0`.

Correction to the earlier draft: the kernel runs **once per token**
(`token_idx = blockIdx.x; grid(num_tokens)`), not once per page. Each `head_dim`
block is one token's 16 channels and is complete the moment that token is
written — there is no page-fill event, and "wait for the page to fill" only makes
sense for the *token-axis* (per-channel) blocking we propose below, not for the
shipped `head_dim` blocking.

## What we measured

A Python harness (`kv_quant.py`) quantizes real Laguna-XS.2 K/V and sweeps
`{nvfp4, int4, int3} × {headdim-block, per-channel-block} × {absmax, mse}`. Every
4-bit cell costs identical memory; this isolates quality.

### Finding 1 — layout dominates (KEY cache)

Avg K-RMSE over 3 prompts (baseline = `nvfp4 / headdim / absmax`, what vLLM ships):

| format | layout | calib | K-RMSE | vs baseline |
|--------|--------|-------|--------|-------------|
| nvfp4  | headdim | absmax | 0.1075 | — |
| int4   | headdim | absmax | 0.1141 | **−7%** (worse) |
| nvfp4  | channel | absmax | 0.1172 | **−9%** (worse) |
| **int4** | **channel** | **mse** | **0.0808** | **+25%** |

The win is a **non-additive synergy**: switching format alone (INT4, head_dim) is
*worse*; switching layout alone (NVFP4, per-channel) is *worse*; only INT4 **and**
per-channel together win (+19%), with MSE calibration adding the last ~5 pts.
Mechanism: a per-channel block holds 16 tokens of *one* channel → near-uniform
magnitude within the block → uniform INT4 is optimal and e2m1's non-uniform levels
are wasted. K's outliers are per-channel, so this layout isolates them; the
shipped `head_dim` block straddles 16 *different* channels and one outlier poisons
the block.

### Finding 2 — V is different (the KIVI asymmetry)

V is well-behaved (no strong per-channel structure): INT4 beats NVFP4 regardless
of layout, and per-token (`head_dim`) blocking is within ~2% of per-channel. So
the design that falls out is exactly **KIVI**: **per-channel INT4 for K,
per-token INT4 for V**.

### Finding 3 — downstream: near-lossless short, win grows long

Teacher-forced KL(bf16‖scheme) and top-1 agreement vs a BF16 reference (identical
context, no autoregressive drift):

- **Short context (<1k tokens, production-faithful frozen-page protocol):** both
  4-bit schemes are near-lossless. int4-kivi: top-1 **98.6% vs 98.5%** (tie),
  KL **+10%**. Real but small — the model barely leans on a small cache.
- **Long context (8k tokens, all-KV-quantized single-pass, 3 windows):** the gap
  **grows with position** and stabilizes:

  | position | int4-kivi KL reduction | top-1 (base → kivi) |
  |----------|------------------------|---------------------|
  | 0–512    | −2%  | 82.0 → 82.1% |
  | 1024–2048 | +27% | 94.0 → 94.9% |
  | 4096–8000 | **+20%** | **93.4 → 94.5%** |

The advantage is neutral when the cache is tiny and rises to a stable ~20–25% KL
reduction (≈17% fewer top-1 errors) once real long-range context accumulates.
*Protocol caveat:* the single-pass test quantizes all K/V including each
position's nearest tokens (no bf16 hot page), so absolute KL is inflated; the
**trend and relative gap** are the signal. The true production number (with a
bf16 hot page) sits between the short-context +10% and this +20–25%.

### Finding 4 — 4-bit is the floor

INT3 (4.57× vs BF16) runs **~2–2.7× the baseline KL** and gets *relatively worse*
with context (−112% at 4–8k). Per-channel layout helps it slightly but
bit-starvation dominates. The extra 28% memory saving isn't worth it. Note: RMSE
predicts the 4-bit layout win but mispredicts at INT3 — treat RMSE as a 4-bit
screen only, and confirm low-bit downstream.

## Why this needs a software INT4 path (the catch)

NVFP4's hardware microscale decodes 16 **contiguous** elements = `head_dim`.
Per-channel blocking is along the **token** axis — incompatible with the hardware
format. In the `head_dim` layout the hardware forces, **NVFP4 ≥ INT4** (0.107 vs
0.114); INT4 only wins *with* per-channel blocking, which requires a software
dequant-to-BF16 read path (the same shape as today's FP8 KV path — store low-bit,
dequant in the attention backend, attend in BF16). On Blackwell this gives up
native FP4 throughput, but KV decode is bandwidth-bound, and INT4 dequant (an
int→float multiply) is cheaper than e2m1 reconstruction.

## Implementation sketch (if pursued)

A software INT4 KV path, not an edit to the NVFP4 kernel:

1. **Layout:** K per-channel (16-token blocks), V per-token (`head_dim` blocks),
   one INT4 + one 8-bit scale per block. Same scale count and memory as the
   NVFP4 layout.
2. **Fill:** quantize a K page when its 16 tokens complete (per-channel scale
   needs the full page — *this* is where the "calibrate once at page fill"
   argument is actually correct), keep the partial hot page in BF16. V quantizes
   per token immediately.
3. **Calibration:** MSE-optimal clip scale (small extra gain over absmax; cheap,
   one-time per page).
4. **Read:** dequant to BF16 in the attention backend; attention unchanged.

Prior art: **KIVI** (per-channel K, per-token V), **QServe** (W4A8KV4 on
A100/H100) — INT4 KV is well-trodden off the NVFP4 hardware path.

## Cost / Benefit

| Item | Value |
|------|-------|
| Memory vs BF16 | 3.56× (same as NVFP4; INT3 4.57× but below the floor) |
| K reconstruction vs NVFP4 baseline | −25% RMSE (robust across prompts/lengths) |
| Downstream KL vs NVFP4 baseline | +10% short ctx → ~+20–25% beyond 1k tokens |
| Top-1 vs NVFP4 baseline | tie short ctx → +~1 pt (≈17% fewer errors) long ctx |
| Cost | new software INT4 KV kernel + attention dequant; forgoes native FP4 MMA |

## Honest caveats

Python simulation, not the kernel. One model (Laguna-XS.2). Long-context evidence
is on code (in-distribution for a code model) over 3 windows; the single-pass
protocol inflates absolute KL. Should be confirmed on natural-language / retrieval
long-context tasks and, ultimately, in the kernel.

## One-sentence summary

The lever for 4-bit KV is the block **layout**, not the number format: uniform
INT4 with **per-channel-K (KIVI)** blocking beats vLLM's NVFP4 baseline by ~25% K
reconstruction and a long-context-growing ~20–25% KL reduction at identical
memory — but it lives off the NVFP4 hardware path, so it costs a software INT4 KV
kernel; INT3 is below the quality floor.
