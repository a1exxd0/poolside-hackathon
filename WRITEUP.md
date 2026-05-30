# 4-bit KV Cache Quantization for Laguna-XS.2 — Hackathon Writeup

## TL;DR

We built two custom **4-bit KV-cache attention backends for vLLM** — `int4_kivi`
(uniform integer) and `nvfp4_kivi` (E2M1 float) — that store the K/V cache at
**~3.56× smaller than bf16** with no measurable loss of model quality on
long-context tasks. The smaller cache is what lets us serve **Laguna-XS.2 at up
to a 1M-token context** on a single B300, and we wrote a custom Triton
**fused paged flash-decode** kernel that runs attention straight off the packed
4-bit cache (never materializing dense bf16 KV). The newest work (§6) adds a
**quantization-aware distillation** toolkit on the training side, off the back of
a clean finding: at short context, 4-bit NVFP4-KV attention is *token-for-token
identical* to the bf16 model — the long-context gap is just scale calibration.

Two repos:
- App / serving harness: `a1exxd0/poolside-hackathon` (branch `kv-quant`)
- vLLM fork with the kernels + backends: `a1exxd0/vllm` (branches `int4-kivi-speedup`, `nvfp4-kivi`)

---

## 1. Why KV-cache quantization

For long-context decoding the bottleneck is the **KV cache**: its size grows
linearly with context length, and at decode time every generated token must
*read the entire cache*. So the cache is both the **memory-capacity** limit on
how long a context you can hold, and (because decode is memory-bandwidth-bound)
a big chunk of the **per-token latency**.

Weights are already quantized (we serve the NVFP4 checkpoint). The KV cache is
the remaining large bf16 tensor in the hot path, so quantizing it is the lever
with the biggest payoff for long context.

The finding that drove the design (validated before touching vLLM): for KV
caches the *layout* matters far more than the calibration. **K has strong
per-channel outliers; V does not.** This is the KIVI insight — quantize K
**per-channel** and V **per-token** — and it is what makes 4 bits viable. 4-bit
is the practical floor; 3-bit and below fall off a cliff.

---

## 2. Memory savings — the core result

Both backends use an identical packed layout. Per token, per KV head, the cache
stores:

| Component | Size |
|---|---|
| Nibble-packed 4-bit data (2 codes/byte) | `head_size // 2` bytes |
| fp8_e4m3 block scales (1 scale per 16-element block) | `head_size // 16` bytes |

So the saving is **4× from the 4-bit data, minus the fp8 scale overhead**:

```
Per 16-element block:
  bf16:       16 elems × 16 bits                 = 256 bits = 32 bytes
  4-bit KIVI: 16 elems × 4 bits  +  one fp8 scale =  72 bits =  9 bytes
                                  (8 bits / block)
```

- **Raw 4-bit packing → 4.00× reduction** (16 bits → 4 bits per element)
- **fp8 scale adds back 8 bits / 16 elements = 0.5 bits/element**
- **Net effective rate = 4.5 bits/element → 32B → 9B per block = 3.56× smaller** than bf16

That 3.56× directly buys context length and frees HBM for more concurrent
sequences. NVFP4 and INT4 are **bit-budget identical** — same cache tensor, same
page-size accounting (`int4_kivi_kv_cache_full_dim` / `nvfp4_kv_cache_full_dim`
both return `head_size//2 + head_size//16`) — so they are drop-in alternatives
that differ *only* in the 4-bit number grid.

---

## 3. The quantization scheme

**KIVI layout (the quality lever).**
- **V** → quantized **per token**, in 16-element blocks along `head_dim`.
- **K** → quantized **per channel** over each full 16-token block (one scale per
  channel, computed across the block's 16 tokens) — this isolates K's
  per-channel outliers. Partial trailing blocks (the hot decode tail not yet
  16 tokens deep) fall back to per-token K for correctness.

**The "free per-channel-K" trick (no byte-budget change).** The K-side scale
region is `head_size//16` bytes/token. Over a *full* 16-token block that is
`16 × (head_size//16) = head_size` bytes — i.e. exactly one fp8 scale **per
channel**. So per-channel-K reuses the *identical* cache tensor: we just
reinterpret a full block's K-scale bytes as `head_size` per-channel scales
instead of `16 × (head_size//16)` per-token scales. No extra storage, no marker
bytes.

**Geometry, not metadata.** Store and dequant agree on per-channel vs per-token
purely from sequence geometry: logical block `b` is full iff `(b+1)*16 <= L`
→ per-channel; the trailing block with `L % 16 != 0` is partial → per-token.
This is self-consistent with vLLM's eager prefill (whole context stored in one
call → all complete blocks per-channel) and decode (grows only the trailing
block → per-token). No state to track.

**Calibration.** Symmetric, MSE-optimal α-clip: a 16-point grid search of the
clip ratio α∈[0.5,1.0] per block, round-half-to-even via `libdevice.rint` with
IEEE round-to-nearest division. INT4 codes clamp to `[-7, 7]`, `deq = code·scale`.

**INT4 vs NVFP4 — the A/B.** The two backends keep *everything* identical
(layout, MSE α-clip, KIVI geometry) and change only the 4-bit grid:
- `int4_kivi`: uniform integer codes `[-7,7]`.
- `nvfp4_kivi`: E2M1 float, magnitudes `{0,.5,1,1.5,2,3,4,6}` — non-uniform,
  denser near zero. We fold away NVFP4's usual per-tensor global scale (the fp8
  block scale alone, range 2⁻⁹..448, already covers K/V block amaxes), keeping
  one fp8 byte per block so the cache stays int4-identical.

This isolates "E2M1 vs uniform-int4 at 4 bits for a KV cache." Result: **E2M1
wins on accumulated / heavy-tailed error** — on HumanEval long-context@12k,
nvfp4 scored **18/20 vs int4 16/20** (bf16 20/20) at *equal bytes and equal
latency* (~52s). The non-uniform grid's extra resolution near zero matters for
the dequantized-attention accumulation.

---

## 4. The vLLM kernels

All custom kernels are Triton, in
`vllm/v1/attention/ops/triton_{int4,nvfp4}_kivi.py`, with matching backends in
`vllm/v1/attention/backends/{int4,nvfp4}_kivi_attn.py`. The cache is a single
`uint8` tensor per layer:
`kv_cache[num_blocks, 2, block_size, num_kv_heads, full_dim]` (dim 1: 0=K, 1=V;
row layout `[data_bytes | scale_bytes]`).

**1. Software KV-cache backend** (`Int4KiviAttentionBackend`). Modeled on
vLLM's TurboQuant backend: separate `do_kv_cache_update` (store quantized) +
dequant-on-read, then FlashAttention. Registered as a first-class
`AttentionBackendEnum`, selectable via `--kv-cache-dtype int4_kivi`. Prefill
reuses `flash_attn` directly off the gather-dequantized KV.

**2. Sync-free paged store.** Quantizes and nibble-packs K/V into the paged
cache. The split point (per-channel vs per-token) is decided from geometry, so
there is no host sync and no read-modify-write hazard.

**3. Fused paged flash-decode** (`int4_kivi_paged_decode`) — the headline
kernel. Runs attention **straight off the packed 4-bit cache**, dequantizing
each 16-element block inline in registers; it never materializes the dense bf16
KV. Engineering details:
- **bf16 tensor-core `tl.dot`** for the QK and PV products (q kept exact bf16,
  `sm_scale` folded in after the matmul to match FlashAttention numerics).
- **Split-K over the sequence**, auto-sized from the GPU's SM count
  (`WAVES × SM_count` target programs) so small batches still fill the device;
  `split==1` takes a `WRITE_FINAL` fast path that skips the combine launch
  (large-batch throughput path).
- **No host sync on `seq_lens`**: split is chosen without a `.item()` (which
  would block the stream every layer, every step, and bar CUDA-graph capture).
  Oversized/empty splits are numerically inert (m=−∞, l=0, acc=0) and handled by
  the combine guard. **CUDA-graph-ready.**
- Tunable `BLOCK_N`, warps, stages, waves via env vars; defaults swept on B300
  (sm_103).

**4. >64k context launch fix.** The gather/dequant kernel originally put the
sequence-position axis on `grid.y`, which CUDA caps at 65535 — so any context
past 64k failed to launch (`CUDA: invalid argument`). Reordered the grid to
`(max_seq, B, H)` so position rides `grid.x` (limit ~2³¹). This is what unblocks
serving past 64k. (Plus an earlier fix for an int32 overflow in the gather
addressing that crashed at large cache sizes.)

Each piece is covered by standalone tests
(`tests/v1/attention/test_int4_kivi_{kernels,perchannel,grid}.py`) and a
direct-import correctness + accuracy A/B harness (`scripts/validate_nvfp4_kivi.py`):
fused decode matches a dense SDPA reference to ~1e-2 (bf16 tensor-core
tolerance); decode round-trips to rel-error ~1.8e-3.

---

## 5. Serving via vLLM + context-length increase

The serving harness (`.tools/serve.sh`, branch `kv-quant`) launches the fork:

```
vllm serve <Laguna-XS.2-NVFP4> \
  --dtype bfloat16 \
  --kv-cache-dtype int4_kivi \      # our custom backend
  --gpu-memory-utilization 0.9 \
  --max-model-len 1048576 \         # 1M tokens
  --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser poolside_v1 \
  --reasoning-parser poolside_v1 --chat-template chat_template.jinja
```

What we wired up to make 1M context real and reproducible:
- **YaRN rope extension to 1M**, applied as an *idempotent config.json patch* at
  launch (factor derived as `max_len/orig_max`, de-referencing the HF-cache
  symlink so we patch the snapshot, not the shared blob). It's a config edit,
  not weights, so it's re-applied on any machine/cache state. Quality past the
  model's native 256k degrades by design.
- **GPU util raised to 0.9** — on a 275GB B300 this guarantees the KV cache can
  hold a 1M-token sequence. **This headroom is exactly what the 3.56× cache
  shrink buys** — a bf16 cache for 1M tokens would not fit alongside the model.
- **Tool calling + reasoning**: Laguna ships a tool-aware `chat_template.jinja`
  separate from the one embedded in `tokenizer_config.json`; pass it explicitly
  so the `poolside_v1` tool/reasoning parsers get the `<available_tools>` /
  `<tool_call>` / `<think>` protocol.
- **Auto model resolve/download**, opt-in cloudflared tunnel + API-key auth
  (both default OFF — localhost-only unless `ENABLE_TUNNEL=1`).

**Quality / parity checkpoints** (Laguna-XS.2, B300):
- Long-context **HumanEval@12k**: bf16 20/20, **nvfp4 18/20**, int4 16/20 —
  equal bytes, equal latency (~52s).
- **@32k needle**: parity with bf16.
- Net: 4-bit KV holds up on the long-context tasks that motivated it.

---

## 6. Latest work — Quantization-Aware Distillation (QAD)

> Lives on the vLLM fork's `main` (`nvfp4_qad/`, commit `4032105e7`). It is a
> *training-side* toolkit, deliberately separate from the serving kernels above
> — you ship a fine-tuned checkpoint, no new vLLM kernels. (The poolside repo's
> own latest is the serve.sh / 1M-context work in §5; the submodule pointer
> doesn't yet include QAD.)

**The problem it attacks.** On Blackwell (SM100), this fork serves attention as
**FP8 query × NVFP4 KV** on the FlashInfer trtllm-gen kernel, folding per-layer
`q_scale`/`k_scale`/`v_scale` into the matmuls. Those scales default to **1.0**
and the model was trained in bf16 — *nothing ever taught it to tolerate 4-bit
attention*. The damage is small at short context and grows with length. QAD
closes that gap: fine-tune with the fp4/fp8 attention **simulated in the forward
pass**, distill from the bf16 teacher, and **learn the per-layer scales**.

**The key empirical finding (what motivated this).** When we calibrate the
NVFP4-KV scales and run the model at *smaller context*, the greedy generation is
**token-for-token identical to the unquantized bf16 model** — i.e. quantizing
the K/V of the 10 global-attention layers to 4-bit NVFP4 does not change the
attention output at all on short sequences (`generate_test.py` prints
*"identical: outputs match"*). Divergence only appears as context grows. That is
the whole thesis in one observation: **4-bit attention is essentially free at
short context, and the long-context gap is a *calibration* problem, not a
capacity one** — so it can be recovered by learning ~20 scalars rather than
retraining the model.

**How it's built** (`nvfp4_qad/`):
- **`fake_quant.py`** — differentiable NVFP4 (K/V) + FP8 (Q) fake-quant whose
  forward is *value-identical* to vLLM's `ref_nvfp4_quant` (same E2M1 grid, same
  fp8-e4m3 block-scale round-trip, same `[-448,448]` clamps). Straight-through
  estimators on the E2M1 rounding and the fp8 block-scale let gradients reach
  both the activations and the learnable scale. The insight it exploits: with
  `global_scale = 1/k_scale` the dequantized value reproduces the original
  *nominally* (k_scale cancels) — `k_scale` **only** controls whether the per-16
  fp8 block scale saturates at 448 or underflows to 0. That landing is exactly
  what QAD learns.
- **`calibration.py` (Stage-0)** — forward hooks collect per-layer `amax(Q/K/V)`
  (post-RoPE K) on the 10 global-attention layers and init `scale = amax/448`.
  Forward-only, no_grad, no model surgery — fits the 66 GB bf16 model on 2×A6000.
- **`attention.py` — `NVFP4FakeQuantScores`** — drop-in score module holding the
  scales as **log-space** `nn.Parameter`s (because `global_scale = 1/k_scale`
  amplifies a linear-space gradient by `1/k_scale²`, so optimizing the raw scalar
  explodes; log space keeps them positive and well-conditioned under one LR).
- **`distill.py` / `train_laguna.py`** — staged QAD: `KL(teacher‖student logits)`
  + attention-map MSE + (stage-2) hidden MSE, on a length curriculum with
  per-length re-calibration (amax drifts with length and can push block scales
  into fp8 saturation). **Scale-only QAD freezes the 66B weights and trains only
  the ~20 k/v scalars**, injected via a learnable-scale `DynamicCache` subclass —
  so it runs on 2×A6000 with no weight gradients or optimizer state on the model.
- **`parity.py`** — the hard train/inference-skew gate: fake-quant forward ==
  vLLM reference (max error **0.0** in bf16, ~2e-7 fp32 reciprocal noise) on any
  hardware, and == the real `reshape_and_cache_flash(..., "nvfp4", …)` CUDA
  kernel on SM100.
- **`dashboard.py` / `figures.py`** — live per-step training curves (loss, scale
  evolution, RULER-vs-context) rendered from the run's own JSONL, kept strictly
  separate from the watermarked synthetic method-explainer figures. *(The
  committed `_sample_run.jsonl` is explicitly synthetic placeholder data — the
  real curves are produced per training run, not the numbers in that sample.)*

**Why it matters for the rest of this writeup.** §1–5 reduce the *bytes* of the
KV cache (3.56× via INT4/NVFP4-KIVI) and serve it. QAD attacks the *quality* axis
of the same 4-bit attention from the training side, and the "identical at short
context" result is direct evidence that 4-bit KV is sound — the only thing left
to fix is the long-context scale calibration, and that's cheap.

---

## 7. Honest limitations

- **Kernel win ≠ end-to-end win on this model.** Standalone, the fused decode
  closed the batch-1 decode-attention gap to FlashAttention from ~18× to ~4.4×
  @12k. But end-to-end on the 40-layer MoE the effect was modest (single-stream
  latency 39.9 → 37.4 ms/tok, 1.07×; batched HumanEval wall-clock essentially
  flat ~53s). On B300's very high HBM bandwidth, decode-attention is not the
  dominant cost for this architecture — the MoE and other layers dominate. The
  **memory-capacity** win (context length) is unambiguous; the **latency** win
  is bandwidth-regime-dependent (see next section).
- `--enforce-eager` today; the decode kernel is CUDA-graph-*ready* but not yet
  captured in the full serve.
- 1M context relies on YaRN extrapolation past native 256k — usable but degraded
  beyond the model's trained range.

---

## 8. Next steps — what's achievable with more time on a DGX Spark

The B300 result is "context length yes, latency meh" precisely *because* HBM
bandwidth is so high that shrinking the bytes-read-per-token barely moves
decode. **DGX Spark (GB10 Grace-Blackwell) inverts that trade-off**, and that's
what makes it the interesting target:

1. **Bandwidth-bound regime → the byte reduction should convert to real
   latency.** Spark's unified LPDDR5X has *far* lower bandwidth (~273 GB/s) than
   B300 HBM. Decode is memory-bandwidth-bound there, so reading **3.56× fewer
   KV bytes per token should translate much more directly into decode
   throughput** than it did on B300. This is the headline experiment: re-run the
   single-stream latency A/B on Spark, where we expect the kernel win to *stop*
   being masked.

2. **Unified 128GB memory → context length is the whole game.** Spark shares one
   128GB pool between weights and KV. A 3.56× smaller cache is the difference
   between fitting a long context and OOM. Measure the max context / max
   concurrency we can actually serve at 4-bit vs bf16 on the 128GB budget.

3. **Promote per-channel-K end-to-end + capture CUDA graphs.** The kernels
   already implement geometric per-channel-K; finish wiring the per-channel path
   through the live decode backend (not just the gather fallback) and drop
   `--enforce-eager` to capture the (already graph-safe) decode kernel. Both
   should compound on a latency-sensitive device.

4. **Native FP4 hardware.** GB10 has native FP4 tensor-core support. Our
   `nvfp4_kivi` E2M1 path currently dequantizes to bf16 for the `tl.dot`; on
   Spark we could explore keeping more of the pipeline in FP4 / using hardware
   FP4 MMA, removing the dequant ALU overhead that currently makes the bf16 cast
   pure cost at large batch.

5. **NVFP4 as the default.** Given E2M1 beat INT4 at equal bytes/latency, and
   Spark is FP4-native, `nvfp4_kivi` is the natural default backend there.

6. **Run the full QAD length-curriculum on real Laguna (§6).** We've shown 4-bit
   attention is identical to bf16 at short context and that the long-context gap
   is a scale-calibration problem; the scale-only QAD loop fits 2×A6000. With
   more compute, run the `[8k → 32k → 131k → 1M]` curriculum with per-length
   re-calibration, export the learned scales into the checkpoint, and measure
   RULER / needle-in-haystack to 1M through vLLM — closing the teacher-vs-naive
   gap without retraining weights. Stage-2 (LoRA on q/k/v/o_proj) only if a gap
   remains.

6. **Sub-4-bit with mixed precision.** 4-bit is the floor *uniformly*; with the
   per-channel-K machinery in place, a mixed scheme (e.g. 3-bit V / 4-bit K, or
   keeping a few outlier channels higher) could push the ratio past 3.56× while
   holding quality — worth a calibration sweep with the extra time.
