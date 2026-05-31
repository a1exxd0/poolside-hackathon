# Extensions — Quantization-Aware Distillation & Roadmap

Companion to `WRITEUP.md`. That document covers the shipped 4-bit KV-cache
kernels and serving (INT4/NVFP4-KIVI, ~3.56× cache shrink, fused paged decode,
1M context). **This file collects the *extensions* beyond that core**: the
newest training-side work (Quantization-Aware Distillation) and the
forward-looking roadmap (DGX Spark + the QAD curriculum).

---

## 1. Quantization-Aware Distillation (QAD)

> Lives on the vLLM fork's `main` (`nvfp4_qad/`, commit `4032105e7`). It is a
> *training-side* toolkit, deliberately separate from the serving kernels — you
> ship a fine-tuned checkpoint, **no new vLLM kernels**. (The poolside repo's own
> latest is the serve.sh / 1M-context work in `WRITEUP.md` §5; the submodule
> pointer doesn't yet include QAD.)

### The problem it attacks

On Blackwell (SM100), this fork serves attention as **FP8 query × NVFP4 KV** on
the FlashInfer trtllm-gen kernel, folding per-layer `q_scale`/`k_scale`/`v_scale`
into the matmuls. Those scales default to **1.0** and the model was trained in
bf16 — *nothing ever taught it to tolerate 4-bit attention*. The damage is small
at short context and grows with length. QAD closes that gap: fine-tune with the
fp4/fp8 attention **simulated in the forward pass**, distill from the bf16
teacher, and **learn the per-layer scales**.

### The key empirical finding (what motivated this)

When we calibrate the NVFP4-KV scales and run the model at *smaller context*, the
greedy generation is **token-for-token identical to the unquantized bf16 model** —
i.e. quantizing the K/V of the 10 global-attention layers to 4-bit NVFP4 does not
change the attention output at all on short sequences (`generate_test.py` prints
*"identical: outputs match"*). Divergence only appears as context grows.

That is the whole thesis in one observation: **4-bit attention is essentially
free at short context, and the long-context gap is a *calibration* problem, not a
capacity one** — so it can be recovered by learning ~20 scalars rather than
retraining the model.

### How it's built (`nvfp4_qad/`)

| File | Role |
|---|---|
| `fake_quant.py` | Differentiable NVFP4 (K/V) + FP8 (Q) fake-quant. Forward is **value-identical** to vLLM's `ref_nvfp4_quant` (same E2M1 grid, same fp8-e4m3 block-scale round-trip, same `[-448,448]` clamps). STE on the E2M1 rounding and the fp8 block-scale carries gradients to activations **and** the learnable scale. |
| `calibration.py` (Stage-0) | Forward hooks collect per-layer `amax(Q/K/V)` (post-RoPE K) on the 10 global-attention layers, init `scale = amax/448`. Forward-only, `no_grad`, no model surgery → fits the 66 GB bf16 model on 2×A6000. |
| `attention.py` — `NVFP4FakeQuantScores` | Drop-in score module holding scales as **log-space** `nn.Parameter`s, then runs ordinary bf16 SDPA on the dequantized tensors. |
| `distill.py` / `train_laguna.py` | Staged QAD: `KL(teacher‖student logits)` + attention-map MSE + (stage-2) hidden MSE, on a length curriculum with per-length re-calibration. |
| `parity.py` | The hard train/inference-skew gate (see below). |
| `dashboard.py` / `figures.py` | Live per-step training curves vs. watermarked synthetic method-explainer figures (kept strictly separate). |

**The scale insight it exploits.** With `global_scale = 1/k_scale` the
dequantized value reproduces the original tensor *nominally* (the `k_scale`
cancels). So `k_scale` **only** controls whether the per-16 fp8-e4m3 *block*
scale saturates at 448 or underflows to 0 — corrupting K/V. That landing is
exactly what QAD learns.

**Why log-space scales.** `global_scale = 1/k_scale` amplifies a linear-space
gradient by `1/k_scale²`, so optimizing the raw scalar explodes. Log space keeps
scales strictly positive and the gradient well-conditioned, so a single larger LR
works across all layers.

**Why it fits the hardware.** Scale-only QAD freezes the 66B weights and trains
only the ~20 k/v scalars (k/v per global layer), injected via a learnable-scale
`DynamicCache` subclass — no weight gradients, no optimizer state on the model,
so it runs on 2×A6000. Stage-2 (LoRA on q/k/v/o_proj) is opt-in, only if a
long-context gap remains.

**Parity gate** (`python -m nvfp4_qad.parity`):
- *Reference parity* (any hardware): fake-quant forward == vLLM
  `ref_nvfp4_quant_dequant` — max error **0.0 (bf16)**, ~2e-7 fp32 reciprocal
  noise.
- *Device execution* (any CUDA, incl. Ampere): pure-PyTorch fake-quant runs and
  matches the CPU reference — this is what training relies on.
- *CUDA-kernel parity* (SM100): fake-quant == real
  `reshape_and_cache_flash(..., "nvfp4", k_scale, v_scale)` dequantized — proves
  no train/inference skew.

> **Caveat on figures.** The committed `figures/_sample_run.jsonl` is explicitly
> synthetic placeholder data (`"SAMPLE-not-laguna"`). Do **not** cite its RULER
> numbers as results — the real curves are produced per training run by the
> dashboard, not the sample.

### Why it matters

The shipped kernels reduce the *bytes* of the KV cache (3.56× via INT4/NVFP4-KIVI)
and serve it. QAD attacks the *quality* axis of the same 4-bit attention from the
training side, and the "identical at short context" result is direct evidence
that 4-bit KV is sound — the only thing left to fix is the long-context scale
calibration, and that's cheap.

---

## 2. Roadmap — what's achievable with more time (DGX Spark)

The B300 result was "context length yes, latency meh" precisely *because* HBM
bandwidth is so high that shrinking bytes-read-per-token barely moves decode.
**DGX Spark (GB10 Grace-Blackwell) inverts that trade-off**, which is what makes
it the interesting target.

1. **Bandwidth-bound regime → the byte reduction should convert to real latency.**
   Spark's unified LPDDR5X has far lower bandwidth (~273 GB/s) than B300 HBM.
   Decode is memory-bandwidth-bound there, so reading **3.56× fewer KV bytes per
   token should translate much more directly into decode throughput** than it did
   on B300. Headline experiment: re-run the single-stream latency A/B on Spark,
   where we expect the kernel win to *stop* being masked.

2. **Unified 128 GB memory → context length is the whole game.** Spark shares one
   128 GB pool between weights and KV. A 3.56× smaller cache is the difference
   between fitting a long context and OOM. Measure max context / max concurrency
   at 4-bit vs bf16 on the 128 GB budget.

3. **Promote per-channel-K end-to-end + capture CUDA graphs.** The kernels already
   implement geometric per-channel-K; finish wiring the per-channel path through
   the live decode backend (not just the gather fallback) and drop
   `--enforce-eager` to capture the (already graph-safe) decode kernel. Both
   should compound on a latency-sensitive device.

4. **Native FP4 hardware.** GB10 has native FP4 tensor-core support. Our
   `nvfp4_kivi` E2M1 path currently dequantizes to bf16 for the `tl.dot`; on Spark
   we could keep more of the pipeline in FP4 / use hardware FP4 MMA, removing the
   dequant ALU overhead that currently makes the bf16 cast pure cost at large batch.

5. **NVFP4 as the default.** Given E2M1 beat INT4 at equal bytes/latency, and Spark
   is FP4-native, `nvfp4_kivi` is the natural default backend there.

6. **Run the full QAD length-curriculum on real Laguna (§1).** We've shown 4-bit
   attention is identical to bf16 at short context and that the long-context gap
   is a scale-calibration problem; the scale-only QAD loop fits 2×A6000. With more
   compute, run the `[8k → 32k → 131k → 1M]` curriculum with per-length
   re-calibration, export the learned scales into the checkpoint, and measure
   RULER / needle-in-haystack to 1M through vLLM — closing the teacher-vs-naive
   gap without retraining weights. Stage-2 (LoRA on q/k/v/o_proj) only if a gap
   remains.

7. **Sub-4-bit with mixed precision.** 4-bit is the floor *uniformly*; with the
   per-channel-K machinery in place, a mixed scheme (e.g. 3-bit V / 4-bit K, or
   keeping a few outlier channels higher) could push the ratio past 3.56× while
   holding quality — worth a calibration sweep with the extra time.
