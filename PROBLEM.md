# Page-Aligned 4-bit KV Cache with Blockwise Scaling

## Problem

vLLM's PagedAttention stores the KV cache in fixed-size contiguous pages
(default 16 tokens per block). For quantization, vLLM's production path is
**FP8 (e4m3) with a single per-tensor scale** (`k_scale` / `v_scale` per
layer). This works because FP8 has 4 exponent bits — it carries real dynamic
range, so one global scale plus the floating-point exponent adapts locally
"for free."

The KV cache is the dominant memory consumer at long context / high batch,
and at decode time attention is **memory-bandwidth bound**. We want to halve
it again — go from FP8 (1 byte/elem) to **4-bit** (0.5 byte/elem), ~3.5× vs
FP16 — to roughly double concurrent sequences (throughput) or context length
in the same HBM, and move fewer bytes per attention read.

**Why we can't just reuse the FP8 approach at 4 bits:**

- INT4 has 16 evenly-spaced levels. NVFP4 (e2m1) is 4 bits with only 2
  exponent bits — 16 representable values with almost no dynamic range.
- With a **single global scale**, that scale must cover the largest magnitude
  in the entire tensor. KV caches have known outlier structure (a few K
  channels blow up). The global scale gets dragged huge by outliers, and the
  remaining ~99% of values collapse into 2–3 quantization levels.
- Reconstruction error explodes. **Single scale + 4-bit is a non-starter** —
  this is dynamic-range math, not a tuning issue.

## Solution

Use **4-bit quantization (INT4 or NVFP4) with blockwise scaling, where the
quant block is aligned to the PagedAttention page**, and compute each block's
scale **once, at the moment the page fills**, by minimizing dequantization
reconstruction error.

### 1. Blockwise scaling is the standard fix (not a hack)

Give each small block of elements its own scale matched to its local max. An
outlier now only degrades the 16–32 values in its own block, not the whole
tensor; every value uses the full 16 levels relative to its local
neighborhood. This is *exactly what NVFP4 is*: 4-bit values + a per-16-element
FP8 block scale (MXFP4 = block 32, power-of-two scale). The block scale is not
a bonus feature — it is the mechanism that makes 4-bit usable. We are applying
a proven recipe (KVQuant, KIVI, Atom, QuaRot, NVFP4) to the KV cache.

### 2. Align blocks to the page — the architecture hands us the boundary

PagedAttention already stores KV in fixed-size contiguous pages and tiles the
K cache along the token dimension. Aligning the quant block to the page (or a
sub-tile of it) means:

- **Scales live next to the page.** Allocate a page → allocate its scales;
  free a page → free its scales. No scattered metadata, no separate
  bookkeeping, no new fragmentation story. It rides the existing block
  allocator.
- **Coherent access.** When a kernel reads a page, the matching scales are
  right there in one contiguous fetch.

### 3. The "free lunch": calibrate once, when the page fills

KV entries for past tokens are **immutable** — once a page is full, those
keys/values never change; they're read thousands of times but written once.
So:

- Compute the scale **once**, at fill time, **off the hot decode path**.
- Because it's one-time, afford better than naive absmax: a small grid search
  over clip ratios to find the scale that minimizes `‖x − dequant(quant(x))‖`
  (MSE-optimal / clipping-optimal). Absmax over-weights the outlier; an
  MSE-optimal scale routinely cuts error meaningfully.
- The per-element search cost is amortized over the entire read lifetime of
  the page → effectively negligible.

The data freezes exactly when we want to calibrate it. This is the cleanest
possible fit.

### 4. Cost/benefit

- **Memory:** 4-bit + FP8 scale per 16 values = 0.5 bit/element overhead →
  ~4.5 effective bits, vs FP8's 8. ~2× cut on top of FP8.
- **Throughput:** ~2× concurrent sequences or context length; decode kernel
  may run *faster* (half the bytes moved, bandwidth-bound).
- **Overhead is tiny and one-time** (fill-time calibration, contiguous scale
  reads).

## Anticipated objections (all engineering cost, not soundness)

- **Dequant cost in the attention kernel** — FP8 already dequants on the fly;
  blockwise just adds a coherent per-block scale read. NVFP4 dequant is native
  on Blackwell; INT4 needs custom kernels (Marlin-style) which exist. Decode
  is bandwidth-bound, so saved bytes often pay for the extra math.
- **Calibration latency** — one-time, off critical path, can run async on page
  completion.
- **Accuracy regression** — mitigate with MSE-calibrated scales; optionally
  keep sensitive layers / first tokens (attention sinks) at higher precision.
  Then *measure*: perplexity + downstream evals vs the FP8 baseline.
- **Block shape for K outliers** — K outliers are per-channel, so choose block
  geometry to capture that (tile along `head_size` within the page, or 2D
  tiles) — exactly what NVFP4's 1×16 blocks along the contraction dim do. A
  refinement, not a blocker.

## One-sentence summary

FP8 survives on one scale because its exponent bits carry dynamic range; 4-bit
has none, so blockwise scaling isn't optional — it's the definition of how
NVFP4 works. PagedAttention hands us free, naturally-aligned,
immutable-once-full blocks to attach those scales to and to calibrate exactly
once. We get ~2× more KV cache for ~0.5 bit/element of overhead, on a proven
recipe. The only real cost is kernel engineering — a question of effort, not
of whether the idea is correct.
