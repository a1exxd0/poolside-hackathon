# INT4-KIVI vLLM Integration Design

> ## 0. Implemented state & results (updated 2026-05-30)
>
> **Status: implemented, serving, and benchmarked on B300.** The `int4_kivi`
> `kv_cache_dtype` selects a custom software INT4 KV backend that quantizes K/V to
> nibble-packed INT4 on write and dequantizes to bf16 on read, then runs
> FlashAttention.
>
> **Shipped files**
> - `vllm/vllm/v1/attention/backends/int4_kivi_attn.py` — backend (`Int4KiviAttentionBackend`,
>   `Int4KiviAttentionImpl`). Store via `do_kv_cache_update`; read via `forward`.
> - `vllm/vllm/v1/attention/ops/triton_int4_kivi.py` — Triton store + gather/dequant kernels.
> - `scripts/needle_serving.py`, `scripts/longctx_code_serving.py` — serving benchmarks.
> - `scripts/diag_kernel_perchannel.py` — kernel-correctness diagnostic (per-channel K).
>
> **Active layout (what the code actually does — differs from the design below)**
> - Cache tensor: `(num_blocks, 2, block_size=16, num_kv_heads, full_dim)` uint8,
>   `full_dim = head_size//2 (INT4 data) + head_size//16 (fp8_e4m3 scales)` (=72 for D=128).
> - **V**: per-token, head_dim 16-element blocks, MSE-optimal e4m3 scale. Stored immediately.
> - **K**: **per-channel** over each *full* 16-token block (one scale per channel over the
>   block's 16 tokens); the **partial trailing block falls back to per-token** K. This is a
>   purely **geometric rule** keyed off `slot_mapping` — there is **no bf16 hot-page buffer**
>   and **no separate K-scale tensor** (the §2(c)/§4 design ideas below were *not* used). The
>   per-channel K scales are packed into the block's existing K-scale byte region (channel `c`
>   → token `c//ND`, scale-byte `c%ND`), so the byte budget is identical to NVFP4 / per-token.
> - **Prefill fast-path**: when a batch is all first-chunk prefill
>   (`max_query_len == max_seq_len`), attention runs flash on the raw bf16 K/V (no dequant).
>   INT4 is exercised on **decode** and **chunked-prefill continuations**, which dequant the
>   whole cached context to dense bf16 and run `flash_attn_varlen`.
>
> **Results (Laguna-XS.2, vLLM serving, enforce_eager):**
>
> | benchmark | bf16 (`auto`) | int4_kivi |
> |---|:--:|:--:|
> | Needle-in-code, total (20 trials × {8,16,32}k) | 90% (54/60) | 78% (47/60) |
> | Needle-in-code @ 32k | 90% | **90%** (parity) |
> | HumanEval pass@1, short (~200 tok) | 100% | 95% |
> | HumanEval pass@1, long (~12.2k tok) | 100% | 80% |
>
> The pipeline runs end-to-end and codes correctly at long context; 4-bit KV costs a few
> problems, growing with context (expected). Decode is ~3× slower than bf16 at 12k because
> the read path materializes the full context as dense bf16 every step — see PROGRESS.md
> "Future work" for the fused-decode optimization.
>
> *The sections below (1–4) are the original pre-implementation design exploration. They
> are kept for the byte-budget math and the vLLM integration map, but where they describe a
> BF16 hot-page buffer, a separate compact K-scale tensor, or token-0 scale reuse, that was
> superseded by the geometric full-block rule described above.*

---

*All file:line citations were verified by reading the listed files.*

---

## 1. Architecture Overview

### What `int4_kivi` is

A new `kv_cache_dtype` string that activates a **software INT4 KV path** with
KIVI-style asymmetric blocking:

- **K**: quantized per-channel (a 16-token block of *one* channel → absmax or
  MSE-optimal scale, INT4 symmetric [-7, 7]). Each K page is quantized *once*
  when all 16 tokens of that page arrive.
- **V**: quantized per-token (one 16-channel head-dim block per token → scale
  per 16 consecutive elements), exactly like NVFP4 but with INT4 data. Each V
  token is quantized immediately on write.

Both use the same memory layout as NVFP4:
`[K_data | K_scale | V_data | V_scale]` per page, scales stored as `uint8`
(fp8-equivalent at 1 B/scale), data packed as INT4 (0.5 B/element).

### Why it lives *off* the NVFP4 hardware path

NVFP4's hardware microscale (SM100 `cvt_warp_fp16_to_fp4`) decodes 16
**contiguous** elements = `head_dim` direction.  Per-channel blocking is along
the **token** axis — the scales straddle rows not columns — so the hardware
format is simply incompatible.  The INT4 per-channel win (+25% K RMSE, growing
to +20–25% KL reduction at long context) requires a software dequant-to-BF16
read path, identical in shape to today's FP8 and turboquant paths: store
low-bit, dequant in the attention backend, attend in BF16.

The backend to model this on is **turboquant_attn.py** (`TurboQuantAttentionBackend`),
not the NVFP4/FlashInfer path: turboquant is already a pure-Python, software
low-bit KV cache with a custom store kernel, custom decode dequant, and no
dependency on hardware FP4 MMA.  The main structural difference from TurboQuant
is that `int4_kivi` stores K and V in **separate** tensors (like NVFP4) rather
than an interleaved K|V slot, and it exposes standard `(num_blocks, 2,
block_size, num_kv_heads, full_dim)` cache shape so it can reuse FlashAttention
for the prefill path without a full continuation-dequant.

---

## 2. Ordered Change List

### (a) Dtype Registration

**Step 1 — `vllm/vllm/config/cache.py`, line 19–35 (`CacheDType` Literal)**

Add `"int4_kivi"` to the `CacheDType` Literal.

```python
CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
    "int4_kivi",           # <-- add this
]
```

**Step 2 — `vllm/vllm/utils/torch_utils.py`, line 32–51 (`STR_DTYPE_TO_TORCH_DTYPE`)**

Add the mapping (INT4 is stored as `int8` — the packing to nibbles happens
inside the store kernel):

```python
"int4_kivi": torch.uint8,   # packed nibbles, same storage dtype as nvfp4
```

**Step 3 — `vllm/vllm/utils/torch_utils.py`, line 76–81 (`is_quantized_kv_cache`)**

Include `int4_kivi`:

```python
def is_quantized_kv_cache(kv_cache_dtype: str) -> bool:
    return (
        kv_cache_dtype.startswith("fp8")
        or kv_cache_dtype.endswith("per_token_head")
        or kv_cache_dtype in ("nvfp4", "int4_kivi")  # extended
    )
```

**Step 4 — `vllm/vllm/v1/kv_cache_interface.py`, line 59–69 (`KVQuantMode` + `get_kv_quant_mode`)**

Add `INT4_KIVI = 5` to `KVQuantMode` and wire it in `get_kv_quant_mode`:

```python
class KVQuantMode(IntEnum):
    ...
    NVFP4 = 4
    INT4_KIVI = 5   # per-channel-K + per-token-V, software INT4

    @property
    def is_int4_kivi(self) -> bool:
        return self == KVQuantMode.INT4_KIVI

def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    ...
    if kv_cache_dtype == "nvfp4":
        return KVQuantMode.NVFP4
    if kv_cache_dtype == "int4_kivi":        # <-- add
        return KVQuantMode.INT4_KIVI
    ...
```

---

### (b) Cache Shape / Byte Budget

**Step 5 — `vllm/vllm/utils/torch_utils.py`, after `nvfp4_kv_cache_full_dim` (line 415)**

Add two helpers that mirror `nvfp4_kv_cache_full_dim` / `nvfp4_kv_cache_split_views`:

```python
def int4_kivi_kv_cache_full_dim(head_size: int) -> int:
    """Packed last dim for INT4-KIVI KV cache.
    
    K layout: int4 data + uint8 scale (per-channel, 16-token blocks).
      Per token per head: head_size//2 data bytes + head_size//16 scale bytes.
    V layout: int4 data + uint8 scale (per-token, 16-channel blocks).
      Identical byte count per token per head.
    Both sides: head_size//2 + head_size//16 = 9*head_size//16 bytes.
    Same formula as nvfp4_kv_cache_full_dim.
    """
    return head_size // 2 + head_size // 16


def int4_kivi_kv_cache_split_views(
    kv_cache: torch.Tensor,
) -> tuple[tuple, tuple]:
    """Split INT4-KIVI cache into data and scale views.
    
    Reuses _nvfp4_split_data_scale — the physical layout is identical
    (data_dim = head_size//2, scale_dim = head_size//16, packed contiguously).
    Per-page layout: [K_data | K_scale | V_data | V_scale].
    """
    return nvfp4_kv_cache_split_views(kv_cache)
```

> **CORRECTION (verified against kv_quant.py + PROBLEM.md).** The derivation
> below at one point miscounts K's scales (8/page) and concludes int4_kivi is
> ~5% smaller than NVFP4. That is **wrong**. Per-channel K over a 16-token page
> has **one scale per channel = `head_size` = 128 scales/page/head** — exactly
> equal to V's `page_tokens * head_size/16 = 128` and to each NVFP4 side. So
> **int4_kivi memory == NVFP4 memory: 0.5625 B/elem, 3.56× vs BF16** (1-byte
> e4m3 scales), exactly as PROBLEM.md states. With fp16 scales the prototype is
> 3.2×. K and V *do* share one `full_dim` formula. Read the rest for the layout
> mechanics, not the (mis-stated) "5% cheaper" conclusion.

**Byte-budget proof (head_dim = 128, PAGE = 16 tokens, K-per-channel layout):**

```
head_dim = 128, page_tokens = 16

---- INT4 data ----
INT4 per element = 0.5 bytes (nibble-packed, 2 elements per uint8 byte).
K data per token per head = 128 × 0.5 = 64 bytes
V data per token per head = 128 × 0.5 = 64 bytes

---- Scales ----
One uint8 scale per 16-element block.

K scales (per-channel): K has a single scale per channel per 16-token page.
  Channels per head = 128. Blocks per page = 128 / 16 = 8... wait.

Clarification: the scale count per token must be *identical* to NVFP4 so that
we reuse the same full_dim layout formula.

NVFP4 layout: scales are stored per 16-element group WITHIN head_dim (per token).
  Scale count per token per head = head_size / 16 = 128 / 16 = 8 scales.
  Scale bytes per token per head = 8 × 1 = 8 bytes.
  Total bytes per token per head (K or V) = 64 + 8 = 72 bytes.
  full_dim = 72 bytes (matches: head_size//2 + head_size//16 = 64 + 8 = 72).

INT4-KIVI K layout (per-channel, 16-token page blocks):
  K quantization is per 16-token block of one channel.
  Channels per head = head_size = 128.
  Blocks per page = head_size / 16 = 8 blocks (same count as NVFP4).
  Scales per page per head = 8.
  Scales per token per head (amortised over page_tokens=16) = 8 / 16 = 0.5.
  → These scales cannot be written per-token; they are written ONCE per page.

To keep the same full_dim layout (and thus same page bytes), K scales for the
per-channel layout must be stored at the *page* granularity, not per token.
Physical approach: the K scale region holds 8 uint8 values PER PAGE (not
per token). The INT4-KIVI cache tensor shape places the `full_dim` axis at the
token level:
  shape = (num_pages, 2, page_tokens, num_kv_heads, full_dim)
But only full pages have meaningful K scales; the hot (partial) page stores K
in BF16 and is NOT in this tensor.

Re-running the budget with this understanding:
  K data per page per head        = page_tokens × head_size/2 = 16 × 64 = 1024 B
  K scales per page per head      = head_size/16 × 1 B        =   8 × 1 =    8 B
  V data per page per head        = page_tokens × head_size/2 = 1024 B
  V scales per page per head      = page_tokens × head_size/16 × 1 B
                                  = 16 × 8 × 1                =  128 B
  Total per page per head         = 1024 + 8 + 1024 + 128     = 2184 B

  BF16 baseline per page per head = page_tokens × head_size × 2
                                  = 16 × 128 × 2              = 4096 B

  Compression ratio (without hot page) = 4096 / 2184 ≈ 1.875×
  B/element = 2184 / (16 × 128) ≈ 1.068 B/elem   ← DOES NOT MATCH NVFP4

The per-channel K layout has FEWER K scales (8 per page) and MORE V scales
(128 per page) than NVFP4. The total scale count (8 + 128 = 136) vs NVFP4
(8 + 128 = 136 ... wait, let's recount).

NVFP4 scale count per page per head:
  K scales: page_tokens × (head_size/16) = 16 × 8 = 128
  V scales: page_tokens × (head_size/16) = 16 × 8 = 128
  Total: 256 scale bytes
  Total page bytes: 2×1024 + 256 = 2304 B per head
  B/elem: 2304 / 2048 = 1.125 B/elem

INT4-KIVI scale count per page per head:
  K scales (per-channel): head_size/16 = 8 per page (not per token)
  V scales (per-token): page_tokens × head_size/16 = 128 per page
  Total: 136 scale bytes
  Total page bytes: 2×1024 + 8 + 128 = 2184 B per head
  B/elem: 2184 / 2048 = 1.066 B/elem

Per-elem budget (with hot page overhead, averaged over long sequences):
  kv_quant.py QuantizedPage.mem_bytes() (line 59-61): 
    2 * (numel//2 + scale.numel() * 2)
  For [1, 16, 128] shape: numel = 16×128 = 2048
    = 2 * (1024 + (16×8)×2) = 2 * (1024 + 256) = 2560 B per K/V pair (total K+V)
    per element (128 elements per token, 16 tokens): 2560 / 2048 = 1.25 B/elem
  kv_quant.py reports the harness uses fp16 scales (2 B/scale), not fp8/uint8
  (1 B/scale). In hardware implementation, scales are 1 byte each (uint8/fp8).

Correcting kv_quant.py's mem_bytes for 1-byte scales:
  = 2 * (1024 + (16×8)×1) = 2 * (1024 + 128) = 2304 B per K+V pair
  = 2304 / 2048 = 1.125 B/elem = 0.5625 B/elem ÷ wait...

Full budget breakdown (1-byte scales, 0.5 B INT4 data):
  NVFP4 per element: 0.5 data + (1/16) scale = 0.5 + 0.0625 = 0.5625 B/elem
  INT4-KIVI (V side, per token): same as NVFP4: 0.5 + 0.0625 = 0.5625 B/elem
  INT4-KIVI (K side, per channel): 0.5 data + (1/(16×16)) scale amortised
                                 = 0.5 + 0.0039 = 0.5039 B/elem
  K+V combined INT4-KIVI average = (0.5039 + 0.5625) / 2 = 0.5332 B/elem

Summary:
  NVFP4 (K+V average)           : 0.5625 B/elem    (verified by kv_quant.py)
  INT4-KIVI (K+V average)       : 0.5332 B/elem    (~5.4% less than NVFP4)

This means INT4-KIVI is SLIGHTLY SMALLER than NVFP4 (fewer K scales due to
per-channel blocking amortising the scale cost over the page). The full_dim
cannot be a single shared formula — K and V have different scale densities.

PRACTICAL LAYOUT for the cache tensor (NHD order, type uint8):
  shape = (num_pages, 2, page_tokens, num_kv_heads, full_dim_v)
  where full_dim_v = head_size//2 + head_size//16 = 72  (for head_dim=128)

K-side: the data region (64 bytes/token) is used fully. The "scale" region
(8 bytes/token) is used only for the FIRST token of each page — the kernel
writes 8 scale bytes into slots [0..7] of token 0's scale region and leaves
the rest. Alternatively, store K scales in a SEPARATE small tensor:
  shape = (num_pages, num_kv_heads, head_size // 16)  → (P, H, 8)

Recommended implementation: keep the main cache shape as for NVFP4
(full_dim = head_size//2 + head_size//16 per token), but write K scales into
a separate compact tensor (P, H, head_size//16). This avoids wasted scale
slots and simplifies the store kernel. The backend dequant must combine both.

MEMORY VS BF16 (Laguna: num_kv_heads=8, head_dim=128, page_tokens=16):
  BF16 page: 2 × 16 × 8 × 128 × 2 B = 65536 B
  INT4-KIVI page (data only, INT4):
    K data: 16 × 8 × 64  B = 8192 B
    V data: 16 × 8 × 64  B = 8192 B
  INT4-KIVI scales:
    K scales: 1 × 8 × 8  B =   64 B  (per page, per head, head_size//16 uint8)
    V scales: 16 × 8 × 8 B = 1024 B  (per token per head)
  Total: 8192+8192+64+1024 = 17472 B
  Ratio: 65536 / 17472 ≈ 3.75× vs BF16  (similar to NVFP4's 3.56×)

The ~5% difference from NVFP4 (3.75 vs 3.56) comes from K having far fewer
scales. Both round to "approximately 3.5× vs BF16" in practice.
```

**Step 6 — `vllm/vllm/v1/kv_cache_interface.py` — `AttentionSpec.real_page_size_bytes` and `FullAttentionSpec.real_page_size_bytes`**

Both compute the page size using `nvfp4_kv_cache_full_dim`. The cleanest
approach is to treat the INT4-KIVI page as having the *same* full_dim as NVFP4
for the shared tensor (meaning the K scale region of each token row is only
partially used), since keeping one tensor shape avoids a second allocation and
simplifies cache management:

In `AttentionSpec.real_page_size_bytes` (currently lines 167–184):
```python
@property
def real_page_size_bytes(self) -> int:
    if self.kv_quant_mode.is_nvfp4 or self.kv_quant_mode.is_int4_kivi:
        full_dim = nvfp4_kv_cache_full_dim(self.head_size)  # same formula
        return (
            2 * self.block_size * self.num_kv_heads
            * full_dim * get_dtype_size(self.dtype)
        )
    ...
```

Apply the same change to `FullAttentionSpec.real_page_size_bytes` (currently
lines 279–298) and `SlidingWindowSpec.real_page_size_bytes` (currently lines
444–461).

---

### (c) Store Kernel + Dispatch (Including Page-Fill / BF16 Hot-Page Logic)

**The core challenge**: NVFP4's store kernel (nvfp4_kv_cache_kernels.cu) runs
per-token — each token is quantized immediately, grid = `(num_tokens,)`, one
block per token (line 75: `const int64_t token_idx = blockIdx.x;`). This works
because the 16-element block is along `head_dim` — complete for every token.

INT4-KIVI K is per-channel (16-token blocks along the sequence axis). A K
page's scales cannot be computed until all 16 tokens of that page have arrived.
This requires the **freeze-at-page-fill + BF16 hot-page** protocol
(kv_quant.py `QuantizedKVLayer.append`, line 96–110):

1. Accumulate new K/V tokens into a BF16 "hot page" buffer.
2. When the hot page reaches `page_size` tokens, quantize the whole page (K
   per-channel, V per-token), write to the INT4 cache, clear the hot buffer.
3. The partial hot page always stays in BF16 and is used as-is for decode.

**Recommended approach: pure-Python / Triton store (model the turboquant_attn path):**

The turboquant store kernel (`triton_turboquant_store`) is launched from
`TurboQuantAttentionImpl.do_kv_cache_update` in Python, which has access to the
sequence state needed for the page-fill decision. INT4-KIVI should do the same.

Create `vllm/vllm/v1/attention/ops/triton_int4_kivi_store.py`:

```python
"""Triton kernel for INT4-KIVI KV cache store.

V: quantize per-token (16-channel blocks) → write immediately.
K: accumulate tokens in bf16 hot-page buffer; quantize entire page
   when page is full (16 tokens); write INT4 + per-channel scales.
"""
import torch
import triton
import triton.language as tl

BLOCK = 16      # elements per quant block
PAGE  = 16      # tokens per page (must equal CacheConfig.block_size)
QMAX  = 7       # symmetric INT4 range


@triton.jit
def _store_v_token_kernel(
    v_ptr,          # [N, H, D] bf16 input
    v_data_ptr,     # [num_pages, PAGE, H, D//2] uint8 output
    v_scale_ptr,    # [num_pages, PAGE, H, D//BLOCK] uint8 output
    slot_mapping,   # [N] int64
    N, H: tl.constexpr, D: tl.constexpr, BLOCK_SIZE: tl.constexpr,
):
    """Quantize V per-token: head_dim blocked, write immediately."""
    ...  # Implementation: read 16 bf16 elems, compute scale=absmax/7,
         # clamp+round to [-7,7], pack two nibbles per byte, write.

@triton.jit
def _quantize_k_page_kernel(
    k_hot_ptr,      # [1, PAGE, H, D] bf16 hot page
    k_data_ptr,     # target data region
    k_scale_ptr,    # target scale region (compact, per channel per page)
    H: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
):
    """Quantize one full K page (per-channel INT4).
    
    For each channel c in [0, D):
        Extract 16 tokens of channel c → compute MSE scale → quantize.
    Scales: [H, D//BLOCK] = [H, 8] per page (for D=128).
    """
    ...


def triton_int4_kivi_store_v(
    value: torch.Tensor,     # [N, H, D] bf16
    v_cache: torch.Tensor,   # [num_pages, PAGE, H, full_dim] uint8
    slot_mapping: torch.Tensor,
    head_size: int,
    block_size: int,
) -> None:
    """Store V tokens immediately (per-token, head-dim blocks)."""
    ...


def triton_int4_kivi_freeze_k_page(
    k_hot: torch.Tensor,     # [1, PAGE, H, D] bf16 — full page ready
    k_cache: torch.Tensor,   # [num_pages, PAGE, H, full_dim] uint8
    k_scale_compact: torch.Tensor,  # [num_pages, H, D//BLOCK] uint8
    page_idx: int,
    head_size: int,
) -> None:
    """Freeze one full K page from bf16 hot buffer into INT4 cache."""
    ...
```

**Step 7 — Cache dispatch for `int4_kivi` in `cache_kernels.cu` (line 771)**

This is a CUDA kernel file and dispatching is done via the string `kv_cache_dtype`.
The NVFP4 path at line 771 calls `reshape_and_cache_nvfp4_dispatch` which does
the per-token V store. For INT4-KIVI, the **V store** can go here (token-level),
but the **K freeze** is page-level and must be called from the Python backend.

Add after the NVFP4 block (line 787):

```cpp
if (kv_cache_dtype == "int4_kivi") {
    // V side: quantize per token (head_dim blocks), write immediately.
    // K side: accumulate in hot buffer (Python side); this kernel only
    //         writes V. K is frozen by a separate triton_int4_kivi_freeze_k_page
    //         call from the Python backend when a page is complete.
    extern void reshape_and_cache_int4_kivi_v_dispatch(
        torch::stable::Tensor& value,
        torch::stable::Tensor& value_cache,
        torch::stable::Tensor& slot_mapping);
    reshape_and_cache_int4_kivi_v_dispatch(value, value_cache, slot_mapping);
    return;
}
```

Alternatively — and simpler during prototyping — keep the entire store in the
Triton Python backend and skip `cache_kernels.cu` for this dtype. The turboquant
path already follows this approach (turboquant_attn.py:do_kv_cache_update calls
`triton_turboquant_store` directly, bypassing `reshape_and_cache_flash`).

**Recommended**: follow the turboquant pattern — do NOT go through
`reshape_and_cache_flash` for `int4_kivi`. The backend's `do_kv_cache_update`
does everything.

**Step 8 — Backend state: BF16 hot-page buffers**

The INT4-KIVI backend must maintain per-request hot-page state. In vLLM V1,
the model runner drives `do_kv_cache_update` once per step for the active
tokens. The backend needs:

```python
# In Int4KiviAttentionImpl.__init__:
# Maps request_id → [H, tokens_in_hot, D] bf16 tensor
self._k_hot_buffers: dict[str, torch.Tensor] = {}
self._k_hot_token_counts: dict[str, int] = {}
```

In `do_kv_cache_update(key, value, kv_cache, slot_mapping)`:

```python
# 1. Store V immediately (per-token, head-dim blocks).
triton_int4_kivi_store_v(value, kv_cache[:, 1], slot_mapping, ...)

# 2. Accumulate K in hot buffer. When hot buffer hits page_size tokens,
#    compute the page_idx from slot_mapping, freeze the page.
for req_id, token_range in enumerate_requests(slot_mapping):
    hot_k = concat_hot_k(self._k_hot_buffers[req_id], key[token_range])
    self._k_hot_buffers[req_id] = hot_k
    while hot_k.shape[1] >= self.page_size:
        page_tokens = hot_k[:, :self.page_size]
        page_idx = slot_mapping[...] // self.page_size  # from slot_mapping
        triton_int4_kivi_freeze_k_page(
            page_tokens, kv_cache[:, 0], k_scale_compact, page_idx, ...
        )
        self._k_hot_buffers[req_id] = hot_k[:, self.page_size:]
```

The K scale compact tensor `k_scale_compact` (shape `[num_pages, H, D//BLOCK]`,
uint8) must be a layer-level parameter (allocated alongside `kv_cache` or
stored in the `kv_cache` tensor itself using the per-token layout wasting the
per-token K scale slots).

**Simplest option**: reuse the K scale region in the main cache tensor (NVFP4
shape); write K scales into token-0's scale slots only, and have the dequant
kernel read from token-0 for all tokens in the page. The wasted scale bytes for
tokens 1–15 are small (7 × 8 = 56 bytes per head per page vs 1024 data bytes).

---

### (d) Backend Dequant-Read

Create `vllm/vllm/v1/attention/backends/int4_kivi_attn.py`, modelled on
`turboquant_attn.py`. Key differences:

1. **Cache shape** = NVFP4 shape `(num_blocks, 2, block_size, num_kv_heads,
   full_dim)` with uint8 dtype. The turboquant backend uses
   `(num_blocks, block_size, num_kv_heads, slot_size)` without the leading 2;
   INT4-KIVI matches NVFP4's separate K/V convention.

2. **Prefill**: for first-chunk prefills (all KV in batch), run flash_attn on
   the raw BF16 K/V (no quantization needed). For continuation prefills with
   cached K/V, dequant the frozen pages (INT4 → BF16) and concatenate with the
   hot page, then run flash_attn. Identical logic to
   `TurboQuantAttentionImpl._continuation_prefill` (turboquant_attn.py:712).

3. **Decode**: dequant K and V from INT4 → BF16 in the Triton decode kernel,
   then run flash_attn (or a Triton decode kernel that reads INT4 directly).
   Minimum viable implementation: dequant all K/V pages + hot page → BF16 →
   pass to flash_attn_varlen. This is O(seq_len) bandwidth per decode step,
   acceptable since decode is already bandwidth-bound.

Create `vllm/vllm/v1/attention/ops/triton_int4_kivi_decode.py`:

```python
@triton.jit
def _int4_kivi_dequant_kv_kernel(
    kv_cache_ptr,          # [num_pages, 2, PAGE, H, full_dim] uint8
    k_hot_ptr,             # [1, hot_len, H, D] bf16
    block_table_ptr,       # [B, max_blocks] int32
    seq_lens_ptr,          # [B] int32
    k_out_ptr,             # [B, H, max_seq, D] bf16 output
    v_out_ptr,
    ...
):
    """Dequant K (per-channel pages + bf16 hot) and V (per-token pages)."""
    page_idx = block_table[req, pos // PAGE]
    
    # V dequant: per-token head-dim blocks (same as NVFP4 read)
    v_int4 = load nibbles from kv_cache[page_idx, 1, pos_in_page, head, :]
    v_scale = load uint8 scale, convert to bf16
    v_out = v_int4 * v_scale  # per 16-element group
    
    # K dequant: per-channel, scale from token-0 of page
    k_int4 = load nibbles from kv_cache[page_idx, 0, pos_in_page, head, c:c+16]
    k_scale = load uint8 scale from kv_cache[page_idx, 0, 0, head, data_dim + c//16]
    k_out[c:c+16] = k_int4 * k_scale
```

**Backend class skeleton**:

```python
class Int4KiviAttentionBackend(AttentionBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["int4_kivi"]

    @staticmethod
    def get_name() -> str:
        return "INT4_KIVI"

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "int4_kivi"

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ) -> tuple[int, ...]:
        from vllm.utils.torch_utils import int4_kivi_kv_cache_full_dim
        full_dim = int4_kivi_kv_cache_full_dim(head_size)
        return (num_blocks, 2, block_size, num_kv_heads, full_dim)
```

**Step 9 — Register the backend**

In `vllm/vllm/v1/attention/backends/registry.py` (line 87), add after TURBOQUANT:

```python
INT4_KIVI = "vllm.v1.attention.backends.int4_kivi_attn.Int4KiviAttentionBackend"
```

In `vllm/vllm/platforms/cuda.py` (lines 132–147), add `INT4_KIVI` to the CUDA
backend priority lists (after TURBOQUANT, lower priority since it requires an
explicit opt-in):

```python
return [
    AttentionBackendEnum.FLASHINFER,
    AttentionBackendEnum.FLASH_ATTN,
    AttentionBackendEnum.TRITON_ATTN,
    AttentionBackendEnum.FLEX_ATTENTION,
    AttentionBackendEnum.TURBOQUANT,
    AttentionBackendEnum.INT4_KIVI,    # <-- add
]
```

**Step 10 — Platform page-size computation**

`vllm/vllm/platforms/interface.py` (line 565) has a special branch for
`turboquant_*` dtypes computing a non-standard page size via `TQFullAttentionSpec`.
INT4-KIVI uses the standard NVFP4-shaped `FullAttentionSpec` (same full_dim
formula), so it falls through to the existing `else` branch (line 601–608) —
but only after `get_kv_quant_mode("int4_kivi")` returns `INT4_KIVI` and the
`real_page_size_bytes` override (Step 6) is in place.

No extra branch needed in interface.py, because `FullAttentionSpec` with
`kv_quant_mode = INT4_KIVI` will call `real_page_size_bytes` which (after
Step 6) handles it via the `is_int4_kivi` property.

---

### (e) Tests to Add

1. **`tests/unit/test_int4_kivi_cache_shape.py`** — verify
   `int4_kivi_kv_cache_full_dim(128) == 72`, verify page byte counts match
   the budget math above, verify `AttentionSpec.real_page_size_bytes` equals
   the NVFP4 value for the same head_size.

2. **`tests/unit/test_int4_kivi_store_dequant.py`** — roundtrip test:
   create synthetic BF16 K/V with known values; store via
   `triton_int4_kivi_store_v` (V) and `triton_int4_kivi_freeze_k_page` (K);
   dequant via `_int4_kivi_dequant_kv_kernel`; check RMSE < threshold from
   `kv_quant.py.measure_page_error`. This validates kernel parity with the
   Python reference.

3. **`tests/unit/test_int4_kivi_hot_page.py`** — test the BF16 hot-page
   protocol: insert 14 tokens (< PAGE), verify K is still in BF16 hot buffer;
   insert 2 more, verify page freeze fires and K is now in INT4 cache. Verify
   that partial hot page comes back verbatim from `get_kv()`.

4. **`tests/unit/test_int4_kivi_config.py`** — verify `"int4_kivi"` passes
   `CacheDType` Literal validation; verify `get_kv_quant_mode("int4_kivi")`
   returns `KVQuantMode.INT4_KIVI`; verify `is_quantized_kv_cache("int4_kivi")`
   is True.

5. **`tests/e2e/test_int4_kivi_vs_bf16.py`** — run Laguna-XS.2 forward pass
   with `kv_cache_dtype="int4_kivi"` and compare top-1 accuracy against BF16
   baseline. Acceptance criterion: top-1 agreement ≥ 98% (from PROBLEM.md
   Finding 3, short-context protocol).

---

## 3. Exact Page Byte-Budget Math

For Laguna-XS.2: `num_key_value_heads=8, head_dim=128, page_tokens=16`.

```
Element counts per page, per KV head:
  Tokens × channels = 16 × 128 = 2048 elements (K or V)

INT4 data (packed nibbles, 2 per uint8 byte):
  K data: 2048 × 0.5 B = 1024 B
  V data: 2048 × 0.5 B = 1024 B

Scales (uint8, 1 B each):
  V scales: per-token, per 16-element head-dim block
    = 16 tokens × (128/16) blocks/token = 16 × 8 = 128 scales → 128 B
  K scales: per-channel (per 16-token page block)
    = 1 page × (128/16) channel blocks/page = 8 scales → 8 B
    NOTE: written once per page, stored in token-0's scale slot in the
    shared cache tensor (8 scale bytes, reused for all 16 tokens by the
    dequant kernel).

Page bytes (INT4-KIVI):
  Per head: 1024 + 8 + 1024 + 128 = 2184 B
  All 8 heads: 2184 × 8 = 17472 B
  K+V combined: 17472 B

BF16 baseline:
  Per head: 16 × 128 × 2 B × 2 (K+V) = 8192 B
  All 8 heads: 8192 × 8 = 65536 B

Compression ratio: 65536 / 17472 ≈ 3.75× vs BF16

B/element (averaged K+V): 17472 B / (2 × 8 × 2048 elems) ≈ 0.534 B/elem

NVFP4 baseline (for comparison):
  K scales: 16 × 8 = 128 per head (per-token, per head-dim block)
  V scales: 128 per head
  Total: 2×1024 + 256 = 2304 B per head
  B/element: 2304 / 4096 = 0.5625 B/elem  ← kv_quant.py's reported value
  Compression vs BF16: 2 × 8192 / 2304 ≈ 7.1× per head ... 

Rechecking kv_quant.py (line 59-61, QuantizedPage.mem_bytes):
  2 * (numel//2 + scale.numel() * 2)  ← scale is fp16 here, 2 B/scale
  numel = 16 × 128 = 2048; scale shape = [16, 8] → scale.numel() = 128
  = 2 * (1024 + 128*2) = 2 * (1024 + 256) = 2560 B for K+V per head
  This is the Python harness with fp16 scales; hardware uses fp8/uint8:
  With 1-byte scales: 2 * (1024 + 128*1) = 2 * 1152 = 2304 B per head
  = same as NVFP4: 0.5625 B/elem
  (kv_quant.py uses headdim-block layout for its scales, not per-channel)

INT4-KIVI per-channel K hardware budget (1-byte scales):
  K: 1024 data + 8 scales = 1032 B per head
  V: 1024 data + 128 scales = 1152 B per head (same as NVFP4 V side)
  Total: 2184 B per head
  B/elem: 2184 / (2 × 2048) = 0.533 B/elem

Conclusion (corrected):
  Per-channel K has ONE scale per channel per 16-token page = 128
  scales/page/head — equal to V (16 tokens x head_size/16 = 128) and to
  each NVFP4 side. So INT4-KIVI memory == NVFP4 memory: 0.5625 B/elem
  with 1-byte (e4m3) scales = 3.56x vs BF16, exactly as PROBLEM.md and
  kv_quant.py state. (The Triton prototype stores fp16 scales -> 0.625
  B/elem = 3.2x; switch to e4m3 scales to hit 3.56x.)
```

---

## 4. Open Risks and Divergences

### 4.1 Per-channel-K streaming: stale hot-page on request preemption

If a request is preempted mid-page (hot_k has < 16 tokens), the BF16 hot
buffer must be saved and restored with the request. vLLM V1's KV cache manager
manages block tables but does NOT save the hot-page buffer — this is not
present in any existing dtype. The INT4-KIVI backend must either:

- (a) Quantize the partial hot page at preemption time using an absmax scale
  (losing the MSE optimality but maintaining correctness), or
- (b) Store the BF16 hot page in a reserved-BF16 cache region alongside the
  INT4 pages (adds a fixed overhead of at most `page_size × H × D × 2` B per
  inflight request).

Option (b) is cleanest but requires a new per-request allocatable scratch
buffer, which vLLM V1's block allocator does not currently support.

### 4.2 No native FP4 MMA — compute cost of INT4 dequant

INT4-KIVI forgoes Blackwell's `wgmma.mma_async` on FP4 inputs. The tradeoff
accepted from PROBLEM.md §"Why this needs a software INT4 path": decode
attention is memory-bandwidth bound; INT4 dequant (integer multiply → float)
is a handful of ALU ops per element and does not become the bottleneck at the
memory-bandwidth ceiling. However, benchmarking is required to confirm this
does not regress on Blackwell (SM100) compared to NVFP4 hardware decode.

### 4.3 K scale layout: token-0 reuse vs compact tensor

Storing K scales in a separate compact tensor
(`[num_pages, H, head_size//16]`, uint8) is cleaner, but requires a second
tensor allocation per layer and changes the cache allocation API. Storing K
scales in token-0's scale slot of the shared tensor (wasting 7×8 = 56 bytes
per head per page) avoids API changes but requires the dequant kernel to know
to always load from token-0's slot. Either works; the compact tensor is
recommended for production but token-0 reuse is acceptable for prototype.

### 4.4 Prefix caching incompatibility with partial hot pages

vLLM's prefix caching uses block hashes to share KV blocks across requests. A
BF16 hot page cannot be shared (its hash changes with each token). Only frozen
INT4 pages are prefix-cache-eligible. The backend must mark hot pages as
non-cacheable. This is already how turboquant handles the same issue (no
per-page caching while the page is being built).

### 4.5 CudaGraph capture with mutable hot-page state

The hot-page buffer is request-specific state that changes at every decode step.
CUDA Graphs replay a static sequence of kernel launches with pre-captured
tensor pointers. If the hot-page buffer is allocated per-request (dict[req_id]),
it changes shape across steps and cannot be captured. The turboquant backend
handles this by using `WorkspaceManager` for shared scratch buffers
(turboquant_attn.py:746, `current_workspace_manager().get_simultaneous()`).
INT4-KIVI should do the same: allocate a fixed-size workspace large enough for
the maximum batch's worth of hot pages.

### 4.6 MSE vs absmax calibration at page fill

`kv_quant.py._mse_optimal_scale` runs a 32-point grid search (O(32 × page_size
× head_dim) FLOPs per page). At large batch sizes with many pages filling
simultaneously, this may add measurable latency. Absmax is ~5% worse quality
(from PROBLEM.md §Finding 1) but eliminates the search. The Triton store kernel
should expose a `use_mse: bool` flag with absmax as the fast default and MSE as
opt-in.

### 4.7 V's headdim-blocking gives equal quality to per-channel for V

PROBLEM.md Finding 2 confirms V is well-behaved: INT4 per-token (headdim
blocks) is within ~2% of per-channel for V. INT4-KIVI's V side therefore uses
the same block geometry as NVFP4, and V can be quantized per-token immediately.
This is a safe assumption for Laguna; should be re-validated for models with
strong V outliers (e.g. models with outlier-prone V projections).

### 4.8 This forgoes turboquant's Hadamard rotation

TurboQuant applies a Hadamard rotation to K before quantization to smooth
outliers. INT4-KIVI's per-channel blocking inherently isolates outliers (each
channel's 16-token block is near-uniform by construction), so the rotation is
unnecessary — this is the mechanism behind the +25% RMSE win.

---

## Appendix: Symbol Cross-Reference

| Symbol | File | Line |
|--------|------|-------|
| `CacheDType` Literal | `vllm/config/cache.py` | 19 |
| `STR_DTYPE_TO_TORCH_DTYPE` | `vllm/utils/torch_utils.py` | 32 |
| `is_quantized_kv_cache` | `vllm/utils/torch_utils.py` | 76 |
| `nvfp4_kv_cache_full_dim` | `vllm/utils/torch_utils.py` | 415 |
| `nvfp4_kv_cache_split_views` | `vllm/utils/torch_utils.py` | 472 |
| `KVQuantMode` | `vllm/v1/kv_cache_interface.py` | 32 |
| `get_kv_quant_mode` | `vllm/v1/kv_cache_interface.py` | 59 |
| `AttentionSpec.real_page_size_bytes` | `vllm/v1/kv_cache_interface.py` | 167 |
| `FullAttentionSpec.real_page_size_bytes` | `vllm/v1/kv_cache_interface.py` | 279 |
| `SlidingWindowSpec.real_page_size_bytes` | `vllm/v1/kv_cache_interface.py` | 444 |
| `TQFullAttentionSpec` | `vllm/v1/kv_cache_interface.py` | 311 |
| `FlashInferBackend.get_kv_cache_shape` nvfp4 branch | `vllm/v1/attention/backends/flashinfer.py` | 364 |
| `FlashInferBackend.supported_kv_cache_dtypes` | `vllm/v1/attention/backends/flashinfer.py` | 328 |
| `TurboQuantAttentionBackend.supported_kv_cache_dtypes` | `vllm/v1/attention/backends/turboquant_attn.py` | 100 |
| `TurboQuantAttentionBackend.supports_kv_cache_dtype` | `vllm/v1/attention/backends/turboquant_attn.py` | 167 |
| `TurboQuantAttentionImpl.do_kv_cache_update` | `vllm/v1/attention/backends/turboquant_attn.py` | 363 |
| `TurboQuantAttentionImpl._continuation_prefill` | `vllm/v1/attention/backends/turboquant_attn.py` | 712 |
| `AttentionBackendEnum.TURBOQUANT` | `vllm/v1/attention/backends/registry.py` | 87 |
| `get_attn_backend` → `get_attn_backend_cls` | `vllm/v1/attention/selector.py` | 54 |
| Platform `get_attn_backend_cls` | `vllm/platforms/cuda.py` | 293 |
| CUDA priority list with TURBOQUANT | `vllm/platforms/cuda.py` | 132 |
| `turboquant_*` branch in page-size calc | `vllm/platforms/interface.py` | 565 |
| `reshape_and_cache_flash` nvfp4 dispatch | `csrc/libtorch_stable/cache_kernels.cu` | 771 |
| NVFP4 store kernel (per-token grid) | `csrc/libtorch_stable/nvfp4_kv_cache_kernels.cu` | 49, 75 |
| `kv_quant.py` hot-page append logic | `kv_quant.py` | 96–110 |
| `kv_quant.py` QuantizedPage mem_bytes | `kv_quant.py` | 59–61 |
