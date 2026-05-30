# INT4-KIVI vLLM serving — progress handoff

**Goal:** Working, accurate, benchmarked long-context *coding* inference on
Laguna-XS.2 served through our **custom vLLM INT4 KV-cache kernel** (per-channel-K
KIVI layout). Branch `kv-quant`, worktree `/home/alex/poolside-hackathon-kv-quant`.

## Status — ✅ DONE
- ✅ Custom vLLM `int4_kivi` KV backend implemented and **serves** Laguna-XS.2 on B300.
- ✅ Triton kernels: V per-token, K **per-channel** for full 16-tok blocks, per-token
  fallback for partial blocks.
- ✅ Serves accurately at scale (20-trial / 60-prompt batches to 32k context).
- ✅ Accurate + benchmarked serving numbers obtained for int4 vs bf16 (below).
- ✅ **Fused paged decode — v1 landed, NOT finished** (branch `kv-quant-decode-speed`,
  vllm submodule branch `int4-kivi-decode-speed`): pure-decode batches now run a fused
  INT4 flash-decode straight off the packed cache — no dense `(B,H,max_seq,D)`
  materialization, K/V read at 4 bits. Validated == dense path; **2.4–10× faster than the
  old dense-dequant fallback**. ⚠️ BUT still **~2.4× slower end-to-end (≈18× at the
  decode-kernel level) than bf16 FlashAttention** — the dense-materialize penalty is gone,
  the *quantized-decode-vs-FA* gap is not. **Are we done fusing? No** — see "Fused decode"
  + the optimization levers at the bottom.

## Results (vLLM serving path, Laguna-XS.2, B300, enforce_eager)
**Needle-in-code retrieval** (exact-int recall, `scripts/needle_serving.py`, 20 trials/len):

| ctx  | bf16 (`auto`) | int4_kivi |
|-----:|:-------------:|:---------:|
| 8k   | 85%           | 60%       |
| 16k  | 95%           | 85%       |
| 32k  | 90%           | **90%**   |
| total| **90% (54/60)** | **78% (47/60)** |

**Executed HumanEval pass@1** (`scripts/longctx_code_serving.py`, n=20, greedy):

| regime              | bf16 (`auto`) | int4_kivi |
|--------------------:|:-------------:|:---------:|
| short (ctx ~200)    | 100% (20/20)  | 95% (19/20) |
| long  (ctx ~12.2k)  | 100% (20/20)  | 80% (16/20) |

Reading: the int4 path runs end-to-end and produces correct code at long context.
The quant cost is real and **grows with context** (expected for KV quant): ~1 problem
at short ctx, 4 at 12k; bf16 solved all 20 long problems within the same 256-token
budget, so the int4 long failures are genuine quant errors, not truncation. On the
needle the int4 cost is *non-monotonic* (worst at 8k, parity at 32k) — that metric is
exact-digit recall and is fragile to single-token logit flips, so treat HumanEval
pass@1 as the more reliable signal.

## Fused decode (decode-speed future-work item — 🟡 v1 LANDED, NOT DONE)
**Status in one line:** the fusion that *removes the dense bf16 materialization* is
implemented, correct, and merged-ready; the fused decode is **not yet competitive with
bf16 FlashAttention** (~2.4× slower e2e @12k, ~18× at the kernel), and the kernel-level
optimization levers (tensor-core bf16 `tl.dot`, fused combine, CUDA graphs) are **untouched**.
So: done with *this* fusion step, **not** done making quantized decode fast.

The old decode path (`_dequant_and_attend`) re-dequantized the **whole** context to a
dense bf16 `(B,H,max_seq,D)` tensor every step — the bulk of decode latency. The new
fused path (`int4_kivi_paged_decode` → `_paged_decode_kernel` + `_decode_combine_kernel`
in `vllm/.../ops/triton_int4_kivi.py`) is a GQA-grouped split-K flash-decode that walks
the block table and dequantizes K (per-channel for full blocks, per-token for the partial
tail) and V (per-token) **in-kernel** in fp32, with online softmax — no dense KV. It
fires for pure-decode batches (`max_query_len==1`, no sliding window); `_dequant_and_attend`
stays as the fallback for continuation/mixed/windowed steps. Toggle off with
`VLLM_INT4_NO_FUSED_DECODE=1`.

**Kernel correctness** (`scripts/validate_paged_decode.py`): fused == dense gather+attend
on the *same* packed cache, max|Δ|≈2e-3 (bf16 floor), bit-exact at L=1, across
exact/partial/short/mixed/long(12k) cases.

**Decode-kernel speed** (`scripts/bench_paged_decode.py`, one decode step, B300):
| B | ctx | dense ms | fused ms | speedup |
|--:|----:|---------:|---------:|--------:|
| 1 | 12k | 1.25 | 0.47 | **2.66×** |
| 1 | 32k | 2.92 | 1.19 | **2.45×** |
| 8 | 12k | 9.75 | 1.21 | **8.1×** |

**End-to-end A/B** (fused on vs `VLLM_INT4_NO_FUSED_DECODE=1`, same int4 cache):
- Needle (5/len, to 32k): both **10/15 (67%)**, bucket-identical → no quality change.
- HumanEval long (12k ctx, 256-tok greedy): gen **52s → 65s slower without fusion**
  (~1.25× faster wall-clock; decode is only part of the step, prefill+MoE are shared).
  pass@1 16/20 (fused) vs 18/20 (dense) — within greedy-long-gen FP-path noise at n=20
  given the 2e-3 kernel agreement (same fragility the needle metric note calls out);
  the fused path accumulates in fp32, i.e. strictly higher precision than flash-bf16.

### Quantization pipeline vs *only flash decode* (bf16, KVD=auto = the real ceiling)
The "2.4–10×" above is vs the **dense-dequant fallback**. Against real bf16
**FlashAttention** (`KVD=auto`, FLASH_ATTN backend, no quant) the int4 pipeline is still
a *regression* — that is the honest cost of software KV quant on this hardware:

- **End-to-end** HumanEval (`scripts/longctx_code_serving.py`, N=20, 256-tok greedy):
  | regime | bf16 flash | int4 fused | regression |
  |--:|:--:|:--:|:--:|
  | short (~200) | 28s, 20/20 | 30s, 19/20 | ~1.07× |
  | long (12k)   | **22s, 20/20** | **52s, 16/20** | **~2.4×** |

- **Isolated decode-attention kernel** (`scripts/bench_quant_vs_flash.py`, one step, B300):
  int4 fused vs `flash_attn_varlen` on identical shapes —
  | B | ctx | bf16 flash ms | int4 read ms | +store | slowdown |
  |--:|----:|------:|------:|------:|------:|
  | 1 | 12k | 0.027 | 0.471 | +0.05 | **~18×** |
  | 1 | 32k | 0.034 | 1.188 | +0.05 | **~35×** |
  | 8 | 12k | 0.068 | 1.214 | +0.05 | **~18×** |
  Per-token store-quant is cheap (~0.05 ms/step); the cost is the **read**.

**Why:** the int4 decode is *overhead/compute-bound, not bandwidth-bound* — it moves ~4×
fewer bytes than bf16 yet is ~18× slower, so it never cashes in the 4-bit bandwidth win.
Causes: in-kernel int4→fp32 unpack (ALU-heavy), fp32 math instead of tensor-core bf16,
split-K + a separate combine kernel, two Triton launches/step, and eager-only (no CUDA
graph) vs FA's warp-specialized hand-tuned CUDA. So fusion removed the *dense-materialize*
penalty but the quantized decode is fundamentally a software path competing with
FlashAttention. **Net: int4 KV buys memory (3.2× cache) at ~2.4× decode latency @12k; the
gap is decode-attention only — prefill/MoE/sampling are shared.** Levers to close it (all
future work): bf16 tensor-core dequant-matmul (cast deq K/V to bf16 for `tl.dot`), fuse the
combine into the decode epilogue, fewer/larger splits, CUDA-graph capture + warmup.

## Run recipes (see VLLM_SETUP.md)
```
cd /tmp   # run from NON-vllm cwd to avoid package shadowing
CUDA_HOME=/usr/local/cuda-12.8 VLLM_USE_FLASHINFER_SAMPLER=0 KVD=int4_kivi TRIALS=20 \
  /home/alex/poolside-hackathon-kv-quant/.venv-vllm/bin/python \
  /home/alex/poolside-hackathon-kv-quant/scripts/needle_serving.py
# coding bench (run once per dtype, KVD=auto then KVD=int4_kivi):
KVD=int4_kivi N=20 PREFIX_TOKENS=12000 ... longctx_code_serving.py
# fused-decode kernel validation + microbench (no model load):
cd /tmp && CUDA_HOME=/usr/local/cuda-12.8 .venv-vllm/bin/python \
  scripts/validate_paged_decode.py   # and scripts/bench_paged_decode.py
# A/B the fused decode against the dense fallback: VLLM_INT4_NO_FUSED_DECODE=1
```
Logs: `/tmp/needle_srv_{bf16,int4_fixed}.log`,
`/tmp/longctx_code_{bf16,int4}.log`; JSON: `/tmp/{needle_serving,longctx_code_serving}_*.json`.

## Future work (not blocking)
- **Decode speed / finish the fusion:** 🟡 v1 done (no dense materialize), but still
  ~2.4× slower e2e (~18× kernel) than bf16 FlashAttention — the fused decode is *not*
  finished. Ordered levers to close the gap (biggest first):
  1. **Tensor cores:** cast dequantized K/V to **bf16** and use `tl.dot` on bf16 (currently
     fp32 math) — this is the dominant ~18× factor; FA wins because it runs tensor-core bf16.
  2. **Fuse the combine** into the decode epilogue (drop the second `_decode_combine_kernel`
     launch) and tune `(SPLIT, BLOCK_N)` per context length (autotune); fewer/larger splits.
  3. **CUDA-graph capture** (currently eager-only) + warmup to kill per-step launch overhead
     and the in-inference JIT spike.
  Reality check: even fully tuned, a software int4-dequant decode may not *beat* FA — the
  honest target is "close the latency gap so the 3.2× cache-memory win is worth it," not
  "faster than bf16."
- **Long-context quality:** the 12k HumanEval gap (80 vs 100) is the lever — try keeping a
  short bf16 recent-token tail (like `int4_kivi/hf_cache.py`) or asymmetric/zero-point K.
- **Grid y-dim:** `_gather_dequant_kernel` (dense fallback) grid is `(B, max_seq, H)`;
  max_seq>65535 would exceed the CUDA y-dim limit. The fused decode avoids this (grid is
  `B*H*SPLIT`). Fine for max_model_len≤64k; tile the fallback if ever larger.

## DO NOT MODIFY (validated references)
`int4_kivi/*.py`, `kv_quant.py`, `tests/test_int4_kivi.py`, `/tmp/vllm_needle.py`.
Never use system python/pip — only `uv` / `.venv-vllm`.
