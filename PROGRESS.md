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

**Speed:** int4 decode is ~3× slower than bf16 at 12k (long-regime gen 65s vs 22s)
because `_dequant_and_attend` re-dequantizes the **whole** context to dense bf16 every
decode step. Correct but not optimized — see "Future work".

## Run recipes (see VLLM_SETUP.md)
```
cd /tmp   # run from NON-vllm cwd to avoid package shadowing
CUDA_HOME=/usr/local/cuda-12.8 VLLM_USE_FLASHINFER_SAMPLER=0 KVD=int4_kivi TRIALS=20 \
  /home/alex/poolside-hackathon-kv-quant/.venv-vllm/bin/python \
  /home/alex/poolside-hackathon-kv-quant/scripts/needle_serving.py
# coding bench (run once per dtype, KVD=auto then KVD=int4_kivi):
KVD=int4_kivi N=20 PREFIX_TOKENS=12000 ... longctx_code_serving.py
```
Logs: `/tmp/needle_srv_{bf16,int4_fixed}.log`,
`/tmp/longctx_code_{bf16,int4}.log`; JSON: `/tmp/{needle_serving,longctx_code_serving}_*.json`.

## Future work (not blocking)
- **Decode speed:** replace dense whole-context dequant with a fused INT4 paged-attention
  decode (dequant-in-kernel, no (B,H,max_seq,D) materialization). Biggest win available.
- **Long-context quality:** the 12k HumanEval gap (80 vs 100) is the lever — try keeping a
  short bf16 recent-token tail (like `int4_kivi/hf_cache.py`) or asymmetric/zero-point K.
- ✅ **Grid y-dim (DONE, branch `kv-quant-grid-y-dim`):** `_gather_dequant_kernel` grid was
  `(B, max_seq, H)`, putting the position axis on grid.y whose CUDA limit is 65535 — any
  gather with `max_seq>65535` (max_model_len past 64k) failed to launch with CUDA "invalid
  argument". Fixed by reordering to `(max_seq, B, H)`: the position axis now rides grid.x
  (limit ~2**31); B and H stay on y/z where they fit. No tiling needed. Regression test:
  `vllm/tests/v1/attention/test_int4_kivi_grid.py` (gather at max_seq=70000 launches +
  round-trips; old grid raised). Verified old code raises, new code passes.
