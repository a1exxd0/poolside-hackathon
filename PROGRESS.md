# INT4-KIVI fused decode — benchmarks & direction

Custom vLLM INT4-KIVI KV-cache decode on Laguna-XS.2 (40 layers, 48 q / 8 kv heads =
GQA group 6, head_dim 128, 256-expert MoE / 8 active), B300.
Outer worktree branch `kv-quant-speedup`; vllm submodule branch `int4-kivi-speedup`
(commit `b9bf08cb0`). Pre-speedup A/B baseline = submodule parent `530247234`.

---

## 1. Benchmark

### Setup (applies to every command)
```bash
# Always: vLLM venv, CUDA 12.8, run from /tmp (avoid vllm/ package shadowing).
cd /tmp
ENV="CUDA_HOME=/usr/local/cuda-12.8 VLLM_USE_FLASHINFER_SAMPLER=0"
PY=/home/alex/poolside-hackathon-kv-quant/.venv-vllm/bin/python
S=/home/alex/poolside-hackathon-kv-quant/scripts
```
**Before trusting ANY timing:** confirm the GPU is idle — a co-tenant silently corrupts
every number (bf16-flash bounces, store shows multi-ms). Correctness is unaffected.
```bash
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
```
Never run a sweep concurrently with a foreground bench.

### Correctness (run first; gate speed work on these)
```bash
env $ENV $PY $S/validate_paged_decode.py     # fused decode == dense gather+attend -> ALL PASS
env $ENV $PY $S/validate_store_equiv.py       # store bytes vs golden -> STORE BIT-IDENTICAL
#   (first run with no golden saves it to /tmp/int4_store_golden.pt; --save to refresh)
```

### Kernel microbench
```bash
env $ENV $PY $S/bench_paged_decode.py    # fused vs old dense-dequant fallback (speedup x)
env $ENV $PY $S/bench_quant_vs_flash.py  # int4 read/store vs bf16 FlashAttention == THE gap
env $ENV $PY $S/sweep_decode.py          # launch-param sweep (BLOCK_N/warps/stages/waves)
```
Tuning knobs are env-overridable: `VLLM_INT4_DECODE_{BLOCK_N,WARPS,STAGES,WAVES,SPLIT,
MAX_SPLIT}`. Current defaults BLOCK_N=64, warps=4, stages=3, waves=4.

### End-to-end serving (run once per dtype: `KVD=auto` = bf16 ceiling, `KVD=int4_kivi`)
```bash
# Batched throughput + executed HumanEval pass@1 (llm.chat, all prompts at once -> B~20):
env $ENV KVD=int4_kivi N=20 PREFIX_TOKENS=12000 MAXNEW=256 $PY $S/longctx_code_serving.py
# Single-stream latency (max_num_seqs=1 -> every decode step is batch-1):
env $ENV KVD=int4_kivi PREFIX_TOKENS=12000 MAXNEW=256 R=8 $PY $S/decode_latency_serving.py
```

### A/B the speedup vs the pre-speedup kernel
```bash
cd /home/alex/poolside-hackathon-kv-quant
git -C vllm show 530247234:vllm/v1/attention/ops/triton_int4_kivi.py \
    > /tmp/triton_int4_kivi_OLD.py
cp /tmp/triton_int4_kivi_OLD.py vllm/vllm/v1/attention/ops/triton_int4_kivi.py   # -> OLD
#   ...run any bench above...
git -C vllm checkout -- vllm/v1/attention/ops/triton_int4_kivi.py               # -> restore NEW
```
Toggle the fused path off (force the dense fallback) with `VLLM_INT4_NO_FUSED_DECODE=1`.

### Current measured numbers (B300, clean GPU, NEW kernel; logs in `results/`)
Correctness: `validate_paged_decode` ALL PASS (fused==dense, max|Δ|≈2e-3, bit-exact L=1);
`validate_store_equiv` STORE BIT-IDENTICAL.

Fused vs old dense-dequant fallback (`bench_paged_decode.py`): **9.4–11.9×** across
B=1..32, ctx 4k–32k.

Decode-attention read vs bf16 FlashAttention (`bench_quant_vs_flash.py`, one step):
| B | ctx | bf16 FA ms | int4 read ms | read× | (pre-speedup) |
|--:|----:|------:|------:|:--:|:--:|
| 1 | 4k  | 0.027 | 0.050 | 1.9× | 6.8× |
| 1 | 12k | 0.027 | 0.115 | 4.3× | 17.8× |
| 1 | 32k | 0.035 | 0.279 | 8.0× | 35.3× |
| 8 | 12k | 0.068 | 1.038 | 15.2× | 17.7× |
| 32| 4k  | 0.091 | 1.262 | 13.9× | 16.4× |
Per-token store: 0.024 ms (B=1) → 0.069 ms (B=32), sync-free.

Batched HumanEval serving (`longctx_code_serving.py`, N=20, 12k long regime):
| KV | short pass@1 / gen | long pass@1 / gen |
|--:|:--:|:--:|
| bf16 (auto) | 20/20 · 29s | 20/20 · **22s** |
| int4 NEW | 19/20 · 35s | 18/20 · **53s** |
| int4 OLD | 19/20 · 31s | 16/20 · **53s** |

Single-stream latency (`decode_latency_serving.py`, max_num_seqs=1, 12k, 256 tok, R=8 median):
| KV | ms/tok | tok/s | vs bf16 |
|--:|--:|--:|:--:|
| bf16 (auto) | 22.5 | 44.4 | — |
| int4 NEW | 37.4 | 26.8 | 1.66× |
| int4 OLD | 39.9 | 25.1 | 1.77× |

---

## 2. Direction of improvement

State of the gap, by regime — the numbers above locate where work remains; this section
names the bottlenecks, it does not propose solutions.

- **Kernel, batch-1: closed.** Decode-attention read is 1.9–8× of bf16 FlashAttention
  (was 6.8–35×). This is the regime the bf16-tensor-core change targeted.
- **Kernel, large batch (B≥8): open, dominant.** The int4 read is still ~14–17× of bf16
  FA and barely moved from the pre-speedup kernel. This is the largest remaining kernel
  gap. Two structural causes are visible in the geometry: GQA group = 6 forces the QK/PV
  `tl.dot` to pad M to 16 (10/16 tensor-core rows unused), and the int4→bf16 unpack is
  ALU-bound rather than bandwidth-bound (it moves 3.2× fewer bytes yet runs slower).
- **End-to-end, single-stream latency: improved, not eliminated.** int4 decode 39.9 → 37.4
  ms/tok (1.07× NEW vs OLD); gap to bf16 1.77× → 1.66×. The int4 e2e penalty (+14.8 ms/tok
  over bf16) is far larger than the decode-attention compute difference alone — i.e. most
  of the single-stream e2e gap is per-step int4-backend overhead, not the attention math.
- **End-to-end, batched throughput: unchanged.** int4 NEW ≈ OLD (~53 s). At B≈20 the read
  sits in the still-slow large-batch regime, and on this 40-layer MoE decode-attention is a
  small fraction of each step (expert FFN + projections + sampling, all shared/unquantized,
  dominate), so attention-only changes cannot move batched wall-clock much. The win at this
  scale is the 3.2× KV-cache memory (longer context / more concurrent streams), not latency.
- **CUDA graphs: unblocked, not enabled.** The decode path is now sync-free, but the
  metadata builder is still `AttentionCGSupport.NEVER`; graph capture has not been turned on
  or measured.
