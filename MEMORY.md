# Project memory — TriAttention on Laguna-XS.2

**Goal:** Implement arXiv:2604.04921v1 *"TriAttention: Efficient Long Reasoning with Trigonometric
KV Compression"* and run it on **`poolside/Laguna-XS.2`** (~63 GB). KV-cache eviction (not a new
model): every β tokens prune each full-attention layer to budget B (sink tokens + top-B by
importance). Paper claims 2.5× throughput / 10.7× KV-mem at matched accuracy (AIME25 40.8%,
MATH500 68.4%).

## Method (one-liner)
Q/K concentrate in pre-RoPE space. Score a key's future importance with a trig series + norm term,
read directly off cached post-RoPE keys (RoPE preserves `|k_f|`, folds position into the angle).
GQA: z-score per query head, max over group. Dominant bands = top-K by `E|q_f|·E|k_f|`.

## Environment (A100-80GB node, CUDA 13)
`uv` at `~/.local/bin` (export PATH). `uv sync` → **py3.14** venv, torch 2.12+cu130, transformers 5.9.
`uv run pytest -q` → **10 pass**. transformers 5.9 gotcha: `apply_chat_template` returns a dict —
pass `return_dict=False`.

## Code map (`triattention/`)
- `rope.py` — complex bands; `to_complex_bands(x, rotary_dim)` handles partial RoPE + `pass_through_dims`.
- `calibration.py` — hooks `q_norm`/`k_norm` (Laguna's true pre-RoPE point), per-layer head counts,
  reads model's real YaRN `inv_freq`, captures `Eq_pass`; calibrates **only full-attention layers**
  (`CalibrationStats.layer_indices`). Falls back to q_proj/k_proj for Qwen2.
- `scoring.py` — trig score on rotated bands + position-independent pass-through term `S_pass`.
- `generate.py` — `DynamicCache(config=...)` auto-bounds the 30 sliding layers; compresses only the
  10 full layers; explicit `position_ids` (Laguna desyncs otherwise); list EOS; `record_kv=` gives
  per-step KV-byte series (split full/sliding/total).
- Scripts: `validate_local.py` (Qwen2-1.5B), `validate_laguna.py`, `sweep_laguna.py`,
  `benchmark_math.py` (MATH-500: random subset, transcripts + KV-mem percentiles → `results/*.json`).

## Laguna-XS.2 facts (why it's not vanilla Qwen2)
40 layers; full-attention at idx 0,4,…,36 (10 layers, 48 q-heads, GQA group 6); 30 sliding (window
512, auto-capped by cache). head_dim 128, kv_heads 8. **Partial RoPE 0.5** (32 rotated bands + 64
pass-through). **YaRN** on full layers. q_norm/k_norm before RoPE. MoE (256 exp) — irrelevant to KV.

## Validated results
- Qwen2-1.5B: fidelity scales with budget (budget 384 → 96.2% token agreement).
- **Laguna-XS.2** (√2/Pell prompt, 700 tok): R=0.663. Full-KV peak 811 → 640 (1.27×)/320 (2.53×)/
  192 (4.22×) at budget 512/256/128; transcripts coherent at every budget. peak = budget+β (cache
  regrows between prunes). Load ~2:47, ~84 s/700-tok decode.

## In progress / next
- **MATH-500 benchmark** running: 5 random problems, configs baseline/b2048/b512/b256, max_new 2048,
  full transcripts + p50/p90/p99 KV-mem → `results/`.
- **b2048 never fires on MATH**: prompt+gen stays < 2048 tokens, so the `keys<=budget` guard skips
  every layer → b2048 ≡ baseline. Needs AIME-length generations to engage; the benchmark prints
  per-problem `peak_full_kv` to show the gap.
- Not done: AIME25; throughput timing; greedy/bs=1 only. g_proj gate correctly ignored (scales
  output, not key importance).
