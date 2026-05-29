# Project memory — TriAttention implementation

> Handoff notes for whoever picks this up (e.g. on the GPU node). Mirrors the working memory
> from the dev session that created this repo.

**Goal:** Implement arXiv:2604.04921v1 — *"TriAttention: Efficient Long Reasoning with
Trigonometric KV Compression"* — to optimize inference for the **LagunaXS.2** model (80 GB
unquantized). Weights come from a **Hugging Face repo** (ID not yet provided). Real benchmark
run is planned on a **rented cloud A100 80 GB** ("the other node").

## Hardware reality
- Original dev machine: Apple M3 Pro, **18 GB unified RAM**, no CUDA — CANNOT hold the 80 GB
  model (even INT4 ≈ 20-24 GB exceeds 18 GB total). Used only for building + small-model validation.
- Local env: `uv` project, **Python 3.12** venv (torch lacks 3.14 wheels), torch 2.12 + MPS
  working, transformers 5.9. On a CUDA node, `uv sync` pulls the CUDA torch wheel.

## What TriAttention does (paper findings)
- KV-cache **eviction** method (not a new model). Every β=128 generated tokens, prune cache to
  budget B (default 2048; 512 for shorter seqs), keeping top-B keys by importance + sink tokens.
- Key insight: Q/K **concentrate in pre-RoPE space** around stable complex centres. Score a key's
  future importance via a trig series + norm term:
  - `S_trig(k) = mean_{δ∈D} Σ_f |E[q_f]|·|k_f|·cos(ω_f(p_q+δ) + arg E[q_f] − arg k_f)`,  D={2^0..2^16}
  - `S_norm(k) = Σ_f (1−R_f)·E[|q_f|]·|k_f|`,  R_f = |E[q_f]| / E[|q_f|]  (mean resultant length)
  - GQA: z-score each query head's scores, max across the group → per-(kv-head, key) score.
  - Dominant bands: pick top-K (K=2) by C_f = E[|q_f|]·E[|k_f|].
- Claims: 2.5× throughput OR 10.7× KV-mem reduction at matched accuracy (AIME25 40.8%, MATH500 68.4%).
- Eval on A100 80 GB; 24 GB RTX 4090 feasible with INT4.

## Key implementation insight (important)
RoPE is a per-band rotation that **preserves magnitude** and adds `p_k·ω_f` to the band angle.
So `|k_f|` and the position term can be read straight from **cached post-RoPE keys** — scoring
needs NO separate key-position tracking, only the current query's absolute position. The trig
score rewrites to `cos(ω_f·p_q + arg E[q_f] − arg k_postRoPE_f)`. Verified by unit tests
(reconstructs true attention logits at r>0.99).

## Current state (DONE)
- Package `triattention/`: `rope.py` (complex bands), `calibration.py` (pre-RoPE Q/K capture via
  q_proj/k_proj forward hooks + stats), `scoring.py` (`per_head_scores`, `score_keys`),
  `generate.py` (`_compress_layer` per-kv-head top-B eviction + manual greedy decode loop that
  passes explicit `cache_position` so eviction can't desync RoPE).
- Tests `tests/` (9 passing): RoPE invariants, logit decomposition, trig-score reconstruction,
  GQA shapes, eviction (sink + top-scored retained, values evicted in lockstep).
- Scripts: `scripts/_common.py` (model loader; local val model = DeepSeek-R1-Distill-Qwen-1.5B,
  Qwen2 GQA 12q/2kv), `scripts/validate_local.py` (calibrate → baseline vs compressed generate).

## NOT done / next steps
1. **Never ran** `validate_local.py` end-to-end (stopped before model download to move to the GPU
   node). First action on the node: `uv sync && uv run pytest -q`, then
   `uv run python -m scripts.validate_local`.
2. Phase 4 (BLOCKED): need **HF repo ID for LagunaXS.2** + A100 access. Then provisioning script,
   real-model calibration, AIME25/MATH500 benchmarks. Needs CUDA torch wheels on the node.
3. Decoder assumes Llama/Qwen2 layout (`model.model.layers[i].self_attn.{q_proj,k_proj}`,
   rotate_half RoPE, full attention). Verify LagunaXS.2 matches or adapt the capture hooks.
4. Only greedy decoding implemented; batch size 1.
