# TriAttention

Implementation of **arXiv:2604.04921v1** — *"TriAttention: Efficient Long Reasoning with
Trigonometric KV Compression"* — for optimizing inference of the LagunaXS.2 model.

TriAttention is a **KV-cache eviction** method. Queries/keys concentrate around stable complex
centres in *pre-RoPE* space, so each cached key's future importance can be predicted from a
trigonometric series plus a norm term, and the cache pruned to a fixed budget during decoding.

## Method (per layer, per kv-head)

Every `β` generated tokens, prune the KV cache to budget `B`, keeping the top-`B` keys by score
plus `sink` leading tokens:

```
S_trig(k) = mean_{δ∈D} Σ_f |E[q_f]|·|k_f|·cos(ω_f·(p_q+δ) + arg E[q_f] − arg k_postRoPE_f)
S_norm(k) = Σ_f (1 − R_f)·E[|q_f|]·|k_f|         R_f = |E[q_f]| / E[|q_f|]
score(k)  = S_trig(k) + S_norm(k)               D = {2^0, …, 2^16}
```
GQA: z-score each query head's scores, then take the max across the group. `E[q_f]`, `E[|q_f|]`,
`R_f` and the dominant bands (top-K by `C_f = E[|q_f|]·E[|k_f|]`) are estimated offline.

**Key trick:** RoPE preserves band magnitude and folds the key position into its angle, so scoring
reads `|k_f|` and `arg k_f` straight off the cached post-RoPE keys — no key-position tracking.

## Layout

```
triattention/
  rope.py          complex-band primitives (rotate_half convention)
  calibration.py   offline pre-RoPE Q/K stats via q_proj/k_proj forward hooks
  scoring.py       per_head_scores / score_keys (S_trig + S_norm, GQA aggregation)
  generate.py      _compress_layer eviction + compression-aware greedy decode loop
tests/             RoPE invariants, logit decomposition, scoring, eviction (pytest)
scripts/
  _common.py       model loader + calibration corpus
  validate_local.py  calibrate → baseline vs compressed generation comparison
```

## Setup & run

```bash
uv sync                       # Python 3.12 venv; on a CUDA node uv installs the CUDA torch wheel
uv run pytest -q              # 9 unit tests (no model download)
uv run python -m scripts.validate_local --budget 256 --beta 64 --max-new 400
```

`validate_local.py` defaults to `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B` (Qwen2, GQA 12q/2kv) —
a small reasoning model for sanity-checking. Pass `--model` to point at LagunaXS.2 once available.

## Status

- ✅ Algorithm, calibration, scoring, eviction, decode loop, unit tests.
- ⏳ End-to-end `validate_local.py` run (do this first on the GPU node).
- ⛔ Real run: needs the LagunaXS.2 HF repo ID + A100 80GB. Then AIME25 / MATH500 benchmarks.

Assumes a Llama/Qwen2-style decoder (`model.model.layers[i].self_attn.{q_proj,k_proj}`,
rotate_half RoPE, full attention). Verify LagunaXS.2 matches or adapt the capture hooks.
Greedy decoding, batch size 1.
