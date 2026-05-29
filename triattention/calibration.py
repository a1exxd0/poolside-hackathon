"""Offline calibration of pre-RoPE Q/K statistics.

The paper's central observation is that, in *pre-RoPE* space, each query/key
frequency band concentrates tightly around a fixed complex centre that is
stable across positions.  We estimate those centres once, offline, from a small
calibration corpus.

Capture strategy
----------------
``apply_rotary_pos_emb`` runs *inside* the attention forward, so we observe the
query/key vectors at the last point before RoPE is applied.

* **Llama/Qwen2**: that point is the output of ``self_attn.q_proj`` /
  ``self_attn.k_proj``.
* **Laguna**: those projections are followed by a per-head ``q_norm`` /
  ``k_norm`` (RMSNorm over ``head_dim``) *before* RoPE, so the relevant pre-RoPE
  vector is the **norm output**, not the projection output.  We hook
  ``q_norm`` / ``k_norm`` when present and fall back to the projections
  otherwise.

We also only calibrate the layers TriAttention will actually compress: with
mixed attention (Laguna) the sliding-window layers are bounded by the cache and
left alone, so we calibrate the **full-attention** layers only.

Per layer we record, for every query head ``g`` and rotated band ``f``:

* ``Eq``      = E[z^q_f]            (complex centre of the query band)
* ``Eq_norm`` = E[|z^q_f|]          (mean magnitude)
* ``R``       = |Eq| / Eq_norm      (mean resultant length, concentration in [0,1])

for every kv head, ``Ek_norm = E[|z^k_f|]`` (to pick dominant bands via
``C_f = Eq_norm * Ek_norm``), and -- with partial RoPE -- the signed mean of the
non-rotated (position-independent) query tail, ``Eq_pass = E[q_pass]``.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import torch
from torch import nn

from .rope import rope_frequencies, to_complex_bands, pass_through_dims


@dataclasses.dataclass
class LayerStats:
    omega: torch.Tensor       # [d2]            RoPE band frequencies (rotated bands only)
    Eq: torch.Tensor          # [n_q, d2]       complex query centre
    Eq_norm: torch.Tensor     # [n_q, d2]       E[|q_f|]
    R: torch.Tensor           # [n_q, d2]       mean resultant length
    Ek_norm: torch.Tensor     # [n_kv, d2]      E[|k_f|]
    dominant_bands: torch.Tensor  # [n_q, K] long  top-K band indices per query head
    # E[q] over non-rotated dims (empty when RoPE is full); number of rotated dims.
    Eq_pass: torch.Tensor = dataclasses.field(default_factory=lambda: torch.empty(0))
    rotary_dim: int | None = None  # None == full head_dim

    def to(self, device) -> "LayerStats":
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            out[f.name] = v.to(device) if torch.is_tensor(v) else v
        return LayerStats(**out)


@dataclasses.dataclass
class CalibrationStats:
    layers: list[LayerStats]
    layer_indices: list[int]      # model layer index for each entry in ``layers``
    head_dim: int
    n_q_heads: int
    n_kv_heads: int
    group_size: int
    theta: float
    n_dominant: int
    num_tokens: int

    def to(self, device) -> "CalibrationStats":
        return dataclasses.replace(self, layers=[l.to(device) for l in self.layers])

    def save(self, path: str) -> None:
        blob = {
            "meta": {k: getattr(self, k) for k in
                     ("layer_indices", "head_dim", "n_q_heads", "n_kv_heads",
                      "group_size", "theta", "n_dominant", "num_tokens")},
            "layers": [dataclasses.asdict(l) for l in self.layers],
        }
        torch.save(blob, path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "CalibrationStats":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        layers = [LayerStats(**l) for l in blob["layers"]]
        return CalibrationStats(layers=layers, **blob["meta"])


class _BandAccumulator:
    """Streaming sums of complex rotated bands + non-rotated tail for one projection."""

    def __init__(self, n_heads: int, head_dim: int, rotary_dim: int, device):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.rotary_dim = rotary_dim
        self.n_pass = head_dim - rotary_dim
        d2 = rotary_dim // 2
        self.sum_complex = torch.zeros(n_heads, d2, dtype=torch.complex64, device=device)
        self.sum_abs = torch.zeros(n_heads, d2, dtype=torch.float32, device=device)
        self.sum_pass = torch.zeros(n_heads, self.n_pass, dtype=torch.float32, device=device)
        self.count = 0

    def add(self, out: torch.Tensor) -> None:
        # ``out`` is either [B, T, n_heads*head_dim] (proj) or [B, T, n_heads, head_dim] (norm).
        if out.dim() == 3:
            b, t, _ = out.shape
            x = out.reshape(b, t, self.n_heads, self.head_dim)
        else:
            b, t = out.shape[0], out.shape[1]
            x = out
        z = to_complex_bands(x, self.rotary_dim)      # [B, T, n_heads, d2]
        self.sum_complex += z.sum(dim=(0, 1))         # [n_heads, d2]
        self.sum_abs += z.abs().sum(dim=(0, 1))
        if self.n_pass:
            self.sum_pass += pass_through_dims(x, self.rotary_dim).float().sum(dim=(0, 1))
        self.count += b * t


def _geometry(model):
    """Read attention geometry from the model: per-layer head counts, layer types,
    rotated-dim count, and the true RoPE inverse frequencies (handles YaRN)."""
    cfg = model.config
    decoder = model.model
    n_layers = len(decoder.layers)
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

    layer_types = getattr(cfg, "layer_types", None) or ["full_attention"] * n_layers
    per_layer_heads = getattr(cfg, "num_attention_heads_per_layer", None) \
        or [cfg.num_attention_heads] * n_layers

    rot = getattr(decoder, "rotary_emb", None)
    # Prefer the model's own inv_freq buffer for full-attention layers (carries YaRN scaling).
    inv_freq = None
    for name in ("full_attention_inv_freq", "inv_freq"):
        if rot is not None and hasattr(rot, name):
            inv_freq = getattr(rot, name).detach().float()
            break
    if inv_freq is None:
        inv_freq = rope_frequencies(head_dim, float(getattr(cfg, "rope_theta", 10000.0)))
    rotary_dim = int(2 * inv_freq.numel())
    return decoder, layer_types, per_layer_heads, head_dim, inv_freq, rotary_dim


def collect_calibration(
    model: nn.Module,
    tokenizer,
    texts: Iterable[str],
    *,
    max_length: int = 2048,
    n_dominant: int = 2,
    device=None,
) -> CalibrationStats:
    """Run ``texts`` through ``model`` and return calibrated :class:`CalibrationStats`
    for the full-attention layers only."""
    device = device or next(model.parameters()).device
    cfg = model.config
    decoder, layer_types, per_layer_heads, head_dim, inv_freq, rotary_dim = _geometry(model)
    inv_freq = inv_freq.to(device)
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    theta = float(getattr(cfg, "rope_theta", 10000.0))

    target = [i for i, t in enumerate(layer_types) if t == "full_attention"]
    q_acc: dict[int, _BandAccumulator] = {}
    k_acc: dict[int, _BandAccumulator] = {}
    handles = []
    for i in target:
        attn = decoder.layers[i].self_attn
        n_q = per_layer_heads[i]
        q_acc[i] = _BandAccumulator(n_q, head_dim, rotary_dim, device)
        k_acc[i] = _BandAccumulator(n_kv, head_dim, rotary_dim, device)
        # Hook the per-head norm (true pre-RoPE vector) when present, else the projection.
        q_mod = getattr(attn, "q_norm", None) or attn.q_proj
        k_mod = getattr(attn, "k_norm", None) or attn.k_proj
        handles.append(q_mod.register_forward_hook(
            lambda mod, inp, out, i=i: q_acc[i].add(out.detach())))
        handles.append(k_mod.register_forward_hook(
            lambda mod, inp, out, i=i: k_acc[i].add(out.detach())))

    try:
        model.eval()
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                enc = {k: v.to(device) for k, v in enc.items()}
                model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()

    layers: list[LayerStats] = []
    for i in target:
        cnt_q = max(q_acc[i].count, 1)
        cnt_k = max(k_acc[i].count, 1)
        n_q = per_layer_heads[i]
        Eq = q_acc[i].sum_complex / cnt_q            # [n_q, d2] complex
        Eq_norm = q_acc[i].sum_abs / cnt_q           # [n_q, d2]
        Ek_norm = k_acc[i].sum_abs / cnt_k           # [n_kv, d2]
        Eq_pass = q_acc[i].sum_pass / cnt_q          # [n_q, n_pass]
        R = (Eq.abs() / Eq_norm.clamp_min(1e-9)).clamp(0.0, 1.0)

        # dominant bands per query head by C_f = E|q_f| * E|k_f| (broadcast kv->q)
        kv_for_q = torch.arange(n_q, device=device) // (n_q // n_kv)
        C = Eq_norm * Ek_norm[kv_for_q]              # [n_q, d2]
        k = min(n_dominant, C.shape[-1])
        dominant = C.topk(k, dim=-1).indices         # [n_q, K]

        layers.append(LayerStats(
            omega=inv_freq.clone(), Eq=Eq, Eq_norm=Eq_norm, R=R, Ek_norm=Ek_norm,
            dominant_bands=dominant, Eq_pass=Eq_pass, rotary_dim=rotary_dim,
        ))

    n_q_full = per_layer_heads[target[0]] if target else cfg.num_attention_heads
    total_tokens = (sum(a.count for a in q_acc.values()) // max(len(target), 1)) if target else 0
    return CalibrationStats(
        layers=layers, layer_indices=target, head_dim=head_dim, n_q_heads=n_q_full,
        n_kv_heads=n_kv, group_size=n_q_full // n_kv, theta=theta,
        n_dominant=n_dominant, num_tokens=total_tokens,
    )
