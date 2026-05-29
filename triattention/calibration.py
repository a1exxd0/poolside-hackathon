"""Offline calibration of pre-RoPE Q/K statistics.

The paper's central observation is that, in *pre-RoPE* space, each query/key
frequency band concentrates tightly around a fixed complex centre that is
stable across positions.  We estimate those centres once, offline, from a small
calibration corpus.

Capture strategy
----------------
``apply_rotary_pos_emb`` runs *inside* the attention forward, so the cleanest
way to observe pre-RoPE vectors is to hook the linear projections that feed it:
the outputs of ``self_attn.q_proj`` / ``self_attn.k_proj`` are exactly the
query/key states *before* RoPE.  We register forward hooks on those modules,
run the corpus through the model, and accumulate per-(layer, head, band)
statistics.

Per layer we record, for every query head ``g`` and band ``f``:

* ``Eq``      = E[z^q_f]            (complex centre of the query band)
* ``Eq_norm`` = E[|z^q_f|]          (mean magnitude)
* ``R``       = |Eq| / Eq_norm      (mean resultant length, concentration in [0,1])

and for every kv head, ``Ek_norm = E[|z^k_f|]`` (used only to pick dominant
bands via ``C_f = Eq_norm * Ek_norm``).
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

import torch
from torch import nn

from .rope import rope_frequencies, to_complex_bands


@dataclasses.dataclass
class LayerStats:
    omega: torch.Tensor       # [d2]            RoPE band frequencies
    Eq: torch.Tensor          # [n_q, d2]       complex query centre
    Eq_norm: torch.Tensor     # [n_q, d2]       E[|q_f|]
    R: torch.Tensor           # [n_q, d2]       mean resultant length
    Ek_norm: torch.Tensor     # [n_kv, d2]      E[|k_f|]
    dominant_bands: torch.Tensor  # [n_q, K] long  top-K band indices per query head

    def to(self, device) -> "LayerStats":
        return LayerStats(**{f.name: getattr(self, f.name).to(device) for f in dataclasses.fields(self)})


@dataclasses.dataclass
class CalibrationStats:
    layers: list[LayerStats]
    head_dim: int
    n_q_heads: int
    n_kv_heads: int
    group_size: int
    theta: float
    n_dominant: int
    num_tokens: int

    def to(self, device) -> "CalibrationStats":
        out = dataclasses.replace(self, layers=[l.to(device) for l in self.layers])
        return out

    def save(self, path: str) -> None:
        blob = {
            "meta": {k: getattr(self, k) for k in
                     ("head_dim", "n_q_heads", "n_kv_heads", "group_size", "theta", "n_dominant", "num_tokens")},
            "layers": [dataclasses.asdict(l) for l in self.layers],
        }
        torch.save(blob, path)

    @staticmethod
    def load(path: str, map_location="cpu") -> "CalibrationStats":
        blob = torch.load(path, map_location=map_location, weights_only=False)
        layers = [LayerStats(**l) for l in blob["layers"]]
        return CalibrationStats(layers=layers, **blob["meta"])


class _BandAccumulator:
    """Streaming sum of complex bands and magnitudes for one projection."""

    def __init__(self, n_heads: int, head_dim: int, device):
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.sum_complex = torch.zeros(n_heads, head_dim // 2, dtype=torch.complex64, device=device)
        self.sum_abs = torch.zeros(n_heads, head_dim // 2, dtype=torch.float32, device=device)
        self.count = 0

    def add(self, proj_out: torch.Tensor) -> None:
        # proj_out: [B, T, n_heads * head_dim]
        b, t, _ = proj_out.shape
        x = proj_out.reshape(b, t, self.n_heads, self.head_dim)
        z = to_complex_bands(x)                       # [B, T, n_heads, d2]
        self.sum_complex += z.sum(dim=(0, 1))         # [n_heads, d2]
        self.sum_abs += z.abs().sum(dim=(0, 1))
        self.count += b * t


def collect_calibration(
    model: nn.Module,
    tokenizer,
    texts: Iterable[str],
    *,
    max_length: int = 2048,
    n_dominant: int = 2,
    device=None,
) -> CalibrationStats:
    """Run ``texts`` through ``model`` and return calibrated :class:`CalibrationStats`."""
    device = device or next(model.parameters()).device
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n_q = cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", n_q)
    theta = float(getattr(cfg, "rope_theta", 10000.0))

    decoder = model.model  # Qwen2/Llama: .model.layers
    n_layers = len(decoder.layers)
    q_acc = [_BandAccumulator(n_q, head_dim, device) for _ in range(n_layers)]
    k_acc = [_BandAccumulator(n_kv, head_dim, device) for _ in range(n_layers)]

    handles = []
    for i, layer in enumerate(decoder.layers):
        attn = layer.self_attn
        handles.append(attn.q_proj.register_forward_hook(
            lambda mod, inp, out, i=i: q_acc[i].add(out.detach())))
        handles.append(attn.k_proj.register_forward_hook(
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

    omega = rope_frequencies(head_dim, theta, device=device)
    layers: list[LayerStats] = []
    for i in range(n_layers):
        cnt_q = max(q_acc[i].count, 1)
        cnt_k = max(k_acc[i].count, 1)
        Eq = q_acc[i].sum_complex / cnt_q            # [n_q, d2] complex
        Eq_norm = q_acc[i].sum_abs / cnt_q           # [n_q, d2]
        Ek_norm = k_acc[i].sum_abs / cnt_k           # [n_kv, d2]
        R = (Eq.abs() / Eq_norm.clamp_min(1e-9)).clamp(0.0, 1.0)

        # dominant bands per query head by C_f = E|q_f| * E|k_f| (broadcast kv->q)
        kv_for_q = torch.arange(n_q, device=device) // (n_q // n_kv)
        C = Eq_norm * Ek_norm[kv_for_q]              # [n_q, d2]
        k = min(n_dominant, C.shape[-1])
        dominant = C.topk(k, dim=-1).indices         # [n_q, K]

        layers.append(LayerStats(
            omega=omega.clone(), Eq=Eq, Eq_norm=Eq_norm, R=R, Ek_norm=Ek_norm,
            dominant_bands=dominant,
        ))

    total_tokens = sum(a.count for a in q_acc) // max(n_layers, 1)
    return CalibrationStats(
        layers=layers, head_dim=head_dim, n_q_heads=n_q, n_kv_heads=n_kv,
        group_size=n_q // n_kv, theta=theta, n_dominant=n_dominant, num_tokens=total_tokens,
    )
