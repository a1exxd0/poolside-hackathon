"""RoPE / complex-band primitives.

TriAttention reasons about queries and keys in *complex* form: for a head of
dimension ``d``, RoPE (the rotate_half convention used by Llama/Qwen2) pairs
dimension ``f`` with ``f + d/2`` and rotates that 2-D vector by ``p * omega_f``
at position ``p``.  We represent each such pair as a single complex number

    z_f = x[..., f] + i * x[..., f + d/2]            for f in [0, d/2)

Two facts that the whole method rests on:

* RoPE is a pure rotation, so ``|z_f|`` is **invariant** to position. The
  magnitude of a key's band can therefore be read straight off the cached
  (post-RoPE) keys.
* A post-RoPE key's *angle* equals its pre-RoPE angle plus ``p_k * omega_f``.
  That is exactly the positional information we need, so scoring never has to
  track key positions separately.
"""

from __future__ import annotations

import torch


def rope_frequencies(head_dim: int, theta: float = 10000.0, *, device=None, dtype=torch.float32) -> torch.Tensor:
    """Return ``omega_f = theta^(-2f/d)`` for ``f in [0, d/2)`` (shape ``[d/2]``).

    These are the same inverse frequencies HF builds for the rotary embedding.
    """
    if head_dim % 2 != 0:
        raise ValueError(f"head_dim must be even, got {head_dim}")
    f = torch.arange(0, head_dim, 2, device=device, dtype=dtype)
    return theta ** (-f / head_dim)


def to_complex_bands(x: torch.Tensor, rotary_dim: int | None = None) -> torch.Tensor:
    """Map a real head tensor ``[..., d]`` to complex bands ``[..., r/2]``.

    Uses the rotate_half pairing ``z_f = x[f] + i * x[f + r/2]`` so that a RoPE
    rotation of ``x`` corresponds to multiplying ``z`` by ``exp(i * p * omega)``.

    With **partial RoPE** (``rotary_dim < d``, e.g. Laguna's 0.5 factor) only the
    first ``rotary_dim`` dims are rotated; the trailing ``d - rotary_dim`` dims pass
    through unchanged. We form bands from the rotated block only — the pairing is
    ``(f, f + rotary_dim/2)`` for ``f in [0, rotary_dim/2)``. Pass-through dims are
    handled separately by the scorer (see :func:`pass_through_dims`).
    """
    d = x.shape[-1]
    r = d if rotary_dim is None else rotary_dim
    if r % 2 != 0:
        raise ValueError(f"rotary_dim must be even, got {r}")
    half = r // 2
    real = x[..., :half]
    imag = x[..., half:r]
    return torch.complex(real.float(), imag.float())


def pass_through_dims(x: torch.Tensor, rotary_dim: int) -> torch.Tensor:
    """Return the non-rotated (position-independent) tail ``x[..., rotary_dim:]``."""
    return x[..., rotary_dim:]
