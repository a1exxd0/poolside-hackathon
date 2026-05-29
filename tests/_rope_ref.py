"""Reference RoPE (HF rotate_half convention) for tests."""

import torch

from triattention.rope import rope_frequencies


def apply_rope(x: torch.Tensor, pos: int, theta: float = 10000.0) -> torch.Tensor:
    """Apply RoPE at absolute position ``pos`` to ``x`` of shape ``[..., d]``."""
    d = x.shape[-1]
    omega = rope_frequencies(d, theta, device=x.device, dtype=torch.float32)  # [d/2]
    ang = pos * omega
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1)
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1)
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin
