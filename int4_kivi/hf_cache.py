"""HuggingFace-compatible INT4-KIVI KV cache for Laguna-XS.2.

Design ("store low-bit, dequant in the read path, attend in bf16", PROBLEM.md):

Each layer keeps its completed 16-token pages as INT4 (K per-channel/mse,
V per-token/mse, via ``int4_kivi.store_kivi`` / Triton) and the partial tail
(< 16 tokens) in bf16 -- the "hot page".  On ``update`` we append the new
tokens to the hot tail, freeze any pages that just completed into INT4, and
return the **full** dequantized bf16 K/V (frozen pages dequant + bf16 hot tail).
The model consumes only that return value, so attention is bit-for-bit the same
math it would run on a ``DynamicCache`` -- except the cached pages took a 4-bit
quant round-trip.

This file only ADDs to the validated package; it imports ``store_kivi`` /
``dequant_kivi`` / ``KIVICache`` and never modifies them.

Cache layout matches what Laguna's ``DynamicCache(config=...)`` returns:
per layer K/V are ``[batch, n_kv_heads, seq, head_dim]`` (batch == 1 for greedy
decode).  Laguna interleaves ``full_attention`` and ``sliding_attention``
layers; the model builds the per-layer attention mask from
``cache.get_mask_sizes(query_length, layer_idx)``, and the *sliding* layer's
contract is that it caches only the last ``sliding_window - 1`` tokens while
``update`` still returns the full (cached + new) states.  We replicate that
contract exactly (truncating the INT4/bf16 store to the trailing window) so the
mask and the returned tensor stay consistent.
"""

from __future__ import annotations

import torch
from torch import Tensor

from transformers.cache_utils import Cache, CacheLayerMixin

from .cache import BLOCK, KIVICache, dequant_kivi, store_kivi


# --------------------------------------------------------------------------- #
# per-layer INT4-KIVI store
# --------------------------------------------------------------------------- #
class _Int4KiviLayer(CacheLayerMixin):
    """One layer's INT4-KIVI store.

    Internally holds:
      * ``frozen``  : list of ``KIVICache`` chunks, each covering a whole number
                      of completed 16-token pages.
      * ``hot_k/hot_v`` : bf16 ``[H, t, D]`` tail with ``t < 16`` (the partial
                      page that is not yet quantized).
    The dequantized full K/V are reconstructed on demand in ``update``.

    ``is_sliding`` / ``sliding_window`` mirror ``DynamicSlidingWindowLayer``:
    for a sliding layer we only keep the trailing ``sliding_window - 1`` tokens.
    """

    def __init__(
        self,
        k_calib: str = "mse",
        v_calib: str = "mse",
        is_sliding: bool = False,
        sliding_window: int | None = None,
    ):
        super().__init__()
        self.k_calib = k_calib
        self.v_calib = v_calib
        self.is_sliding = is_sliding
        self.sliding_window = sliding_window

        self.frozen: list[KIVICache] = []
        self.hot_k: Tensor | None = None      # [H, t, D] bf16, t < BLOCK
        self.hot_v: Tensor | None = None
        self.cumulative_length = 0            # total tokens ever seen (sliding)
        self._frozen_tokens = 0               # tokens currently held in `frozen`

    # -- transformers Cache layer interface ---------------------------------- #
    def lazy_initialization(self, key_states: Tensor, value_states: Tensor) -> None:
        self.dtype, self.device = key_states.dtype, key_states.device
        self.is_initialized = True

    def get_max_cache_shape(self) -> int:
        return self.sliding_window if self.is_sliding else -1

    def get_seq_length(self) -> int:
        if self.is_sliding:
            return self.cumulative_length
        return self._stored_len()

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        if not self.is_sliding:
            return self.get_seq_length() + query_length, 0
        # DynamicSlidingWindowLayer semantics.
        is_full = self.cumulative_length >= self.sliding_window
        kv_offset = max(self.cumulative_length - self.sliding_window + 1, 0)
        if is_full:
            kv_length = self.sliding_window - 1 + query_length
        else:
            kv_length = self.cumulative_length + query_length
        return kv_length, kv_offset

    # -- internal helpers ---------------------------------------------------- #
    def _hot_len(self) -> int:
        return 0 if self.hot_k is None else self.hot_k.shape[1]

    def _stored_len(self) -> int:
        """Tokens physically held in this layer (frozen pages + hot tail)."""
        return self._frozen_tokens + self._hot_len()

    def _freeze_pages(self) -> None:
        """Move every completed 16-token page out of the hot tail into INT4."""
        n = self._hot_len()
        n_full = (n // BLOCK) * BLOCK
        if n_full == 0:
            return
        k_pages = self.hot_k[:, :n_full].contiguous()
        v_pages = self.hot_v[:, :n_full].contiguous()
        # store_kivi with S a multiple of BLOCK fully quantizes (empty hot tail).
        chunk = store_kivi(k_pages, v_pages, self.k_calib, self.v_calib)
        self.frozen.append(chunk)
        self._frozen_tokens += n_full
        self.hot_k = self.hot_k[:, n_full:].contiguous()
        self.hot_v = self.hot_v[:, n_full:].contiguous()

    def _evict_sliding(self) -> None:
        """Drop fully-out-of-window frozen chunks (keeps mask/offset consistent).

        We must retain at least the trailing ``sliding_window - 1`` tokens that
        ``get_mask_sizes`` promises.  Whole 16-token chunks older than that are
        dropped; partial trimming inside a chunk is unnecessary because the
        attention mask zeroes any over-retained leading tokens.
        """
        if not self.is_sliding:
            return
        keep = self.sliding_window - 1 + BLOCK  # small slack so the mask offset is covered
        while self.frozen and (self._frozen_tokens - self.frozen[0].S) >= keep:
            dropped = self.frozen.pop(0)
            self._frozen_tokens -= dropped.S

    def _dequant_all(self) -> tuple[Tensor, Tensor]:
        """Reconstruct full bf16 (K, V) = dequant(frozen pages) ++ hot tail."""
        ks: list[Tensor] = []
        vs: list[Tensor] = []
        for chunk in self.frozen:
            k, v = dequant_kivi(chunk)        # [H, chunk.S, D] bf16
            ks.append(k)
            vs.append(v)
        if self._hot_len() > 0:
            ks.append(self.hot_k)
            vs.append(self.hot_v)
        if not ks:
            # Should not happen: update always has at least the new tokens.
            empty = torch.empty(
                (self._H, 0, self._D), dtype=torch.bfloat16, device=self.device
            )
            return empty, empty
        return torch.cat(ks, dim=1), torch.cat(vs, dim=1)

    # -- the contract the model actually calls ------------------------------- #
    def update(
        self, key_states: Tensor, value_states: Tensor, *args, **kwargs
    ) -> tuple[Tensor, Tensor]:
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)

        # key/value_states: [B, H, n_new, D]; this cache supports B == 1.
        assert key_states.shape[0] == 1, "Int4KiviCache supports batch size 1"
        k_new = key_states[0].to(torch.bfloat16)   # [H, n_new, D]
        v_new = value_states[0].to(torch.bfloat16)
        self._H, self._D = k_new.shape[0], k_new.shape[2]
        n_new = k_new.shape[1]
        self.cumulative_length += n_new

        # Append to the hot tail.
        if self.hot_k is None:
            self.hot_k, self.hot_v = k_new, v_new
        else:
            self.hot_k = torch.cat([self.hot_k, k_new], dim=1)
            self.hot_v = torch.cat([self.hot_v, v_new], dim=1)

        # Freeze any newly-completed pages into INT4, then evict if sliding.
        self._freeze_pages()
        self._evict_sliding()

        # Return the full dequantized bf16 states, unsqueezed to [1, H, S, D].
        k_full, v_full = self._dequant_all()
        if self.is_sliding and self.cumulative_length >= self.sliding_window:
            # Trim to the window the mask expects: sliding_window - 1 + n_new.
            want = self.sliding_window - 1 + n_new
            if k_full.shape[1] > want:
                k_full = k_full[:, -want:]
                v_full = v_full[:, -want:]
        return k_full.unsqueeze(0), v_full.unsqueeze(0)

    # -- memory accounting --------------------------------------------------- #
    def nbytes(self) -> int:
        total = sum(c.nbytes for c in self.frozen)
        if self.hot_k is not None:
            total += self.hot_k.numel() * self.hot_k.element_size()
            total += self.hot_v.numel() * self.hot_v.element_size()
        return total

    def bf16_nbytes(self) -> int:
        return 2 * (self._stored_len() * getattr(self, "_H", 0) * getattr(self, "_D", 0)) * 2

    # -- misc Cache layer methods (greedy decode does not need these) -------- #
    def reset(self) -> None:
        self.frozen.clear()
        self.hot_k = self.hot_v = None
        self.cumulative_length = 0
        self._frozen_tokens = 0

    def reorder_cache(self, beam_idx) -> None:  # batch==1, no-op
        pass

    def crop(self, max_length: int) -> None:
        raise NotImplementedError("crop is not supported by Int4KiviCache")

    def batch_repeat_interleave(self, repeats: int) -> None:
        raise NotImplementedError("batch expansion is not supported by Int4KiviCache")

    def batch_select_indices(self, indices) -> None:
        raise NotImplementedError("batch selection is not supported by Int4KiviCache")


# --------------------------------------------------------------------------- #
# the Cache container
# --------------------------------------------------------------------------- #
class Int4KiviCache(Cache):
    """Drop-in ``past_key_values`` for ``model.generate`` on Laguna-XS.2.

    Build with the model config so per-layer sliding/full structure matches:

        cache = Int4KiviCache(config=model.config)
        model.generate(..., past_key_values=cache)
    """

    def __init__(self, config, k_calib: str = "mse", v_calib: str = "mse"):
        decoder = config.get_text_config(decoder=True)
        sliding_window = getattr(decoder, "sliding_window", None) or getattr(
            decoder, "attention_chunk_size", None
        )
        layer_types = getattr(decoder, "layer_types", None)
        if layer_types is None:
            n = decoder.num_hidden_layers
            kind = "sliding_attention" if sliding_window is not None else "full_attention"
            layer_types = [kind] * n

        layers = []
        for lt in layer_types:
            is_sliding = lt == "sliding_attention"
            layers.append(
                _Int4KiviLayer(
                    k_calib=k_calib,
                    v_calib=v_calib,
                    is_sliding=is_sliding,
                    sliding_window=sliding_window if is_sliding else None,
                )
            )
        super().__init__(layers=layers)

    # -- memory accounting over all layers ----------------------------------- #
    def nbytes(self) -> int:
        return sum(l.nbytes() for l in self.layers)

    def bf16_nbytes(self) -> int:
        return sum(l.bf16_nbytes() for l in self.layers)

    def compression_ratio_vs_bf16(self) -> float:
        return self.bf16_nbytes() / max(self.nbytes(), 1)
