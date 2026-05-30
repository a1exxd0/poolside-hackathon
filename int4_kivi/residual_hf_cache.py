"""Residual-window INT4-KIVI KV cache (long-context quality lever).

This file *only ADDs* to the validated package -- it never modifies
``hf_cache.py`` / ``cache.py`` / the kernels.  It reuses ``_Int4KiviLayer``'s
update/dequant machinery verbatim and changes exactly one policy: how many
recent tokens are kept in **bf16** instead of being frozen to INT4.

Motivation (PROGRESS.md "Long-context quality" future-work item)
---------------------------------------------------------------
The base ``Int4KiviCache`` keeps only the trailing ``< 16`` tokens (the partial
"hot page") in bf16 and quantizes everything else to INT4.  At long context that
means the *most recent* tokens -- the ones attention weights concentrate on --
take a 4-bit round-trip.  KIVI's own ablations keep a short bf16 **residual
window** of the last R tokens lossless; doing so recovers most of the
long-context quality lost to KV quant at a tiny, bounded memory cost
(R * H * D * 2 bytes per K and V, independent of sequence length).

This layer freezes a 16-token page to INT4 only once it falls *entirely* outside
the trailing ``residual`` tokens, so at all times the last ``residual`` (plus up
to 15 partial) tokens are exact bf16.  With ``residual == 0`` the behaviour is
bit-identical to ``_Int4KiviLayer`` (only the ``< 16`` partial page stays bf16),
which makes a clean A/B: residual 0 == the old path, residual R == the lever on.
"""

from __future__ import annotations

from .cache import BLOCK
from .hf_cache import Int4KiviCache, _Int4KiviLayer


class _ResidualInt4KiviLayer(_Int4KiviLayer):
    """``_Int4KiviLayer`` that keeps the trailing ``residual`` tokens in bf16.

    Only ``__init__`` (to carry ``residual``) and ``_freeze_pages`` (the policy
    that decides which completed pages become INT4) differ from the base layer;
    everything else -- ``update``, ``_dequant_all``, sliding-window handling,
    memory accounting -- is inherited unchanged.
    """

    def __init__(self, *args, residual: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        # Round up to a whole number of pages: we freeze in BLOCK-sized chunks,
        # so the effective bf16 window is at least ``residual`` tokens.
        self.residual = max(int(residual), 0)

    def _freeze_pages(self) -> None:
        """Freeze completed 16-token pages that lie *entirely* before the
        trailing ``residual`` window; keep the rest (incl. the partial tail) bf16.

        With ``residual == 0`` this is exactly the base policy: freeze all but the
        ``< BLOCK`` partial page.
        """
        n = self._hot_len()
        # Tokens we are allowed to push to INT4 = everything older than the
        # residual window, rounded down to a whole number of 16-token pages.
        freezable = n - self.residual
        if freezable <= 0:
            return
        n_full = (freezable // BLOCK) * BLOCK
        if n_full == 0:
            return
        from .cache import store_kivi  # local import: keep module import light

        k_pages = self.hot_k[:, :n_full].contiguous()
        v_pages = self.hot_v[:, :n_full].contiguous()
        chunk = store_kivi(k_pages, v_pages, self.k_calib, self.v_calib)
        self.frozen.append(chunk)
        self._frozen_tokens += n_full
        self.hot_k = self.hot_k[:, n_full:].contiguous()
        self.hot_v = self.hot_v[:, n_full:].contiguous()


class ResidualInt4KiviCache(Int4KiviCache):
    """``Int4KiviCache`` with a configurable bf16 residual window.

        cache = ResidualInt4KiviCache(config=model.config, residual=128)
        model.generate(..., past_key_values=cache)

    ``residual == 0`` reproduces ``Int4KiviCache`` exactly.
    """

    def __init__(self, config, k_calib: str = "mse", v_calib: str = "mse",
                 residual: int = 0):
        # Reproduce Int4KiviCache's per-layer sliding/full construction, but
        # build _ResidualInt4KiviLayer carrying the residual window.  (We bypass
        # the base __init__ to swap the layer class; the parsing below mirrors it
        # one-for-one.)
        from transformers.cache_utils import Cache

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
                _ResidualInt4KiviLayer(
                    k_calib=k_calib,
                    v_calib=v_calib,
                    is_sliding=is_sliding,
                    sliding_window=sliding_window if is_sliding else None,
                    residual=residual,
                )
            )
        # Skip Int4KiviCache.__init__ (it would rebuild non-residual layers);
        # go straight to the transformers Cache container.
        Cache.__init__(self, layers=layers)
        self.residual = residual
