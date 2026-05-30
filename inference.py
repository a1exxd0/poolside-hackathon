"""
Shared inference helpers for Laguna XS.2.

Provides model loading, prompt encoding, generation (baseline + dynamic),
and output post-processing. Imported by benchmark.py and dynamic_inference.py.
"""

import os
import re
import time

import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer

from boundary_detector import detect_boundaries
from dynamic_tokenizer import (
    apply_dynamic_tokenization,
    apply_dynamic_tokenization_bpe,
    EmbeddingCache,
)

load_dotenv()

MODEL_ID = "poolside/Laguna-XS.2"


def strip_think_block(text: str) -> str:
    """Remove <think>...</think> reasoning content from model output.

    Laguna prepends <think> to the generation prompt. <think> is a special
    token skipped by decode, but </think> stays. This strips the dangling
    closing tag and any full think blocks re-opened in the output.
    """
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'^\s*</think>', '', text)
    return text.strip()


def load_model_and_tokenizer():
    print(f"Loading {MODEL_ID}...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        token=os.getenv("HF_TOKEN"),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16,
        device_map="auto",
        token=os.getenv("HF_TOKEN"),
    )
    model.eval()
    return model, tokenizer


def encode_chat_prompt(system: str, user: str, tokenizer, model) -> torch.Tensor:
    """Build a system+user chat prompt; return a 1-D input_ids tensor."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    result = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(result, torch.Tensor):
        result = result["input_ids"]
    return result.squeeze(0).to(model.device)


def encode_user_prompt(prompt: str, tokenizer, model) -> torch.Tensor:
    """Build a user-only chat prompt; return a 1-D input_ids tensor."""
    messages = [{"role": "user", "content": prompt}]
    result = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if not isinstance(result, torch.Tensor):
        result = result["input_ids"]
    return result.squeeze(0).to(model.device)


class TTFTTimer(BaseStreamer):
    """Records time-to-first-token during model.generate()."""

    def __init__(self):
        self._t0: float | None = None
        self._ttft: float | None = None

    def start(self, t0: float):
        self._t0 = t0

    def put(self, value):
        if self._ttft is None and self._t0 is not None:
            self._ttft = time.perf_counter() - self._t0

    def end(self):
        pass

    @property
    def ttft(self) -> float:
        return self._ttft or 0.0


def _peak_kv_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1024 ** 2


def baseline_generate(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    max_new_tokens: int,
) -> tuple:
    """Standard generation (no dynamic tokenization).

    Returns (text, elapsed_s, gen_tokens, ttft_s, peak_kv_mb).
    """
    timer = TTFTTimer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_before = _peak_kv_mb()
    t0 = time.perf_counter()
    timer.start(t0)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=timer,
        )
    elapsed = time.perf_counter() - t0
    peak_mb = _peak_kv_mb() - mem_before
    generated = output_ids[0][input_ids.shape[0]:]
    text = strip_think_block(tokenizer.decode(generated, skip_special_tokens=True))
    return text, elapsed, len(generated), timer.ttft, peak_mb


def dynamic_generate(
    input_ids: torch.Tensor,
    model,
    tokenizer,
    method: str,
    cache: EmbeddingCache,
    max_new_tokens: int,
    num_merges: int | None = None,
    sample_merges: bool = False,
) -> tuple:
    """Dynamic-tokenization generation: merging + FVT + generate.

    method="bpe" uses the batch-level BPE-style algorithm from Feher et al. 2025
    (§3.1).  Legacy methods "whitespace", "entropy", "unigram" use the
    boundary-detection path from Nawrot et al. 2023.

    Timing starts before merging so comparisons against baseline_generate are
    apples-to-apples (all overhead included).

    Returns (text, merged_len, elapsed_s, gen_tokens, ttft_s, peak_kv_mb, boundary_time_s).
    boundary_time_s covers only the merging + embedding step.
    """
    t0 = time.perf_counter()

    embed_table = model.model.embed_tokens.weight

    if method == "bpe":
        inputs_embeds_list, _segs, _m = apply_dynamic_tokenization_bpe(
            [input_ids], tokenizer, embed_table,
            m=num_merges, sample_merges=sample_merges, cache=cache,
        )
        inputs_embeds = inputs_embeds_list[0]
    else:
        boundaries = detect_boundaries(input_ids, method=method, tokenizer=tokenizer, model=model)
        inputs_embeds, _segments = apply_dynamic_tokenization(input_ids, boundaries, embed_table)

    merged_len = inputs_embeds.shape[0]

    t_generate = time.perf_counter()
    boundary_time_s = t_generate - t0

    timer = TTFTTimer()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    mem_before = _peak_kv_mb()
    timer.start(t_generate)
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds.unsqueeze(0),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=timer,
        )
    elapsed = time.perf_counter() - t0
    peak_mb = _peak_kv_mb() - mem_before
    text = strip_think_block(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    return text, merged_len, elapsed, len(output_ids[0]), timer.ttft, peak_mb, boundary_time_s
