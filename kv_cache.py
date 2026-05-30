"""
kv_cache.py — Hierarchical KV cache eviction driven by the hierarchy.py AST

The hierarchy.py tree (TopicNode → LeafCluster) is the inode tree.
Quest scoring walks this two-level tree at each eviction step to decide
which semantic clusters to materialise into the active KV cache.

Design:
  - hierarchy.py processes the prompt text → builds TopicNode → LeafCluster tree
  - Each LeafCluster covers a set of semantically related prompt tokens
  - Each LeafCluster is augmented with Quest min/max key metadata from real keys
  - Each TopicNode is augmented with aggregate min/max over its child clusters
  - Prefill K/V is stored immutably as the flat backing store
  - Active cache = sink ∪ quest_selected_prefill ∪ recent_generated
  - Every β decode steps: re-run Quest on the tree with the current query
  - PyramidKV: each layer gets a different budget (bottom layers more, top less)
"""

from __future__ import annotations

import bisect
import re
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

from sentence_transformers import SentenceTransformer
from hierarchy import HierarchyPipeline, HierarchyTree, SBERT_MODEL

# Minimum per-layer slack above sink+recent in the PyramidKV b_min formula.
_PYRAMID_SLACK = 16

# Loaded once per process; shared across all generate_with_hierarchy calls.
_sbert: Optional[SentenceTransformer] = None

def _get_sbert() -> SentenceTransformer:
    global _sbert
    if _sbert is None:
        _sbert = SentenceTransformer(SBERT_MODEL)
    return _sbert


# ─── Inode data structures ────────────────────────────────────────────────────

@dataclass
class LeafInode:
    """A LeafCluster augmented with Quest min/max key metadata."""
    cluster_id:      int
    token_positions: List[int]     # positions in the full prefill sequence
    key_min:         torch.Tensor  # [n_kv_heads, head_dim]
    key_max:         torch.Tensor  # [n_kv_heads, head_dim]


@dataclass
class TopicInode:
    """A TopicNode augmented with aggregate Quest metadata over its children."""
    topic_id: int
    leaf_ids: List[int]            # child cluster_ids present in leaf_inodes
    key_min:  torch.Tensor         # [n_kv_heads, head_dim] — min over all children
    key_max:  torch.Tensor         # [n_kv_heads, head_dim] — max over all children


# ─── Build inodes from the hierarchy tree ────────────────────────────────────

def build_inodes(
    tree:           HierarchyTree,
    node_token_map: Dict[int, List[int]],  # GraphNode.node_id → token positions
    keys:           torch.Tensor,           # [n_kv_heads, prefill_len, head_dim]
) -> tuple[Dict[int, LeafInode], Dict[int, TopicInode]]:
    """
    Augment every LeafCluster and TopicNode with Quest min/max key metadata.
    Uses scatter_reduce for batched per-cluster min/max; avoids creating one
    intermediate tensor per cluster.
    """
    n_kv, prefill_len, head_dim = keys.shape

    cluster_ids = list(tree.leaf_clusters.keys())
    C = len(cluster_ids)
    if C == 0:
        return {}, {}

    cid_to_dense = {cid: i for i, cid in enumerate(cluster_ids)}

    # Assign each valid token position to a dense cluster index.
    pos_to_cluster = torch.full((prefill_len,), -1, dtype=torch.long, device=keys.device)
    cluster_positions: Dict[int, List[int]] = {}  # dense_idx → sorted token positions

    for cid, cluster in tree.leaf_clusters.items():
        dense_idx = cid_to_dense[cid]
        positions = sorted({
            p
            for nid in cluster.member_node_ids
            for p in node_token_map.get(nid, [])
            if p < prefill_len
        })
        if not positions:
            continue
        cluster_positions[dense_idx] = positions
        for p in positions:
            pos_to_cluster[p] = dense_idx

    assigned_pos = (pos_to_cluster >= 0).nonzero(as_tuple=True)[0]
    if assigned_pos.numel() == 0:
        return {}, {}

    keys_assigned = keys[:, assigned_pos, :]                                  # [n_kv, n_asgn, head_dim]
    cluster_idx   = pos_to_cluster[assigned_pos]                              # [n_asgn]
    idx_expanded  = cluster_idx.view(1, -1, 1).expand(n_kv, -1, head_dim)    # [n_kv, n_asgn, head_dim]

    cluster_mins = torch.full((n_kv, C, head_dim), float("inf"),
                              device=keys.device, dtype=keys.dtype)
    cluster_maxs = torch.full((n_kv, C, head_dim), float("-inf"),
                              device=keys.device, dtype=keys.dtype)
    cluster_mins.scatter_reduce_(1, idx_expanded, keys_assigned, reduce="amin", include_self=True)
    cluster_maxs.scatter_reduce_(1, idx_expanded, keys_assigned, reduce="amax", include_self=True)

    # Leaf inodes: one per cluster that has at least one mapped position.
    leaf_inodes: Dict[int, LeafInode] = {}
    for cid in cluster_ids:
        dense_idx = cid_to_dense[cid]
        if dense_idx not in cluster_positions:
            continue
        leaf_inodes[cid] = LeafInode(
            cluster_id=cid,
            token_positions=cluster_positions[dense_idx],
            key_min=cluster_mins[:, dense_idx, :],
            key_max=cluster_maxs[:, dense_idx, :],
        )

    # Topic inodes: aggregate min/max over child leaves.
    topic_inodes: Dict[int, TopicInode] = {}
    for tid, topic in tree.topic_nodes.items():
        child_ids = [cid for cid in topic.leaf_cluster_ids if cid in leaf_inodes]
        if not child_ids:
            continue
        child_mins = torch.stack([leaf_inodes[cid].key_min for cid in child_ids])
        child_maxs = torch.stack([leaf_inodes[cid].key_max for cid in child_ids])
        topic_inodes[tid] = TopicInode(
            topic_id=tid,
            leaf_ids=child_ids,
            key_min=child_mins.min(dim=0).values,
            key_max=child_maxs.max(dim=0).values,
        )

    return leaf_inodes, topic_inodes


# ─── Two-stage Quest traversal of the AST ────────────────────────────────────

def select_by_hierarchy(
    query:        torch.Tensor,
    topic_inodes: Dict[int, TopicInode],
    leaf_inodes:  Dict[int, LeafInode],
    k_topics:     int,
    k_leaves:     int,
) -> Set[int]:
    """
    Level 1: Quest-score TopicInodes → top k_topics topics.
    Level 2: Quest-score their child LeafInodes → top k_leaves per topic.
    Returns set of token positions to load into the active cache.

    Scoring is fully vectorized per level: all topics (or leaves within a
    topic) are scored in a single tensor op rather than one call per inode.
    Falls back to direct leaf scoring for leaves not yet assigned to a topic.
    """
    covered       = {cid for t in topic_inodes.values() for cid in t.leaf_ids}
    orphan_leaves = {cid: lf for cid, lf in leaf_inodes.items() if cid not in covered}

    positions: Set[int] = set()
    # Unsqueeze once; broadcast over stacked topic/leaf tensors.
    q = query.unsqueeze(0)  # [1, n_kv, head_dim]

    if topic_inodes:
        t_ids    = list(topic_inodes.keys())
        t_mins   = torch.stack([topic_inodes[tid].key_min for tid in t_ids])  # [T, n_kv, head_dim]
        t_maxs   = torch.stack([topic_inodes[tid].key_max for tid in t_ids])
        t_scores = (q * torch.where(q > 0, t_maxs, t_mins)).sum(dim=(-2, -1))  # [T]
        _, top_t = t_scores.topk(min(k_topics, len(t_ids)))

        for i in top_t.tolist():
            topic = topic_inodes[t_ids[i]]
            avail = [cid for cid in topic.leaf_ids if cid in leaf_inodes]
            if not avail:
                continue
            l_mins   = torch.stack([leaf_inodes[cid].key_min for cid in avail])  # [L, n_kv, head_dim]
            l_maxs   = torch.stack([leaf_inodes[cid].key_max for cid in avail])
            l_scores = (q * torch.where(q > 0, l_maxs, l_mins)).sum(dim=(-2, -1))  # [L]
            _, top_l = l_scores.topk(min(k_leaves, len(avail)))
            for li in top_l.tolist():
                positions.update(leaf_inodes[avail[li]].token_positions)

    if orphan_leaves:
        o_cids   = list(orphan_leaves.keys())
        o_mins   = torch.stack([orphan_leaves[cid].key_min for cid in o_cids])
        o_maxs   = torch.stack([orphan_leaves[cid].key_max for cid in o_cids])
        o_scores = (q * torch.where(q > 0, o_maxs, o_mins)).sum(dim=(-2, -1))
        _, top_o = o_scores.topk(min(k_leaves, len(o_cids)))
        for i in top_o.tolist():
            positions.update(orphan_leaves[o_cids[i]].token_positions)

    return positions


# ─── PyramidKV per-layer budget ───────────────────────────────────────────────

def pyramid_budgets(n_layers: int, b_min: int, b_max: int) -> List[int]:
    """Layer 0 (bottom) → b_max, layer n_layers-1 (top) → b_min. Linear."""
    if n_layers == 1:
        return [b_max]
    return [
        round(b_min + (b_max - b_min) * (n_layers - 1 - l) / (n_layers - 1))
        for l in range(n_layers)
    ]


# ─── Query capture hook ───────────────────────────────────────────────────────

class _QueryCapture:
    """
    Post-hook on q_proj: captures the last-token query per layer.
    GQA groups are averaged to kv-head count so the query shape matches
    the key shape used in inode metadata.
    """

    def __init__(self):
        self.queries: Dict[int, torch.Tensor] = {}
        self._handles = []

    def install(self, model):
        cfg  = model.config
        n_q  = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_q)

        for l in range(cfg.num_hidden_layers):
            try:
                q_proj = model.model.layers[l].self_attn.q_proj
            except AttributeError:
                continue

            def make_hook(li, nq, nkv):
                def hook(module, inp, output):
                    last = output[0, -1]          # [n_q * head_dim]
                    if last.numel() % nq != 0:
                        return
                    hd = last.numel() // nq
                    q  = last.reshape(nq, hd)
                    if nq != nkv:
                        q = q.reshape(nkv, nq // nkv, hd).mean(1)
                    self.queries[li] = q.detach()
                return hook

            self._handles.append(
                q_proj.register_forward_hook(make_hook(l, n_q, n_kv))
            )

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.queries.clear()


# ─── Node → token position mapping ───────────────────────────────────────────

def build_node_token_map(
    nodes_info:  Dict[int, dict],  # pipeline.get_hierarchy()["nodes"]
    tokenizer,
    prompt_text: str,
) -> Dict[int, List[int]]:
    """
    Map each GraphNode to its token positions in the tokenized prompt.

    For fast tokenizers: uses character-level offset mapping for exact BPE
    alignment. Each chunk is located by its word-index span (chunks are built
    from consecutive words of prompt_text.split()), then token indices whose
    character midpoint falls within that span are collected.

    For slow tokenizers: falls back to cumulative re-tokenization (approximate).
    """
    if not getattr(tokenizer, "is_fast", False):
        node_token_map: Dict[int, List[int]] = {}
        cumulative = 0
        for nid in sorted(nodes_info.keys()):
            tokens = tokenizer.encode(nodes_info[nid]["text"], add_special_tokens=False)
            node_token_map[nid] = list(range(cumulative, cumulative + len(tokens)))
            cumulative += len(tokens)
        return node_token_map

    encoding = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets  = encoding["offset_mapping"]  # [(char_start, char_end), ...]

    # Character spans of each whitespace-delimited word (matches str.split()).
    word_spans  = [(m.start(), m.end()) for m in re.finditer(r"\S+", prompt_text)]
    word_starts = [ws for ws, _ in word_spans]

    # Map each token to the word containing its character midpoint.
    token_to_word: Dict[int, int] = {}
    for tok_idx, (ts, te) in enumerate(offsets):
        if ts == te:
            continue  # zero-length token (special token sentinel)
        mid = (ts + te) // 2
        wi  = bisect.bisect_right(word_starts, mid) - 1
        if 0 <= wi < len(word_spans) and mid < word_spans[wi][1]:
            token_to_word[tok_idx] = wi

    word_cursor = 0
    node_token_map = {}
    for nid in sorted(nodes_info.keys()):
        n_words        = len(nodes_info[nid]["text"].split())
        chunk_word_set = set(range(word_cursor, word_cursor + n_words))
        node_token_map[nid] = sorted(
            tok_idx for tok_idx, wi in token_to_word.items()
            if wi in chunk_word_set
        )
        word_cursor += n_words
    return node_token_map


# ─── Main generate function ───────────────────────────────────────────────────

def generate_with_hierarchy(
    model,
    tokenizer,
    input_ids:      torch.Tensor,    # [1, prompt_len]
    max_new_tokens: int = 200,
    budget:         int = 512,
    sink:           int = 4,
    recent:         int = 64,
    k_topics:       int = 3,
    k_leaves:       int = 4,
    beta:           int = 32,
    eos_token_id:   Optional[int] = None,
    verbose:        bool = False,
) -> dict:
    """
    Hierarchical KV cache generation.

    The prompt is chunked and clustered by hierarchy.py into a two-level
    semantic tree.  At every eviction step Quest scoring walks that tree
    to select which clusters to materialise.

    Active cache at all times:
      first `sink` prefill tokens  (always kept — attention sinks)
    ∪ Quest-selected prefill clusters  (hierarchy-driven)
    ∪ last `recent` generated tokens   (recency window)
    """
    from transformers import DynamicCache

    device   = input_ids.device
    model.eval()
    n_layers = model.config.num_hidden_layers

    # PyramidKV: bottom layer gets 2×budget, top layer gets budget÷2
    b_min = max(sink + recent + _PYRAMID_SLACK, budget // 2)
    b_max = budget * 2
    layer_budgets = pyramid_budgets(n_layers, b_min, b_max)

    # ── Build semantic hierarchy from prompt text ──────────────────────────
    prompt_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    pipeline    = HierarchyPipeline(sbert_instance=_get_sbert())
    for word in prompt_text.split():
        pipeline.process_word(word)
    pipeline.flush()

    hier           = pipeline.get_hierarchy()
    tree           = hier["tree"]
    node_token_map = build_node_token_map(hier["nodes"], tokenizer, prompt_text)

    # ── Install query capture hooks ───────────────────────────────────────
    qcap = _QueryCapture()
    qcap.install(model)

    # ── Prefill ───────────────────────────────────────────────────────────
    cache = DynamicCache(config=model.config)
    with torch.no_grad():
        out = model(input_ids, past_key_values=cache, use_cache=True)

    prefill_len = input_ids.size(1)
    current_pos = prefill_len

    full_attn = [l for l in range(n_layers)
                 if not getattr(cache.layers[l], "is_sliding", False)]
    sliding   = [l for l in range(n_layers) if l not in full_attn]

    if verbose:
        print(f"\n[hierarchy] Prompt: {prefill_len} tokens")
        print(f"[hierarchy] Layers — full-attention: {len(full_attn)}, "
              f"sliding-window: {len(sliding)}")
        print(f"[hierarchy] Full-attn indices (first 10): {full_attn[:10]}")

        _tree = pipeline._hierarchy.get_tree()
        print(f"\n[hierarchy] Tree — {len(_tree.leaf_clusters)} leaf clusters, "
              f"{len(_tree.topic_nodes)} topic nodes")
        nodes_info = hier["nodes"]
        for cid, cluster in sorted(_tree.leaf_clusters.items()):
            positions = sorted({
                p
                for nid in cluster.member_node_ids
                for p in node_token_map.get(nid, [])
            })
            texts = [nodes_info[nid]["text"][:40] for nid in cluster.member_node_ids
                     if nid in nodes_info]
            print(f"  Cluster {cid}: {len(cluster.member_node_ids)} nodes, "
                  f"{len(positions)} tokens [{positions[0] if positions else '?'}"
                  f"–{positions[-1] if positions else '?'}]")
            for t in texts:
                print(f"    · {t!r}")
        for tid, topic in sorted(_tree.topic_nodes.items()):
            print(f"  Topic {tid}: leaves {topic.leaf_cluster_ids}")

    # Immutable flat backing store for full-attention layers only
    prefill_keys   = {l: cache.layers[l].keys[0].clone()   for l in full_attn}
    prefill_values = {l: cache.layers[l].values[0].clone() for l in full_attn}

    n_kv     = next(iter(prefill_keys.values())).size(0)
    head_dim = next(iter(prefill_keys.values())).size(2)

    # ── Build inodes for full-attention layers only ───────────────────────
    layer_leaf_inodes:  Dict[int, Dict[int, LeafInode]]  = {}
    layer_topic_inodes: Dict[int, Dict[int, TopicInode]] = {}
    for l in full_attn:
        li, ti = build_inodes(tree, node_token_map, prefill_keys[l])
        layer_leaf_inodes[l]  = li
        layer_topic_inodes[l] = ti

    # ── Active cache reconstruction (full-attention layers only) ──────────
    def reconstruct(l: int, query: torch.Tensor,
                    gen_k: torch.Tensor, gen_v: torch.Tensor):
        budget_l     = layer_budgets[l]
        n_gen        = gen_k.size(1)
        n_recent_gen = min(recent, n_gen)

        # Fast path: everything fits — return full prefill + all gen, no eviction.
        if prefill_len + n_gen <= budget_l:
            pk = prefill_keys[l]
            pv = prefill_values[l]
            if n_gen > 0:
                pk = torch.cat([pk, gen_k], dim=1)
                pv = torch.cat([pv, gen_v], dim=1)
            if verbose:
                print(f"  [layer {l}] no eviction needed "
                      f"({prefill_len} prefill + {n_gen} gen = {prefill_len + n_gen} ≤ {budget_l})")
            return pk.unsqueeze(0), pv.unsqueeze(0)

        # Always keep: first `sink` tokens (attention sinks) and last `sink`
        # prefill tokens (chat-template boundary / start-of-answer tokens).
        sink_pos   = set(range(min(sink, prefill_len)))
        tail_pos   = set(range(max(sink, prefill_len - sink), prefill_len))
        always_pos = sink_pos | tail_pos

        quest_pos = select_by_hierarchy(
            query,
            layer_topic_inodes[l],
            layer_leaf_inodes[l],
            k_topics=k_topics,
            k_leaves=k_leaves,
        )

        prefill_pos = sorted(always_pos | quest_pos)
        max_prefill = budget_l - n_recent_gen
        if len(prefill_pos) > max_prefill:
            quest_only  = sorted(quest_pos - always_pos)
            allowed     = max(0, max_prefill - len(always_pos))
            prefill_pos = sorted(always_pos | set(quest_only[:allowed]))

        pk = prefill_keys[l][:,   prefill_pos, :]
        pv = prefill_values[l][:, prefill_pos, :]

        if n_recent_gen > 0:
            pk = torch.cat([pk, gen_k[:, -n_recent_gen:, :]], dim=1)
            pv = torch.cat([pv, gen_v[:, -n_recent_gen:, :]], dim=1)

        if verbose:
            n_quest = len(quest_pos - always_pos)
            print(f"  [layer {l}] prefill selected: {len(prefill_pos)} tokens "
                  f"({len(always_pos)} always + {n_quest} quest), "
                  f"+ {n_recent_gen} recent_gen → active {len(prefill_pos) + n_recent_gen}")

        return pk.unsqueeze(0), pv.unsqueeze(0)   # [1, n_kv, active, head_dim]

    # ── Pre-allocate generation K/V buffers ───────────────────────────────
    # Pre-allocating avoids O(n²) torch.cat reallocations in the decode loop.
    _dtype     = next(iter(prefill_keys.values())).dtype
    gen_keys   = {l: torch.empty(n_kv, max_new_tokens, head_dim, device=device, dtype=_dtype)
                  for l in full_attn}
    gen_values = {l: torch.empty(n_kv, max_new_tokens, head_dim, device=device, dtype=_dtype)
                  for l in full_attn}
    _gen_step  = 0

    if verbose:
        print(f"\n[hierarchy] Initial cache reconstruction "
              f"(showing first 3 full-attn layers):")
    for l in full_attn:
        q = qcap.queries.get(l, prefill_keys[l][:, -1, :])
        cache.layers[l].keys, cache.layers[l].values = reconstruct(
            l, q, gen_keys[l][:, :_gen_step, :], gen_values[l][:, :_gen_step, :],
        )

    # ── Decode loop ───────────────────────────────────────────────────────
    next_tok  = out.logits[:, -1:, :].argmax(dim=-1)
    generated = [next_tok.item()]

    for step in range(max_new_tokens - 1):
        if eos_token_id is not None and generated[-1] == eos_token_id:
            break

        cache_pos = torch.tensor([current_pos], device=device)
        with torch.no_grad():
            out = model(
                next_tok,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_pos,
            )
        current_pos += 1

        # Write new token K/V into pre-allocated buffers at current step index.
        for l in full_attn:
            gen_keys[l][:, _gen_step, :]   = cache.layers[l].keys[0, :, -1, :]
            gen_values[l][:, _gen_step, :] = cache.layers[l].values[0, :, -1, :]
        _gen_step += 1

        # Every β steps: rebuild full-attention layer caches via Quest selection.
        if (step + 1) % beta == 0:
            if verbose:
                print(f"\n[hierarchy] Eviction at step {step+1} "
                      f"(generated {step+1} tokens so far):")
            for l in full_attn:
                q = qcap.queries.get(l, prefill_keys[l][:, -1, :])
                cache.layers[l].keys, cache.layers[l].values = reconstruct(
                    l, q, gen_keys[l][:, :_gen_step, :], gen_values[l][:, :_gen_step, :]
                )

        next_tok = out.logits[:, -1:, :].argmax(dim=-1)
        generated.append(next_tok.item())

    qcap.remove()

    return {
        "sequences":        torch.cat(
                                [input_ids, torch.tensor([generated], device=device)], dim=1
                            ),
        "generated_ids":    generated,
        "final_kv_lens":    [cache.layers[l].get_seq_length() for l in range(n_layers)],
        "layer_budgets":    layer_budgets,
        "n_full_attn":      len(full_attn),
        "n_leaf_clusters":  len(tree.leaf_clusters),
        "n_topic_nodes":    len(tree.topic_nodes),
    }


# ─── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id  = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2-1.5B-Instruct"
    print(f"Loading {model_id} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model     = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map="auto"
    )

    prompt = "The quick brown fox jumps over the lazy dog. " * 100
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    print(f"Prompt: {inputs.input_ids.size(1)} tokens")

    result = generate_with_hierarchy(
        model, tokenizer, inputs.input_ids,
        max_new_tokens=50,
        budget=256, sink=4, recent=64,
        k_topics=3, k_leaves=4,
        beta=16,
        eos_token_id=tokenizer.eos_token_id,
    )

    budgets = result["layer_budgets"]
    print(f"Hierarchy: {result['n_leaf_clusters']} leaf clusters, "
          f"{result['n_topic_nodes']} topic nodes")
    print(f"PyramidKV: layer 0={budgets[0]}, top layer={budgets[-1]}")
    assert budgets[0] >= budgets[-1], "PyramidKV monotone check failed"

    for l, (kl, bl) in enumerate(zip(result["final_kv_lens"], budgets)):
        assert kl <= bl + 64, f"Layer {l}: KV length {kl} exceeds budget {bl}"
    print("Budget checks OK")

    text = tokenizer.decode(result["sequences"][0][inputs.input_ids.size(1):],
                            skip_special_tokens=True)
    print(f"Output: {text[:200]}")
