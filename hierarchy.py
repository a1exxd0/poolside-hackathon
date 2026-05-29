"""
hierarchy.py — Streaming text hierarchy for KV-cache organisation.

Pipeline:
  words → StreamingChunker → GraphNode
        → SemanticGraph (HNSW)
        → LocalBeliefPropagator
        → DynamicCommunityDetector (Louvain)
        → OnlineHierarchy (2-level cluster tree)
        → HierarchicalKVPolicy

Standalone module; does not import from the existing TriAttention codebase.
"""

from __future__ import annotations

import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import networkx as nx
import hnswlib
import community as community_louvain
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

SBERT_MODEL = "all-MiniLM-L6-v2"   # 384-dim, ~90 MB, fast CPU inference
EMBED_DIM = 384
HNSW_INITIAL_MAX = 512
HNSW_EF_CONSTRUCTION = 200
HNSW_M = 16

DISCOURSE_MARKERS: frozenset = frozenset({
    "however", "therefore", "furthermore", "moreover", "nevertheless",
    "consequently", "contrast", "example", "instance", "addition",
    "alternatively", "meanwhile", "subsequently", "specifically",
    "notably", "importantly", "conclusion", "summary", "overall",
    "hence", "thus", "besides", "regardless", "nonetheless",
})

# Punctuation that strongly suggests a sentence boundary
_SENTENCE_TERMINALS = frozenset(".!?")


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Dot product of two unit-normalised vectors = cosine similarity."""
    return float(np.dot(a, b))


def _normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v


def _word_is_discourse(word: str) -> bool:
    clean = re.sub(r"[^a-z]", "", word.lower())
    return clean in DISCOURSE_MARKERS


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    node_id: int
    text: str
    embedding: np.ndarray           # shape (EMBED_DIM,), unit-normalised
    timestamp: float
    temporal_prev: Optional[int] = None
    temporal_next: Optional[int] = None
    semantic_edges: List[int] = field(default_factory=list)
    community_id: int = -1
    belief_score: float = 0.5
    kv_summary: Optional[Any] = None


@dataclass
class LeafCluster:
    cluster_id: int
    centroid: np.ndarray            # unit-normalised running mean
    member_node_ids: List[int] = field(default_factory=list)


@dataclass
class TopicNode:
    topic_id: int
    centroid: np.ndarray
    leaf_cluster_ids: List[int] = field(default_factory=list)


@dataclass
class HierarchyTree:
    leaf_clusters: Dict[int, LeafCluster] = field(default_factory=dict)
    topic_nodes: Dict[int, TopicNode] = field(default_factory=dict)
    node_to_leaf: Dict[int, int] = field(default_factory=dict)    # graph_node_id → leaf_cluster_id
    leaf_to_topic: Dict[int, int] = field(default_factory=dict)   # leaf_cluster_id → topic_id


@dataclass
class KVEntry:
    node_id: int
    text_summary: str
    embedding: np.ndarray
    belief_score: float
    community_id: int
    children: List[int] = field(default_factory=list)
    raw_kv_recent: Optional[Any] = None
    raw_kv_important: Optional[Any] = None
    summary_kv: Optional[Any] = None


# ---------------------------------------------------------------------------
# SemanticGraph — HNSW-backed node store
# ---------------------------------------------------------------------------

class SemanticGraph:
    """
    Maintains all GraphNodes in an HNSW approximate nearest-neighbour index.
    Automatically resizes the index (lazy doubling) as new nodes arrive.
    """

    def __init__(self, dim: int = EMBED_DIM, k_neighbors: int = 5) -> None:
        self._dim = dim
        self._k = k_neighbors
        self._max_elements = HNSW_INITIAL_MAX
        self._nodes: Dict[int, GraphNode] = {}
        self._index = self._new_index(self._max_elements)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def insert(self, node: GraphNode) -> None:
        """Add node to HNSW index, populate semantic_edges, wire temporal link."""
        self._maybe_resize()

        # Add to HNSW (expects 2-D float32 array)
        emb = node.embedding.astype(np.float32)
        self._index.add_items(emb.reshape(1, -1), [node.node_id])

        # Find k nearest neighbours (excluding self)
        n_query = min(self._k + 1, len(self._nodes))
        if n_query > 0:
            labels, distances = self._index.knn_query(emb.reshape(1, -1), k=n_query)
            for nbr_id, dist in zip(labels[0], distances[0]):
                if nbr_id == node.node_id:
                    continue
                if nbr_id not in self._nodes:
                    continue
                # Add bidirectional semantic edges
                if nbr_id not in node.semantic_edges:
                    node.semantic_edges.append(int(nbr_id))
                nbr = self._nodes[int(nbr_id)]
                if node.node_id not in nbr.semantic_edges:
                    nbr.semantic_edges.append(node.node_id)

        # Wire temporal edge to previous node
        if node.temporal_prev is not None and node.temporal_prev in self._nodes:
            prev = self._nodes[node.temporal_prev]
            prev.temporal_next = node.node_id
            if node.temporal_prev not in node.semantic_edges:
                node.semantic_edges.append(node.temporal_prev)
            if node.node_id not in prev.semantic_edges:
                prev.semantic_edges.append(node.node_id)

        self._nodes[node.node_id] = node

    def get_neighbourhood(self, node_id: int, radius: int = 2) -> List[GraphNode]:
        """BFS up to `radius` hops through semantic_edges (includes temporal)."""
        if node_id not in self._nodes:
            return []
        visited: Set[int] = {node_id}
        frontier = deque([(node_id, 0)])
        result: List[GraphNode] = [self._nodes[node_id]]

        while frontier:
            current_id, depth = frontier.popleft()
            if depth >= radius:
                continue
            node = self._nodes[current_id]
            neighbours = list(node.semantic_edges)
            if node.temporal_prev is not None:
                neighbours.append(node.temporal_prev)
            if node.temporal_next is not None:
                neighbours.append(node.temporal_next)
            for nbr_id in neighbours:
                if nbr_id not in visited and nbr_id in self._nodes:
                    visited.add(nbr_id)
                    result.append(self._nodes[nbr_id])
                    frontier.append((nbr_id, depth + 1))

        return result

    @property
    def nodes(self) -> Dict[int, GraphNode]:
        return self._nodes

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_index(self, max_elements: int) -> hnswlib.Index:
        idx = hnswlib.Index(space="cosine", dim=self._dim)
        idx.init_index(
            max_elements=max_elements,
            ef_construction=HNSW_EF_CONSTRUCTION,
            M=HNSW_M,
        )
        idx.set_ef(50)
        return idx

    def _maybe_resize(self) -> None:
        """Double HNSW capacity when 90% full."""
        if len(self._nodes) < int(self._max_elements * 0.9):
            return
        new_max = self._max_elements * 2
        new_index = self._new_index(new_max)
        if self._nodes:
            ids = list(self._nodes.keys())
            embeddings = np.stack(
                [self._nodes[i].embedding.astype(np.float32) for i in ids]
            )
            new_index.add_items(embeddings, ids)
        self._index = new_index
        self._max_elements = new_max


# ---------------------------------------------------------------------------
# StreamingChunker — word-by-word boundary detection
# ---------------------------------------------------------------------------

class StreamingChunker:
    """
    Accumulates words and freezes chunks when a boundary is detected.
    Boundary score = 0.6 * hard_signal + 0.4 * soft_signal (cosine shift).
    Embeddings are re-computed only every `embed_every_n` words to avoid
    running SBERT on every token.
    """

    def __init__(
        self,
        sbert: SentenceTransformer,
        boundary_threshold: float = 0.55,
        max_tokens: int = 80,
        embed_every_n: int = 5,
    ) -> None:
        self._sbert = sbert
        self._threshold = boundary_threshold
        self._max_tokens = max_tokens
        self._embed_every_n = embed_every_n

        self._buffer: List[str] = []
        self._last_chunk_embedding: Optional[np.ndarray] = None
        self._candidate_embedding: Optional[np.ndarray] = None  # cached mid-buffer embed
        self._node_counter: int = 0
        self._prev_node_id: Optional[int] = None

    def add_word(self, word: str) -> Optional[GraphNode]:
        """Feed one word. Returns a frozen GraphNode on boundary, else None."""
        self._buffer.append(word)

        # Re-embed candidate periodically
        if len(self._buffer) % self._embed_every_n == 0:
            self._candidate_embedding = self._embed_text(" ".join(self._buffer))

        if self._compute_boundary_score() >= self._threshold:
            return self._freeze_chunk()

        return None

    def flush(self) -> Optional[GraphNode]:
        """Force-freeze any remaining buffered words."""
        if not self._buffer:
            return None
        return self._freeze_chunk()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _compute_boundary_score(self) -> float:
        if not self._buffer:
            return 0.0

        # --- Hard signals (0 or 1 each) ---
        last_word = self._buffer[-1]
        first_word = self._buffer[0]

        # Sentence-terminal punctuation
        punct_score = 1.0 if last_word and last_word[-1] in _SENTENCE_TERMINALS else 0.0

        # Paragraph break (double newline embedded in word)
        para_score = 1.0 if "\n\n" in last_word else 0.0

        # Discourse marker at start of chunk
        discourse_score = 1.0 if _word_is_discourse(first_word) else 0.0

        # Max token budget
        budget_score = 1.0 if len(self._buffer) >= self._max_tokens else 0.0

        hard_signal = max(punct_score, para_score, discourse_score, budget_score)

        # --- Soft signal: cosine shift from last chunk ---
        soft_signal = 0.0
        if self._last_chunk_embedding is not None and self._candidate_embedding is not None:
            sim = _cosine_similarity(self._last_chunk_embedding, self._candidate_embedding)
            soft_signal = max(0.0, 1.0 - sim)

        return 0.6 * hard_signal + 0.4 * soft_signal

    def _freeze_chunk(self) -> GraphNode:
        text = " ".join(self._buffer)
        embedding = self._embed_text(text)

        node = GraphNode(
            node_id=self._node_counter,
            text=text,
            embedding=embedding,
            timestamp=time.time(),
            temporal_prev=self._prev_node_id,
        )

        self._last_chunk_embedding = embedding
        self._candidate_embedding = None
        self._prev_node_id = self._node_counter
        self._node_counter += 1
        self._buffer = []
        return node

    def _embed_text(self, text: str) -> np.ndarray:
        emb = self._sbert.encode(text, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype(np.float32)


# ---------------------------------------------------------------------------
# LocalBeliefPropagator — coherence scores over local neighbourhood
# ---------------------------------------------------------------------------

class LocalBeliefPropagator:
    """
    Updates belief_score for a newly inserted node and its 2-hop neighbourhood.
    belief_score = mean cosine similarity to direct semantic neighbours.
    """

    def __init__(self, graph: SemanticGraph, max_hops: int = 2) -> None:
        self._graph = graph
        self._max_hops = max_hops

    def update(self, new_node: GraphNode) -> None:
        affected = self._graph.get_neighbourhood(new_node.node_id, radius=self._max_hops)
        for node in affected:
            neighbours = [
                self._graph.nodes[nid]
                for nid in node.semantic_edges
                if nid in self._graph.nodes
            ]
            if not neighbours:
                node.belief_score = 0.5
                continue
            sims = [_cosine_similarity(node.embedding, n.embedding) for n in neighbours]
            node.belief_score = float(np.mean(sims))


# ---------------------------------------------------------------------------
# DynamicCommunityDetector — incremental Louvain on 2-hop subgraphs
# ---------------------------------------------------------------------------

class DynamicCommunityDetector:
    """
    Maintains a networkx graph mirroring semantic edges.
    On each new node, runs Louvain on the 2-hop subgraph and reconciles
    local community IDs to stable global IDs via plurality vote.
    """

    def __init__(self, graph: SemanticGraph) -> None:
        self._graph = graph
        self._nx: nx.Graph = nx.Graph()
        self._global_map: Dict[int, int] = {}   # node_id → global community ID
        self._global_counter: int = 0

    def update(self, new_node: GraphNode) -> None:
        # Add node and all its edges to networkx graph
        self._nx.add_node(new_node.node_id)
        for nbr_id in new_node.semantic_edges:
            if nbr_id in self._graph.nodes:
                nbr = self._graph.nodes[nbr_id]
                sim = _cosine_similarity(new_node.embedding, nbr.embedding)
                self._nx.add_edge(new_node.node_id, nbr_id, weight=float(sim))

        # Extract 2-hop subgraph
        affected_ids = [n.node_id for n in
                        self._graph.get_neighbourhood(new_node.node_id, radius=2)]
        subgraph = self._nx.subgraph(affected_ids).copy()

        if subgraph.number_of_nodes() < 2:
            # Isolated node: assign fresh community
            cid = self._global_counter
            self._global_counter += 1
            self._global_map[new_node.node_id] = cid
            new_node.community_id = cid
            return

        # Run Louvain on subgraph
        partition: Dict[int, int] = community_louvain.best_partition(
            subgraph, weight="weight", random_state=42
        )

        # Reconcile local → global community IDs
        reconciled = self._reconcile(partition, list(subgraph.nodes))

        # Write back to GraphNodes
        for node_id, global_cid in reconciled.items():
            self._global_map[node_id] = global_cid
            if node_id in self._graph.nodes:
                self._graph.nodes[node_id].community_id = global_cid

    def _reconcile(
        self,
        partition: Dict[int, int],
        node_ids: List[int],
    ) -> Dict[int, int]:
        """Map Louvain local IDs → stable global IDs via plurality vote."""
        # Group node_ids by local community
        local_groups: Dict[int, List[int]] = {}
        for nid in node_ids:
            local_cid = partition.get(nid, 0)
            local_groups.setdefault(local_cid, []).append(nid)

        claimed_globals: Set[int] = set()
        result: Dict[int, int] = {}

        for local_cid, members in local_groups.items():
            # Find plurality of previously-assigned global IDs for these members
            votes: Dict[int, int] = {}
            for nid in members:
                if nid in self._global_map:
                    g = self._global_map[nid]
                    votes[g] = votes.get(g, 0) + 1

            chosen_global: Optional[int] = None
            if votes:
                best_g, best_count = max(votes.items(), key=lambda x: x[1])
                # Reuse if majority (>50%) and not claimed by another local community
                if best_count > len(members) * 0.5 and best_g not in claimed_globals:
                    chosen_global = best_g

            if chosen_global is None:
                chosen_global = self._global_counter
                self._global_counter += 1

            claimed_globals.add(chosen_global)
            for nid in members:
                result[nid] = chosen_global

        return result

    @property
    def global_map(self) -> Dict[int, int]:
        return self._global_map


# ---------------------------------------------------------------------------
# OnlineHierarchy — 2-level cluster tree
# ---------------------------------------------------------------------------

class OnlineHierarchy:
    """
    Level 0: leaf clusters of semantically similar GraphNodes.
    Level 1: topic nodes grouping leaf clusters via k-means on centroids.
    """

    def __init__(
        self,
        graph: SemanticGraph,
        join_threshold: float = 0.70,
        split_threshold: int = 20,
        n_topics: int = 5,
        topic_rebuild_every: int = 5,
    ) -> None:
        self._graph = graph
        self._join_threshold = join_threshold
        self._split_threshold = split_threshold
        self._n_topics = n_topics
        self._topic_rebuild_every = topic_rebuild_every
        self._tree = HierarchyTree()
        self._leaf_counter = 0
        self._topic_counter = 0
        self._inserts_since_rebuild = 0

    def insert(self, node: GraphNode) -> None:
        nearest_id, best_sim = self._find_nearest_leaf(node.embedding)

        if nearest_id is not None and best_sim >= self._join_threshold:
            cluster = self._tree.leaf_clusters[nearest_id]
            self._update_centroid(cluster, node.embedding)
            cluster.member_node_ids.append(node.node_id)
            self._tree.node_to_leaf[node.node_id] = nearest_id
            if len(cluster.member_node_ids) > self._split_threshold:
                self._split_cluster(nearest_id)
        else:
            # Create new leaf cluster
            new_cluster = LeafCluster(
                cluster_id=self._leaf_counter,
                centroid=node.embedding.copy(),
                member_node_ids=[node.node_id],
            )
            self._tree.leaf_clusters[self._leaf_counter] = new_cluster
            self._tree.node_to_leaf[node.node_id] = self._leaf_counter
            self._leaf_counter += 1

        self._inserts_since_rebuild += 1
        if self._inserts_since_rebuild >= self._topic_rebuild_every:
            self._rebuild_topics()
            self._inserts_since_rebuild = 0

    def get_tree(self) -> HierarchyTree:
        return self._tree

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _find_nearest_leaf(self, embedding: np.ndarray) -> Tuple[Optional[int], float]:
        best_id: Optional[int] = None
        best_sim = -1.0
        for cid, cluster in self._tree.leaf_clusters.items():
            sim = _cosine_similarity(embedding, cluster.centroid)
            if sim > best_sim:
                best_sim = sim
                best_id = cid
        return best_id, best_sim

    def _update_centroid(self, cluster: LeafCluster, new_embedding: np.ndarray) -> None:
        n = len(cluster.member_node_ids)
        raw = (n * cluster.centroid + new_embedding) / (n + 1)
        cluster.centroid = _normalize(raw)

    def _split_cluster(self, cluster_id: int) -> None:
        cluster = self._tree.leaf_clusters[cluster_id]
        member_ids = cluster.member_node_ids
        if len(member_ids) < 2:
            return

        embeddings = np.stack([
            self._graph.nodes[nid].embedding
            for nid in member_ids
            if nid in self._graph.nodes
        ]).astype(np.float32)

        if len(embeddings) < 2:
            return

        km = KMeans(n_clusters=2, n_init=3, random_state=42)
        labels = km.fit_predict(embeddings)

        # Build two new clusters
        for group_label in (0, 1):
            group_ids = [member_ids[i] for i, lbl in enumerate(labels) if lbl == group_label]
            if not group_ids:
                continue
            centroid = _normalize(np.mean(embeddings[labels == group_label], axis=0))
            new_cluster = LeafCluster(
                cluster_id=self._leaf_counter,
                centroid=centroid,
                member_node_ids=group_ids,
            )
            self._tree.leaf_clusters[self._leaf_counter] = new_cluster
            for nid in group_ids:
                self._tree.node_to_leaf[nid] = self._leaf_counter
            self._leaf_counter += 1

        # Remove original cluster
        del self._tree.leaf_clusters[cluster_id]

    def _rebuild_topics(self) -> None:
        n_clusters = len(self._tree.leaf_clusters)
        if n_clusters == 0:
            self._tree.topic_nodes = {}
            self._tree.leaf_to_topic = {}
            return

        k = min(self._n_topics, n_clusters)
        cluster_ids = list(self._tree.leaf_clusters.keys())
        centroids = np.stack([
            self._tree.leaf_clusters[cid].centroid for cid in cluster_ids
        ]).astype(np.float32)

        if k == 1:
            # All in one topic
            topic = TopicNode(
                topic_id=0,
                centroid=_normalize(centroids[0]),
                leaf_cluster_ids=cluster_ids,
            )
            self._tree.topic_nodes = {0: topic}
            self._tree.leaf_to_topic = {cid: 0 for cid in cluster_ids}
            return

        km = KMeans(n_clusters=k, n_init=3, random_state=42)
        labels = km.fit_predict(centroids)

        new_topics: Dict[int, TopicNode] = {}
        new_leaf_to_topic: Dict[int, int] = {}
        label_to_topic: Dict[int, int] = {}

        for idx, (cid, lbl) in enumerate(zip(cluster_ids, labels)):
            lbl = int(lbl)
            if lbl not in label_to_topic:
                label_to_topic[lbl] = self._topic_counter
                self._topic_counter += 1
            tid = label_to_topic[lbl]
            new_leaf_to_topic[cid] = tid
            if tid not in new_topics:
                new_topics[tid] = TopicNode(
                    topic_id=tid,
                    centroid=_normalize(km.cluster_centers_[lbl].astype(np.float32)),
                    leaf_cluster_ids=[],
                )
            new_topics[tid].leaf_cluster_ids.append(cid)

        self._tree.topic_nodes = new_topics
        self._tree.leaf_to_topic = new_leaf_to_topic


# ---------------------------------------------------------------------------
# HierarchicalKVPolicy — metadata store + retrieval plan
# ---------------------------------------------------------------------------

class HierarchicalKVPolicy:
    """
    Stores per-node KV metadata and produces structured retrieval plans
    for generation (Step 8 of the algorithm).
    """

    _RECENT_WINDOW_SIZE = 5
    _SINK_COUNT = 3

    def __init__(self, graph: SemanticGraph, hierarchy: OnlineHierarchy) -> None:
        self._graph = graph
        self._hierarchy = hierarchy
        self._kv_store: Dict[int, KVEntry] = {}
        self._insertion_order: List[int] = []

    def register_node(self, node: GraphNode) -> None:
        tree = self._hierarchy.get_tree()
        leaf_id = tree.node_to_leaf.get(node.node_id)
        children: List[int] = []
        if leaf_id is not None:
            cluster = tree.leaf_clusters.get(leaf_id)
            if cluster:
                children = [nid for nid in cluster.member_node_ids if nid != node.node_id]

        entry = KVEntry(
            node_id=node.node_id,
            text_summary=node.text[:120],
            embedding=node.embedding,
            belief_score=node.belief_score,
            community_id=node.community_id,
            children=children,
        )
        self._kv_store[node.node_id] = entry
        self._insertion_order.append(node.node_id)

    def get_retrieval_plan(
        self,
        query_embedding: np.ndarray,
        top_k_communities: int = 3,
    ) -> dict:
        # 1. Recent window (last N nodes)
        recent = list(self._insertion_order[-self._RECENT_WINDOW_SIZE:])

        # 2. Attention sinks (highest belief_score)
        sorted_by_belief = sorted(
            self._kv_store.values(), key=lambda e: e.belief_score, reverse=True
        )
        sinks = [e.node_id for e in sorted_by_belief[: self._SINK_COUNT]]

        # 3. Rank communities by avg cosine sim of members to query
        community_scores: Dict[int, List[float]] = {}
        for entry in self._kv_store.values():
            cid = entry.community_id
            sim = _cosine_similarity(query_embedding, entry.embedding)
            community_scores.setdefault(cid, []).append(sim)

        ranked_communities = sorted(
            community_scores.keys(),
            key=lambda cid: float(np.mean(community_scores[cid])),
            reverse=True,
        )[:top_k_communities]

        # 4 & 5. Community representatives (highest belief per community)
        representatives: List[int] = []
        for cid in ranked_communities:
            members = [e for e in self._kv_store.values() if e.community_id == cid]
            if members:
                rep = max(members, key=lambda e: e.belief_score)
                representatives.append(rep.node_id)

        # 6. Expanded children of representatives
        expanded: List[int] = []
        for rep_id in representatives:
            entry = self._kv_store.get(rep_id)
            if entry:
                expanded.extend(entry.children)

        policy_description = (
            f"Attend to: {len(recent)} recent nodes, "
            f"{len(sinks)} attention sinks, "
            f"{len(representatives)} community representatives "
            f"(communities {ranked_communities}), "
            f"and {len(expanded)} expanded child nodes."
        )

        return {
            "recent_window": recent,
            "attention_sinks": sinks,
            "top_communities": ranked_communities,
            "community_representatives": representatives,
            "expanded_children": expanded,
            "policy_description": policy_description,
        }

    @property
    def kv_store(self) -> Dict[int, KVEntry]:
        return self._kv_store


# ---------------------------------------------------------------------------
# HierarchyPipeline — public API
# ---------------------------------------------------------------------------

class HierarchyPipeline:
    """
    End-to-end streaming text hierarchy pipeline.

    Usage:
        pipeline = HierarchyPipeline()
        for word in text.split():
            pipeline.process_word(word)
        pipeline.flush()
        result = pipeline.get_hierarchy()
    """

    def __init__(
        self,
        sbert_model: str = SBERT_MODEL,
        boundary_threshold: float = 0.55,
        max_tokens: int = 80,
        k_neighbors: int = 5,
        join_threshold: float = 0.70,
        split_threshold: int = 20,
        n_topics: int = 5,
        topic_rebuild_every: int = 5,
        sbert_instance=None,
    ) -> None:
        # Accept a pre-loaded SentenceTransformer to avoid reloading across calls
        self._sbert = sbert_instance if sbert_instance is not None else SentenceTransformer(sbert_model)
        self._graph = SemanticGraph(dim=EMBED_DIM, k_neighbors=k_neighbors)
        self._chunker = StreamingChunker(
            sbert=self._sbert,
            boundary_threshold=boundary_threshold,
            max_tokens=max_tokens,
        )
        self._bp = LocalBeliefPropagator(self._graph)
        self._community = DynamicCommunityDetector(self._graph)
        self._hierarchy = OnlineHierarchy(
            graph=self._graph,
            join_threshold=join_threshold,
            split_threshold=split_threshold,
            n_topics=n_topics,
            topic_rebuild_every=topic_rebuild_every,
        )
        self._kv_policy = HierarchicalKVPolicy(self._graph, self._hierarchy)
        self._node_count = 0

    def process_word(self, word: str) -> Optional[int]:
        """Feed one word. Returns new node_id if a chunk was frozen, else None."""
        node = self._chunker.add_word(word)
        if node is not None:
            return self._process_node(node)
        return None

    def flush(self) -> Optional[int]:
        """Drain the remaining word buffer."""
        node = self._chunker.flush()
        if node is not None:
            return self._process_node(node)
        return None

    def get_hierarchy(self) -> dict:
        tree = self._hierarchy.get_tree()
        nodes_summary = {}
        for nid, node in self._graph.nodes.items():
            leaf_id = tree.node_to_leaf.get(nid)
            topic_id = tree.leaf_to_topic.get(leaf_id, -1) if leaf_id is not None else -1
            nodes_summary[nid] = {
                "text": node.text,
                "belief_score": node.belief_score,
                "community_id": node.community_id,
                "cluster_id": leaf_id,
                "topic_id": topic_id,
            }
        return {
            "node_count": self._node_count,
            "tree": tree,
            "nodes": nodes_summary,
            "kv_policy": self._kv_policy,
        }

    def _process_node(self, node: GraphNode) -> int:
        self._graph.insert(node)
        self._bp.update(node)
        self._community.update(node)
        self._hierarchy.insert(node)
        self._kv_policy.register_node(node)
        self._node_count += 1
        return node.node_id


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = """
    The transformer architecture introduced attention mechanisms that revolutionised
    natural language processing. Self-attention allows each token to attend to all
    other tokens in the sequence, enabling rich contextual representations.

    However, the quadratic memory cost of full attention becomes prohibitive for
    long sequences. Researchers have proposed many approximations including sparse
    attention, linear attention, and sliding window approaches.

    For example, the Longformer model uses a combination of local window attention
    and global attention on special tokens. This reduces complexity from O(n^2) to
    O(n) while retaining most of the modelling power.

    Therefore, KV-cache optimisation is critical for efficient inference. The KV
    cache stores precomputed key and value tensors to avoid recomputation during
    autoregressive generation. Eviction policies determine which entries to discard
    when the cache is full.

    In conclusion, hierarchical organisation of the KV cache based on semantic
    structure offers a principled approach to cache management that mirrors how
    human memory organises information across different levels of abstraction.
    """

    print("Initialising HierarchyPipeline...")
    pipeline = HierarchyPipeline(boundary_threshold=0.45, max_tokens=40, n_topics=3)

    print("Processing words...")
    nodes_created = []
    for word in SAMPLE.split():
        node_id = pipeline.process_word(word)
        if node_id is not None:
            nodes_created.append(node_id)
            print(f"  -> Froze chunk as node {node_id}")

    final_id = pipeline.flush()
    if final_id is not None:
        nodes_created.append(final_id)
        print(f"  -> Flushed final chunk as node {final_id}")

    result = pipeline.get_hierarchy()
    tree = result["tree"]

    print(f"\nTotal nodes: {result['node_count']}")
    print(f"Leaf clusters: {len(tree.leaf_clusters)}")
    print(f"Topic nodes:   {len(tree.topic_nodes)}")

    for cid, cluster in tree.leaf_clusters.items():
        tid = tree.leaf_to_topic.get(cid, -1)
        print(f"  Cluster {cid}: {len(cluster.member_node_ids)} members, topic={tid}")

    print("\nNode summaries:")
    for nid, info in result["nodes"].items():
        print(
            f"  Node {nid}: belief={info['belief_score']:.3f}  "
            f"comm={info['community_id']}  cluster={info['cluster_id']}  "
            f"topic={info['topic_id']}  | {info['text'][:60]!r}"
        )

    kv = result["kv_policy"]
    dummy_query = np.random.randn(EMBED_DIM).astype(np.float32)
    dummy_query = _normalize(dummy_query)
    plan = kv.get_retrieval_plan(dummy_query, top_k_communities=2)
    print(f"\nKV policy: {plan['policy_description']}")

    assert result["node_count"] > 0, "No nodes created"
    assert len(tree.leaf_clusters) > 0, "No leaf clusters"
    print("\nAll assertions passed.")
