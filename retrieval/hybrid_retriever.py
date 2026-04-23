"""
HYBRID RETRIEVER — The 3-Detective System with Reciprocal Rank Fusion
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Query
    │
    ├── 🔵 Detective A (Vector)  → "find chunks that MEAN the same thing"
    │      Uses: ChromaDB cosine similarity on embeddings
    │      Strength: paraphrasing, synonyms, context
    │      Weakness: exact names, IDs, rare acronyms
    │
    ├── 🟡 Detective B (BM25)    → "find chunks that have the EXACT words"
    │      Uses: BM25Okapi keyword scoring
    │      Strength: proper nouns, tech names (AWS, React), phone numbers
    │      Weakness: can't understand "developer" = "engineer"
    │
    └── 🟢 Detective C (Graph)   → "find chunks CONNECTED to what you mentioned"
           Uses: NetworkX relationship traversal (2 hops)
           Strength: implicit relationships — ask about Company X, get its projects
           Weakness: only knows entities it extracted, misses unseen connections
                │
                ▼
    ⚖️  Reciprocal Rank Fusion (RRF)
           Each detective submits a ranked list.
           RRF converts ranks → scores → merges fairly.
           Formula: score += 1 / (60 + rank)
           Why 60? It's the standard smoothing constant — stops rank-1 from
           completely dominating when another detective has it at rank-2.
                │
                ▼
    📋 Final merged list (deduplicated, re-ranked by combined evidence)
"""

import networkx as nx
from rank_bm25 import BM25Okapi

from ingestion.embedder import embed_query
from ingestion.vector_store import query_similar
from ingestion.graph_builder import get_related_chunks, identify_query_entities
from ingestion.bm25_index import query_bm25


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────────────

def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """
    RRF merges multiple ranked lists into one fair combined ranking.

    Why not just concatenate?
      If Vector puts chunk X at rank 1 and BM25 puts it at rank 3,
      it's clearly very relevant — both detectives agree.
      Simple concatenation would duplicate it. RRF combines and boosts it.

    Why 1/(k+rank) not just 1/rank?
      k=60 prevents rank-1 from being 60× more valuable than rank-60.
      It smooths the curve so mid-ranked results from multiple detectives
      beat a rank-1 result from only one detective.
    """
    rrf_scores: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            key = hit["text"][:120]  # fingerprint by first 120 chars
            if key not in rrf_scores:
                rrf_scores[key] = {"hit": hit, "rrf_score": 0.0, "sources": []}
            rrf_scores[key]["rrf_score"] += 1.0 / (k + rank + 1)
            rrf_scores[key]["sources"].append(hit.get("retrieval_source", "?"))

    # sort by combined RRF score descending
    sorted_results = sorted(rrf_scores.values(), key=lambda x: x["rrf_score"], reverse=True)

    merged = []
    for item in sorted_results:
        hit = dict(item["hit"])
        hit["rrf_score"] = round(item["rrf_score"], 6)
        # tag with all detectives that found this chunk
        unique_sources = list(dict.fromkeys(item["sources"]))  # preserve order, deduplicate
        hit["retrieval_source"] = "+".join(unique_sources)
        merged.append(hit)

    return merged


# ── Main Retrieval Entry Point ─────────────────────────────────────────────────

def retrieve(
    query: str,
    graph: nx.DiGraph,
    all_chunks: list[dict],
    bm25_index: BM25Okapi,
    top_k: int = 5,
) -> dict:
    """
    Runs all 3 detectives, collects their individual results,
    fuses with RRF, returns both the merged result AND each detective's
    individual findings (for the learning commentary in the UI).
    """

    # ── Detective A: Vector Search ─────────────────────────────────────────────
    # Converts question to embedding, finds nearest chunks in ChromaDB
    query_embedding = embed_query(query)
    vector_hits = query_similar(query_embedding, top_k=top_k)
    for h in vector_hits:
        h["retrieval_source"] = "vector"

    # ── Detective B: BM25 Keyword Search ──────────────────────────────────────
    # Tokenizes question, scores all chunks by BM25, returns top_k
    bm25_hits = query_bm25(bm25_index, all_chunks, query, top_k=top_k)

    # ── Detective C: Graph Traversal ──────────────────────────────────────────
    # Finds entities in query that exist in the graph, walks outward 2 hops
    query_entities = identify_query_entities(query, graph)
    hops = 2
    # walk the graph manually to collect visited nodes for commentary
    visited_nodes: set[str] = set()
    frontier = list(query_entities)
    for _ in range(hops):
        next_frontier: list[str] = []
        for node in frontier:
            if node in visited_nodes or not graph.has_node(node):
                continue
            visited_nodes.add(node)
            next_frontier.extend(list(graph.successors(node)) + list(graph.predecessors(node)))
        frontier = next_frontier

    graph_hits = get_related_chunks(graph, query_entities, all_chunks, hops=hops)
    for h in graph_hits:
        h["retrieval_source"] = "graph"

    # ── RRF Fusion ────────────────────────────────────────────────────────────
    active_lists = [lst for lst in [vector_hits, bm25_hits, graph_hits] if lst]
    merged = reciprocal_rank_fusion(active_lists)

    # ── Real-time parameters for the learning commentary ──────────────────────
    query_tokens = query.lower().split()
    bm25_token_idfs = {tok: round(float(bm25_index.idf.get(tok, 0.0)), 4) for tok in query_tokens}

    avg_doc_len = (
        sum(bm25_index.doc_len) / len(bm25_index.doc_len)
        if hasattr(bm25_index, "doc_len") and bm25_index.doc_len else 0
    )

    return {
        "merged": merged,
        "detective_a_vector": vector_hits,
        "detective_b_bm25": bm25_hits,
        "detective_c_graph": graph_hits,
        "query_entities": query_entities,

        # real-time parameters for the step-by-step walkthrough
        "params": {
            # Vector
            "embedding_model":  "text-embedding-3-small",
            "embedding_dim":    len(query_embedding),
            "embedding_preview": [round(float(x), 4) for x in query_embedding[:8]],
            "vector_distance_metric": "cosine",

            # BM25
            "query_tokens":     query_tokens,
            "bm25_k1":          round(bm25_index.k1, 2),
            "bm25_b":           round(bm25_index.b, 2),
            "bm25_avg_doc_len": round(avg_doc_len, 1),
            "bm25_token_idfs":  bm25_token_idfs,
            "bm25_corpus_size": len(all_chunks),

            # Graph
            "graph_total_nodes": graph.number_of_nodes(),
            "graph_total_edges": graph.number_of_edges(),
            "graph_hops":        hops,
            "graph_nodes_visited": sorted(visited_nodes),
            "graph_visit_count": len(visited_nodes),

            # RRF
            "rrf_k": 60,
            "rrf_top_results": [
                {
                    "preview": m["text"][:80],
                    "score":   round(m.get("rrf_score", 0.0), 6),
                    "sources": m.get("retrieval_source", ""),
                }
                for m in merged[:5]
            ],
        },
    }
