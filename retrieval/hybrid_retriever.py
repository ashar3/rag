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
    graph_hits = get_related_chunks(graph, query_entities, all_chunks, hops=2)
    for h in graph_hits:
        h["retrieval_source"] = "graph"

    # ── RRF Fusion ────────────────────────────────────────────────────────────
    # Only fuse lists that actually found something
    active_lists = [lst for lst in [vector_hits, bm25_hits, graph_hits] if lst]
    merged = reciprocal_rank_fusion(active_lists)

    return {
        "merged": merged,                    # final ranked context for LLM
        "detective_a_vector": vector_hits,   # raw Vector results
        "detective_b_bm25": bm25_hits,       # raw BM25 results
        "detective_c_graph": graph_hits,     # raw Graph results
        "query_entities": query_entities,    # what the graph matched on
    }
