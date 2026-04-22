"""
STEP 5 — HYBRID RETRIEVER
Analogy: Two detectives working the same case.
  Detective A (Vector) searches for text that SOUNDS similar to your question.
  Detective B (Graph)  follows the CONNECTIONS between people/skills/companies.
Both bring their findings. We merge them. The answer is more complete than either alone.

Example:
  Question: "What did Anand build at his last job?"
  Vector finds: chunks mentioning 'built', 'developed', 'created'
  Graph finds:  Company X → Project Y → Technologies Used  (even if 'built' wasn't in that chunk)
"""

import networkx as nx
from ingestion.embedder import embed_query
from ingestion.vector_store import query_similar
from ingestion.graph_builder import get_related_chunks, identify_query_entities


def retrieve(
    query: str,
    graph: nx.DiGraph,
    all_chunks: list[dict],
    top_k: int = 5,
) -> list[dict]:
    """
    Hybrid retrieval: vector search + graph traversal, merged and deduplicated.
    Returns a list of chunk dicts, each tagged with its retrieval source.
    """

    # --- PATH A: Vector Search ---
    # Convert question to embedding, find top_k nearest chunks in ChromaDB
    query_embedding = embed_query(query)
    vector_hits = query_similar(query_embedding, top_k=top_k)
    for hit in vector_hits:
        hit["retrieval_source"] = "vector"

    # --- PATH B: Graph Traversal ---
    # Find entities in the query that exist in the graph, walk outward 2 hops
    query_entities = identify_query_entities(query, graph)
    graph_hits = get_related_chunks(graph, query_entities, all_chunks, hops=2)
    for hit in graph_hits:
        hit["retrieval_source"] = "graph"

    # --- MERGE + DEDUPLICATE ---
    # Use chunk text as the deduplication key (same text = same chunk)
    seen_texts = set()
    merged = []

    # Vector hits go first (they are ranked by similarity score)
    for hit in vector_hits:
        key = hit["text"][:100]  # first 100 chars as fingerprint
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(hit)

    # Graph hits fill in structural context not caught by vector
    for hit in graph_hits:
        key = hit["text"][:100]
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(hit)

    return merged
