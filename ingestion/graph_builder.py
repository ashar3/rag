"""
STEP 4 — GRAPH BUILDER
Analogy: After reading each chunk, we ask the LLM: "Who/what is mentioned here,
and how do they connect?" We build a map of relationships — like a family tree
but for resume concepts. Later, when you ask a question, we can WALK this map
to find related context that a keyword search would completely miss.

Example graph for a resume:
  "Python" -[USED_AT]-> "Company X"
  "Company X" -[EMPLOYED]-> "Anand"
  "Company X" -[DURING]-> "2020-2023"
  "React" -[USED_AT]-> "Company X"

Query: "what frontend tech did he use?"
→ Graph knows: frontend tech → React → Company X → other context about Company X
"""

import json
import os
import networkx as nx
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


EXTRACTION_PROMPT = """Extract entities and relationships from this resume text.

Return ONLY valid JSON in this exact format:
{{
  "entities": [
    {{"name": "entity name", "type": "SKILL|COMPANY|PERSON|ROLE|EDUCATION|DATE|PROJECT|TOOL"}}
  ],
  "relationships": [
    {{"from": "entity1", "relation": "USED_AT|WORKED_AT|HAS_SKILL|STUDIED_AT|BUILT|DURING|LED|PART_OF", "to": "entity2"}}
  ]
}}

Rules:
- Only include entities explicitly mentioned in the text
- Keep entity names short (2-4 words max)
- Only include relationships where both entities appear in your entities list
- Return empty lists if nothing relevant found

Text:
{text}"""


def extract_entities_and_relations(chunk_text: str) -> dict:
    """
    Sends one chunk to the LLM and gets back entities + relationships as JSON.
    """
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT.format(text=chunk_text)}
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        data = json.loads(raw)
        return {
            "entities": data.get("entities", []),
            "relationships": data.get("relationships", []),
        }
    except Exception:
        return {"entities": [], "relationships": []}


def build_graph(chunks: list[dict]) -> nx.DiGraph:
    """
    Processes every chunk, extracts entities+relationships, builds a directed graph.

    Graph nodes: entities with a 'type' attribute
    Graph edges: relationships with a 'relation' attribute
    Each node also stores which chunks mention it (for retrieval bridging).
    """
    G = nx.DiGraph()

    for chunk in chunks:
        extracted = extract_entities_and_relations(chunk["text"])

        for entity in extracted["entities"]:
            name = entity["name"].strip()
            if not name:
                continue
            if G.has_node(name):
                # entity already exists — add this chunk to its mention list
                G.nodes[name]["chunks"].append(chunk["chunk_index"])
            else:
                G.add_node(name, type=entity["type"], chunks=[chunk["chunk_index"]])

        for rel in extracted["relationships"]:
            frm = rel["from"].strip()
            to = rel["to"].strip()
            relation = rel["relation"].strip()
            if frm and to and G.has_node(frm) and G.has_node(to):
                G.add_edge(frm, to, relation=relation)

    return G


def get_related_chunks(G: nx.DiGraph, query_entities: list[str], all_chunks: list[dict], hops: int = 2) -> list[dict]:
    """
    Given a list of entity names found in the query, walks the graph outward
    (up to `hops` edges) and collects all chunk indices that are connected.

    This is the Graph RAG retrieval step — we get context that is
    *structurally related* even if it doesn't use the same words as the query.
    """
    visited_nodes = set()
    relevant_chunk_indices = set()

    queue = list(query_entities)
    current_hop = 0

    while queue and current_hop < hops:
        next_queue = []
        for node in queue:
            if node in visited_nodes or not G.has_node(node):
                continue
            visited_nodes.add(node)
            relevant_chunk_indices.update(G.nodes[node].get("chunks", []))
            neighbors = list(G.successors(node)) + list(G.predecessors(node))
            next_queue.extend(neighbors)
        queue = next_queue
        current_hop += 1

    chunk_map = {c["chunk_index"]: c for c in all_chunks}
    return [chunk_map[i] for i in relevant_chunk_indices if i in chunk_map]


def identify_query_entities(query: str, G: nx.DiGraph) -> list[str]:
    """
    Simple matching: find graph node names that appear in the query string.
    Case-insensitive. Used to seed the graph traversal.
    """
    query_lower = query.lower()
    matched = []
    for node in G.nodes():
        if node.lower() in query_lower:
            matched.append(node)
    return matched
