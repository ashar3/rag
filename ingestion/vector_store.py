"""
STEP 3b — VECTOR STORE (ChromaDB)
Analogy: ChromaDB is a filing cabinet where every folder has GPS coordinates.
When you ask a question, we convert it to coordinates too, then open the 5
nearest folders. That's semantic search — no keyword matching needed.
"""

import os
import uuid
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

load_dotenv()

_chroma_client = None
COLLECTION_NAME = "resume_chunks"


def _get_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine = measures angle between vectors
    )


def store_chunks(chunks: list[dict], embeddings: list[list[float]]) -> int:
    """
    Stores chunks + their embeddings into ChromaDB.
    Each entry has:  id, embedding, document (text), metadata (source, chunk_index)
    """
    collection = _get_collection()

    ids = [str(uuid.uuid4()) for _ in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(chunks)


def query_similar(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    """
    Given a query embedding, returns the top_k most similar chunks.
    Each result: { text, source, chunk_index, distance }
    distance closer to 0 = more similar (cosine).
    """
    collection = _get_collection()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text": doc,
            "source": meta.get("source", ""),
            "chunk_index": meta.get("chunk_index", -1),
            "distance": round(dist, 4),
        })
    return hits


def clear_collection():
    """Wipe all stored chunks — useful when re-ingesting a new PDF."""
    client = _get_client()
    client.delete_collection(COLLECTION_NAME)
