"""
STEP 3b — VECTOR STORE (ChromaDB)
Analogy: ChromaDB is a filing cabinet where every folder has GPS coordinates.
When you ask a question, we convert it to coordinates too, then open the 5
nearest folders. That's semantic search — no keyword matching needed.
"""

import os
import uuid
import shutil
import tempfile
import chromadb
from dotenv import load_dotenv

load_dotenv()

_chroma_client = None
_temp_dir = None                    # tracks the temp dir for memory-mode clients
COLLECTION_NAME = "resume_chunks"


def _get_client():
    """
    Returns a ChromaDB client.

    CHROMA_MODE=memory → PersistentClient on a fresh temp directory per session.
      (We use PersistentClient + tmpdir instead of EphemeralClient because the
       latter has a known tenant-initialization bug in chromadb 0.6.3.)
    CHROMA_MODE=persist (default) → normal disk-backed client at CHROMA_PERSIST_DIR.
    """
    global _chroma_client, _temp_dir
    if _chroma_client is None:
        mode = os.getenv("CHROMA_MODE", "persist")
        if mode == "memory":
            _temp_dir = tempfile.mkdtemp(prefix="chroma_session_")
            _chroma_client = chromadb.PersistentClient(path=_temp_dir)
        else:
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
            _chroma_client = chromadb.PersistentClient(path=persist_dir)
    return _chroma_client


def reset_client():
    """Force a new client on next call — needed when switching between sessions.
    Also cleans up the memory-mode temp directory if one was created."""
    global _chroma_client, _temp_dir
    _chroma_client = None
    if _temp_dir and os.path.exists(_temp_dir):
        try:
            shutil.rmtree(_temp_dir)
        except Exception:
            pass
    _temp_dir = None


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
    """Wipe all stored chunks — useful when re-ingesting a new PDF.
    Safe to call even when the collection or tenant doesn't exist yet
    (e.g. right after reset_client() on an EphemeralClient)."""
    client = _get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection/tenant didn't exist — nothing to clean up
