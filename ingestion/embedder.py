"""
STEP 3a — EMBEDDER
Analogy: Each chunk gets GPS coordinates in "meaning space".
"Software engineer" and "developer" land very close. "banana" lands far away.
We use OpenAI's model to generate these coordinates (called embeddings).
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Converts a list of text strings into embeddings.
    Batches up to 100 at a time (OpenAI limit per call).
    Returns list of float vectors, one per input text.
    """
    client = _get_client()
    all_embeddings = []

    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """Single query embedding — used at retrieval time."""
    return embed_texts([query])[0]
