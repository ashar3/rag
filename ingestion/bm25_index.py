"""
BM25 INDEX — Detective B: The Photographic Memory
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Analogy: Think of a very fast librarian who has memorized every word in every
document. When you ask about "AWS", she doesn't ponder meaning — she instantly
pulls every card that contains the exact letters A-W-S.

Why BM25 over plain keyword count?
  Plain count:  "the the the Python" scores 3 for "the" — wrong, "the" is everywhere
  BM25 thinks:  "Python" is rare across all chunks → reward it more
                "the" appears in every chunk → nearly worthless signal
                Longer chunks get penalized so short precise chunks aren't buried

Formula (simplified):
  score = (word_frequency × (k1+1))           ← boosts rare words in this chunk
          ─────────────────────────────── × IDF  ← multiplied by how rare across ALL chunks
          (word_frequency + k1×(1 - b + b×L))

  k1 = 1.5  (term frequency saturation — stops a word appearing 100x from dominating)
  b  = 0.75 (length normalization — penalizes very long chunks)
  L  = chunk_length / avg_chunk_length
"""

from rank_bm25 import BM25Okapi


def build_bm25_index(chunks: list[dict]) -> BM25Okapi:
    """
    At ingest time: tokenize every chunk and build the BM25 index.
    Tokenization here is simple (lowercase + split) — good enough for resumes.
    Production systems use stemming (running→run) and stopword removal.
    """
    tokenized_corpus = [chunk["text"].lower().split() for chunk in chunks]
    return BM25Okapi(tokenized_corpus)


def query_bm25(index: BM25Okapi, chunks: list[dict], query: str, top_k: int = 5) -> list[dict]:
    """
    At query time: tokenize the question the same way, score every chunk,
    return top_k with their BM25 scores attached.

    Score = 0 means the chunk shares zero words with the query → excluded.
    Higher score = better keyword overlap weighted by rarity.
    """
    query_tokens = query.lower().split()
    scores = index.get_scores(query_tokens)

    # pair each chunk with its score, sort descending, take top_k
    scored = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    results = []
    for idx, score in scored:
        if score > 0:  # skip zero-score chunks — no keyword overlap at all
            hit = dict(chunks[idx])
            hit["bm25_score"] = round(float(score), 4)
            hit["retrieval_source"] = "bm25"
            results.append(hit)

    return results
