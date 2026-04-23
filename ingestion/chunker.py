"""
STEP 2 — CHUNKER
Analogy: You can't search a 10-page resume instantly. So we cut it into
small index cards (chunks). Each card overlaps the next slightly so we
never lose a sentence that sits on a boundary between two cards.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(full_text: str, source: str, chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """
    Splits text into overlapping chunks and attaches metadata.

    chunk_size    = max characters per chunk (~100-150 words)
    chunk_overlap = shared characters between consecutive chunks
                    so a sentence crossing a boundary isn't lost

    Returns list of:
        { "text": str, "chunk_index": int, "source": str }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # tries to split on paragraph → sentence → word (in that order)
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    raw_chunks = splitter.split_text(full_text)

    chunks = []
    for i, text in enumerate(raw_chunks):
        text = text.strip()
        if len(text) > 30:  # skip tiny fragments
            chunks.append({
                "text": text,
                "chunk_index": i,
                "source": source,
            })

    return chunks
