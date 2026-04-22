"""
STEP 1 — PDF PARSER
Analogy: Your resume is a locked book. pdfplumber is the key that opens it
and reads every page out loud. We get back plain text we can actually work with.
"""

import pdfplumber
from pathlib import Path


def parse_pdf(pdf_path: str) -> dict:
    """
    Opens a PDF and extracts:
    - full_text : everything concatenated (used for chunking)
    - pages     : list of {page_num, text} (useful for metadata later)
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []
    full_text_parts = []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page_num": i + 1, "text": text})
                full_text_parts.append(text)

    full_text = "\n\n".join(full_text_parts)

    return {
        "full_text": full_text,
        "pages": pages,
        "total_pages": len(pages),
        "source": path.name,
    }
