"""
FASTAPI BACKEND — The Factory Floor Manager
Connects all rooms, exposes two main doors:
  POST /ingest  → upload PDF → full ingestion pipeline
  POST /chat    → ask question → 3-detective retrieval → LLM answer + detective report
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_texts
from ingestion.vector_store import store_chunks, clear_collection
from ingestion.graph_builder import build_graph
from ingestion.bm25_index import build_bm25_index
from retrieval.hybrid_retriever import retrieve
from generation.prompt_builder import build_prompt
from generation.llm_client import generate_answer
from memory.chat_history import ChatHistory

load_dotenv()

app = FastAPI(title="PDF RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session state (single user). Production: Redis per session_id.
_state: dict = {
    "chunks": [],
    "graph": None,
    "bm25_index": None,   # NEW: BM25 index built at ingest time
    "history": ChatHistory(),
    "pdf_name": None,
}

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── INGESTION ──────────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    parsed   = parse_pdf(str(save_path))
    chunks   = chunk_text(parsed["full_text"], source=parsed["source"])
    texts    = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    clear_collection()
    store_chunks(chunks, embeddings)

    graph       = build_graph(chunks)
    bm25_index  = build_bm25_index(chunks)   # NEW: build BM25 index

    _state["chunks"]     = chunks
    _state["graph"]      = graph
    _state["bm25_index"] = bm25_index
    _state["pdf_name"]   = parsed["source"]
    _state["history"].clear()

    return {
        "status": "ok",
        "pdf": parsed["source"],
        "total_pages": parsed["total_pages"],
        "total_chunks": len(chunks),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
    }


# ── CHAT ───────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):
    if not _state["chunks"]:
        raise HTTPException(status_code=400, detail="No PDF ingested yet. Call /ingest first.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # 3-detective retrieval — returns merged result + each detective's raw findings
    retrieval_result = retrieve(
        query=question,
        graph=_state["graph"],
        all_chunks=_state["chunks"],
        bm25_index=_state["bm25_index"],
    )

    merged_chunks = retrieval_result["merged"]

    messages = build_prompt(
        query=question,
        retrieved_chunks=merged_chunks,
        chat_history=_state["history"].get(),
    )

    answer = generate_answer(messages)

    _state["history"].add("user", question)
    _state["history"].add("assistant", answer)

    def format_hits(hits: list[dict]) -> list[dict]:
        return [
            {
                "text": h["text"][:150] + ("..." if len(h["text"]) > 150 else ""),
                "source": h.get("retrieval_source", "?"),
                "score": h.get("bm25_score") or h.get("distance") or h.get("rrf_score"),
            }
            for h in hits
        ]

    return {
        "answer": answer,
        # final merged context used by LLM
        "merged_sources": format_hits(merged_chunks),
        # each detective's individual findings — for the learning commentary UI
        "detective_report": {
            "vector":  format_hits(retrieval_result["detective_a_vector"]),
            "bm25":    format_hits(retrieval_result["detective_b_bm25"]),
            "graph":   format_hits(retrieval_result["detective_c_graph"]),
            "query_entities": retrieval_result["query_entities"],
        },
    }


# ── UTILITIES ──────────────────────────────────────────────────────────────────

@app.get("/status")
def status():
    return {
        "pdf_loaded": _state["pdf_name"],
        "chunks": len(_state["chunks"]),
        "graph_nodes": _state["graph"].number_of_nodes() if _state["graph"] else 0,
        "history_turns": len(_state["history"]),
    }


@app.delete("/reset")
def reset():
    clear_collection()
    _state["chunks"]     = []
    _state["graph"]      = None
    _state["bm25_index"] = None
    _state["history"].clear()
    _state["pdf_name"]   = None
    return {"status": "reset"}
