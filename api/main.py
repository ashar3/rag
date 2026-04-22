"""
STEP 8 — FASTAPI BACKEND
Analogy: The factory floor manager — connects all rooms and exposes
two doors to the outside world:
  POST /ingest  → upload PDF, run the full ingestion pipeline
  POST /chat    → ask a question, get an answer

Everything flows through here in the right order.
"""

import os
import shutil
from pathlib import Path

import networkx as nx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_texts
from ingestion.vector_store import store_chunks, clear_collection
from ingestion.graph_builder import build_graph
from retrieval.hybrid_retriever import retrieve
from generation.prompt_builder import build_prompt
from generation.llm_client import generate_answer
from memory.chat_history import ChatHistory

load_dotenv()

app = FastAPI(title="PDF RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-memory state (single user, single PDF session) ---
# In production: store per session_id in Redis/DB
_state: dict = {
    "chunks": [],        # all chunks from the ingested PDF
    "graph": None,       # NetworkX DiGraph
    "history": ChatHistory(),
    "pdf_name": None,
}

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./data/uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# ── INGESTION ENDPOINT ─────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF → parse → chunk → embed → store in ChromaDB → build graph.
    This is the full ingestion pipeline triggered by one API call.
    """
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save uploaded file to disk
    save_path = UPLOAD_DIR / file.filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Step 1: Parse PDF → raw text
    parsed = parse_pdf(str(save_path))

    # Step 2: Chunk the text
    chunks = chunk_text(parsed["full_text"], source=parsed["source"])

    # Step 3: Embed all chunks
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Step 4: Clear old data and store new chunks in ChromaDB
    clear_collection()
    store_chunks(chunks, embeddings)

    # Step 5: Build knowledge graph from chunks
    graph = build_graph(chunks)

    # Store in session state
    _state["chunks"] = chunks
    _state["graph"] = graph
    _state["pdf_name"] = parsed["source"]
    _state["history"].clear()

    return {
        "status": "ok",
        "pdf": parsed["source"],
        "total_pages": parsed["total_pages"],
        "total_chunks": len(chunks),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
    }


# ── CHAT ENDPOINT ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str


@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Receive a question → retrieve relevant chunks (hybrid) → generate answer.
    Chat history is maintained automatically between calls.
    """
    if not _state["chunks"]:
        raise HTTPException(status_code=400, detail="No PDF ingested yet. Call /ingest first.")

    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Step 6: Hybrid retrieval (vector + graph)
    retrieved = retrieve(
        query=question,
        graph=_state["graph"],
        all_chunks=_state["chunks"],
    )

    # Step 7: Build prompt with context + history
    messages = build_prompt(
        query=question,
        retrieved_chunks=retrieved,
        chat_history=_state["history"].get(),
    )

    # Step 8: Generate answer
    answer = generate_answer(messages)

    # Step 9: Save to history for follow-up questions
    _state["history"].add("user", question)
    _state["history"].add("assistant", answer)

    return {
        "answer": answer,
        "sources": [
            {"text": c["text"][:120] + "...", "source": c.get("retrieval_source", "?")}
            for c in retrieved
        ],
    }


# ── UTILITY ENDPOINTS ──────────────────────────────────────────────────────────

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
    _state["chunks"] = []
    _state["graph"] = None
    _state["history"].clear()
    _state["pdf_name"] = None
    return {"status": "reset"}
