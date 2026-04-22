"""
STREAMLIT CLOUD ENTRY POINT
Calls pipeline functions directly — no FastAPI needed.
State lives in st.session_state (per browser session).

On Streamlit Cloud:
  - OPENAI_API_KEY comes from Streamlit Secrets (Settings → Secrets)
  - ChromaDB runs in-memory (CHROMA_MODE=memory, set via os.environ below)
  - Each user session gets its own isolated state via st.session_state
"""

import os
import tempfile
import streamlit as st

# Tell ChromaDB to run in-memory — no disk needed on Streamlit Cloud
os.environ["CHROMA_MODE"] = "memory"

# Pull OpenAI key from Streamlit Secrets if available
if "OPENAI_API_KEY" in st.secrets:
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ── Pipeline imports (called directly, no HTTP) ────────────────────────────────
from ingestion.pdf_parser import parse_pdf
from ingestion.chunker import chunk_text
from ingestion.embedder import embed_texts
from ingestion.vector_store import store_chunks, clear_collection, reset_client
from ingestion.graph_builder import build_graph
from ingestion.bm25_index import build_bm25_index
from retrieval.hybrid_retriever import retrieve
from generation.prompt_builder import build_prompt
from generation.llm_client import generate_answer
from memory.chat_history import ChatHistory

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")
st.title("📄 Resume Chat — 3-Detective RAG")
st.caption("Upload a resume PDF and ask anything about it.")

# ── Session state init ─────────────────────────────────────────────────────────
# Each browser tab gets its own isolated session
if "chunks"      not in st.session_state: st.session_state.chunks      = []
if "graph"       not in st.session_state: st.session_state.graph       = None
if "bm25_index"  not in st.session_state: st.session_state.bm25_index  = None
if "history"     not in st.session_state: st.session_state.history     = ChatHistory()
if "pdf_name"    not in st.session_state: st.session_state.pdf_name    = None
if "messages"    not in st.session_state: st.session_state.messages    = []
if "chroma_ready" not in st.session_state: st.session_state.chroma_ready = False


# ── INGESTION ──────────────────────────────────────────────────────────────────

def run_ingestion(uploaded_file) -> dict:
    """
    Full pipeline: PDF → chunks → embeddings → ChromaDB → graph → BM25
    Runs in the Streamlit process directly (no HTTP call).
    """
    # Save upload to a temp file (Streamlit gives us a BytesIO object)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    parsed     = parse_pdf(tmp_path)
    chunks     = chunk_text(parsed["full_text"], source=uploaded_file.name)
    texts      = [c["text"] for c in chunks]
    embeddings = embed_texts(texts)

    # Reset ChromaDB client so we get a fresh in-memory instance
    reset_client()
    clear_collection()
    store_chunks(chunks, embeddings)

    graph      = build_graph(chunks)
    bm25_index = build_bm25_index(chunks)

    os.unlink(tmp_path)  # clean up temp file

    # Store everything in session state
    st.session_state.chunks       = chunks
    st.session_state.graph        = graph
    st.session_state.bm25_index   = bm25_index
    st.session_state.pdf_name     = uploaded_file.name
    st.session_state.chroma_ready = True
    st.session_state.history      = ChatHistory()
    st.session_state.messages     = []

    return {
        "total_chunks": len(chunks),
        "graph_nodes":  graph.number_of_nodes(),
        "graph_edges":  graph.number_of_edges(),
    }


# ── DETECTIVE COMMENTARY ───────────────────────────────────────────────────────

def show_detective_report(detective_report: dict):
    st.markdown("---")
    st.markdown("### 🕵️ Detective Report — How This Answer Was Found")
    st.markdown("""
```
Query
  │
  ├── 🔵 Detective A (Vector)   → semantic similarity via ChromaDB
  ├── 🟡 Detective B (BM25)     → keyword scoring via rank_bm25
  └── 🟢 Detective C (Graph)    → relationship traversal via NetworkX
                │
                ▼
        ⚖️ Reciprocal Rank Fusion  →  score = 1 / (60 + rank)
                │
                ▼
        📋 Merged context → LLM
```
""")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🔵 Detective A — Vector")
        st.caption("Finds chunks that MEAN the same as your question, even different words.")
        st.caption("*'software engineer' matches 'developer'*")
        for i, h in enumerate(detective_report.get("vector", [])):
            with st.expander(f"Result {i+1}  ·  dist={h.get('score', '?')}"):
                st.text(h["text"])
        if not detective_report.get("vector"):
            st.info("No results")

    with col2:
        st.markdown("#### 🟡 Detective B — BM25")
        st.caption("Finds chunks with EXACT keywords, weighted by rarity.")
        st.caption("*Great for: AWS, React, phone numbers, proper nouns*")
        for i, h in enumerate(detective_report.get("bm25", [])):
            with st.expander(f"Result {i+1}  ·  score={h.get('score', '?')}"):
                st.text(h["text"])
        if not detective_report.get("bm25"):
            st.info("No keyword overlap found")

    with col3:
        st.markdown("#### 🟢 Detective C — Graph")
        entities = detective_report.get("query_entities", [])
        st.caption("Walks knowledge graph outward from entities in your question.")
        st.caption(f"*Matched: {', '.join(entities) if entities else 'none'}*")
        for i, h in enumerate(detective_report.get("graph", [])):
            with st.expander(f"Result {i+1}"):
                st.text(h["text"])
        if not detective_report.get("graph"):
            st.info("No graph connections found")

    st.markdown("#### ⚖️ Reciprocal Rank Fusion")
    st.markdown("""
Each detective submits a ranked list. RRF scores every chunk:
```
rrf_score += 1 / (60 + rank)   ← for each detective that found this chunk
```
A chunk found by **all 3 detectives** beats a chunk found by only **1**, even at rank 1.
The constant **60** prevents any single rank-1 result from dominating everything.
""")


# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("Ingest PDF", type="primary"):
        with st.spinner("Parsing → Chunking → Embedding → Graph → BM25..."):
            try:
                result = run_ingestion(uploaded_file)
                st.success(
                    f"✅ Done!\n\n"
                    f"- **Chunks:** {result['total_chunks']}\n"
                    f"- **Graph nodes:** {result['graph_nodes']}\n"
                    f"- **Graph edges:** {result['graph_edges']}"
                )
            except Exception as e:
                st.error(f"Ingestion failed: {type(e).__name__}: {e}")

    st.divider()

    if st.session_state.pdf_name:
        st.success(f"**Loaded:** {st.session_state.pdf_name}")
        st.metric("Chunks", len(st.session_state.chunks))
        st.metric("Graph nodes", st.session_state.graph.number_of_nodes() if st.session_state.graph else 0)
        st.metric("Chat turns", len(st.session_state.history))
    else:
        st.info("No PDF loaded yet.")

    st.divider()
    if st.button("🔄 Reset / New PDF"):
        reset_client()
        st.session_state.chunks       = []
        st.session_state.graph        = None
        st.session_state.bm25_index   = None
        st.session_state.history      = ChatHistory()
        st.session_state.pdf_name     = None
        st.session_state.messages     = []
        st.session_state.chroma_ready = False
        st.rerun()


# ── CHAT ───────────────────────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("detective_report"):
            show_detective_report(msg["detective_report"])

if question := st.chat_input("Ask about the resume..."):
    if not st.session_state.chunks:
        st.warning("Please upload and ingest a PDF first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("3 detectives working..."):
            try:
                retrieval_result = retrieve(
                    query=question,
                    graph=st.session_state.graph,
                    all_chunks=st.session_state.chunks,
                    bm25_index=st.session_state.bm25_index,
                )
                merged_chunks = retrieval_result["merged"]

                messages = build_prompt(
                    query=question,
                    retrieved_chunks=merged_chunks,
                    chat_history=st.session_state.history.get(),
                )
                answer = generate_answer(messages)

                st.session_state.history.add("user", question)
                st.session_state.history.add("assistant", answer)

                def fmt(hits):
                    return [{"text": h["text"][:150], "source": h.get("retrieval_source","?"),
                             "score": h.get("bm25_score") or h.get("distance") or h.get("rrf_score")}
                            for h in hits]

                detective_report = {
                    "vector":         fmt(retrieval_result["detective_a_vector"]),
                    "bm25":           fmt(retrieval_result["detective_b_bm25"]),
                    "graph":          fmt(retrieval_result["detective_c_graph"]),
                    "query_entities": retrieval_result["query_entities"],
                }

                st.markdown(answer)
                show_detective_report(detective_report)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "detective_report": detective_report,
                })

            except Exception as e:
                st.error(f"Chat failed: {type(e).__name__}: {e}")
