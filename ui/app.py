"""
STREAMLIT UI — Resume Chat with 3-Detective Learning Commentary
Shows the full retrieval pipeline visually after every answer.
"""

import streamlit as st
import requests

API_BASE = "http://127.0.0.1:8000"

st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")
st.title("📄 Resume Chat — Graph RAG + BM25")
st.caption("Upload your resume PDF and ask anything about it.")

# ── SIDEBAR ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("Ingest PDF", type="primary"):
        with st.spinner("Parsing → Chunking → Embedding → Graph → BM25 index..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    timeout=180,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(
                        f"✅ Done!\n\n"
                        f"- **Chunks:** {data['total_chunks']}\n"
                        f"- **Graph nodes:** {data['graph_nodes']}\n"
                        f"- **Graph edges:** {data['graph_edges']}"
                    )
                    st.session_state["pdf_ready"] = True
                    st.session_state["messages"] = []
                else:
                    try:
                        detail = resp.json().get("detail", resp.text)
                    except Exception:
                        detail = resp.text or f"HTTP {resp.status_code}"
                    st.error(f"Error: {detail}")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API on 127.0.0.1:8000 — is uvicorn running?")
            except Exception as e:
                st.error(f"Ingest failed: {type(e).__name__}: {e}")

    st.divider()

    try:
        status = requests.get(f"{API_BASE}/status", timeout=5).json()
        if status["pdf_loaded"]:
            st.success(f"**Loaded:** {status['pdf_loaded']}")
            st.metric("Chunks", status["chunks"])
            st.metric("Graph nodes", status["graph_nodes"])
            st.metric("Chat turns", status["history_turns"])
        else:
            st.info("No PDF loaded yet.")
    except Exception:
        st.warning("API not running.\n\n`uvicorn api.main:app --reload --host 0.0.0.0`")

    st.divider()
    if st.button("🔄 Reset / New PDF"):
        try:
            requests.delete(f"{API_BASE}/reset", timeout=10)
            st.session_state["messages"] = []
            st.session_state["pdf_ready"] = False
            st.rerun()
        except Exception:
            pass

# ── DETECTIVE COMMENTARY COMPONENT ────────────────────────────────────────────

def show_detective_report(detective_report: dict):
    """
    Renders the full 3-detective system visually as a learning panel.
    Shows what each detective found independently, then how RRF fused them.
    """
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

    # ── Detective A: Vector ──
    with col1:
        st.markdown("#### 🔵 Detective A — Vector")
        st.caption("Finds chunks that MEAN the same as your question, even with different words.")
        st.caption("*'software engineer' matches 'developer'*")
        hits = detective_report.get("vector", [])
        if hits:
            for i, h in enumerate(hits):
                with st.expander(f"Result {i+1}  ·  dist={h.get('score', '?')}"):
                    st.text(h["text"])
        else:
            st.info("No results")

    # ── Detective B: BM25 ──
    with col2:
        st.markdown("#### 🟡 Detective B — BM25")
        st.caption("Finds chunks with the EXACT keywords, weighted by rarity.")
        st.caption("*Great for: AWS, React, phone numbers, proper nouns*")
        hits = detective_report.get("bm25", [])
        if hits:
            for i, h in enumerate(hits):
                with st.expander(f"Result {i+1}  ·  score={h.get('score', '?')}"):
                    st.text(h["text"])
        else:
            st.info("No keyword overlap found")

    # ── Detective C: Graph ──
    with col3:
        st.markdown("#### 🟢 Detective C — Graph")
        entities = detective_report.get("query_entities", [])
        st.caption("Walks the knowledge graph outward from entities in your question.")
        if entities:
            st.caption(f"*Matched entities: {', '.join(entities)}*")
        else:
            st.caption("*No entities from question matched graph nodes*")
        hits = detective_report.get("graph", [])
        if hits:
            for i, h in enumerate(hits):
                with st.expander(f"Result {i+1}"):
                    st.text(h["text"])
        else:
            st.info("No graph connections found")

    # ── RRF Explanation ──
    st.markdown("#### ⚖️ Reciprocal Rank Fusion — How Scores Are Combined")
    st.markdown("""
Each detective submits their ranked list. RRF scores each chunk as:

```
rrf_score += 1 / (60 + rank)   ← for each detective that found it
```

**Why this works:**
- A chunk found by **all 3 detectives** gets 3× contribution → rises to top
- A chunk found by **only 1 detective** at rank 1 scores `1/61 = 0.016`
- A chunk found by **all 3** even at rank 5 scores `3 × 1/65 = 0.046` → wins

The constant **60** prevents rank-1 from completely dominating rank-2.
It smooths the curve so agreement across detectives beats dominance from one.
""")


# ── CHAT INTERFACE ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("detective_report"):
            show_detective_report(msg["detective_report"])

# Chat input
if question := st.chat_input("Ask about the resume..."):
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("3 detectives working..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": question},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    detective_report = data.get("detective_report", {})

                    st.markdown(answer)
                    show_detective_report(detective_report)

                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "detective_report": detective_report,
                    })
                else:
                    try:
                        err = resp.json().get("detail", resp.text)
                    except Exception:
                        err = resp.text or f"HTTP {resp.status_code}"
                    st.error(err)
                    st.session_state["messages"].append({"role": "assistant", "content": f"Error: {err}"})
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API on 127.0.0.1:8000 — is uvicorn running?")
            except Exception as e:
                st.error(f"Chat failed: {type(e).__name__}: {e}")
