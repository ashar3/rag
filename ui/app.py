"""
STEP 9 — STREAMLIT UI
Analogy: The reception desk. You upload your resume through a slot,
then chat with it through a speaker. Everything complex is hidden behind the wall.
"""

import streamlit as st
import requests

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")
st.title("📄 Resume Chat — Graph RAG")
st.caption("Upload your resume PDF and ask anything about it.")

# ── SIDEBAR: Upload + Status ───────────────────────────────────────────────────

with st.sidebar:
    st.header("1. Upload Resume")
    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file and st.button("Ingest PDF", type="primary"):
        with st.spinner("Reading PDF → Chunking → Embedding → Building graph..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/ingest",
                    files={"file": (uploaded_file.name, uploaded_file, "application/pdf")},
                    timeout=120,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.success(f"Done! {data['total_chunks']} chunks, {data['graph_nodes']} graph nodes")
                    st.session_state["pdf_ready"] = True
                    st.session_state["messages"] = []
                else:
                    st.error(f"Error: {resp.json().get('detail', 'Unknown error')}")
            except Exception as e:
                st.error(f"Could not reach API: {e}")

    st.divider()

    # Show current status
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
        st.warning("API not running. Start with:\n`uvicorn api.main:app --reload`")

    st.divider()
    if st.button("Reset / New PDF"):
        try:
            requests.delete(f"{API_BASE}/reset", timeout=10)
            st.session_state["messages"] = []
            st.session_state["pdf_ready"] = False
            st.rerun()
        except Exception:
            pass

# ── MAIN: Chat Interface ───────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("View retrieved sources"):
                for i, src in enumerate(msg["sources"]):
                    tag = "🔵 Vector" if src["source"] == "vector" else "🟢 Graph"
                    st.markdown(f"**{tag} — Result {i+1}**")
                    st.text(src["text"])

# Chat input
if question := st.chat_input("Ask about the resume..."):
    # Show user message immediately
    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Get answer from API
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_BASE}/chat",
                    json={"question": question},
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    st.markdown(answer)
                    if sources:
                        with st.expander("View retrieved sources"):
                            for i, src in enumerate(sources):
                                tag = "🔵 Vector" if src["source"] == "vector" else "🟢 Graph"
                                st.markdown(f"**{tag} — Result {i+1}**")
                                st.text(src["text"])
                    st.session_state["messages"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                else:
                    err = resp.json().get("detail", "Unknown error")
                    st.error(err)
                    st.session_state["messages"].append({"role": "assistant", "content": f"Error: {err}"})
            except Exception as e:
                st.error(f"Could not reach API: {e}")
