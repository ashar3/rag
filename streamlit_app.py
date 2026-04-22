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
    """
    Step-by-step learning commentary — explains what happened for THIS question,
    as if the reader is seeing RAG for the first time.
    """
    query        = detective_report.get("query", "")
    vector_hits  = detective_report.get("vector", [])
    bm25_hits    = detective_report.get("bm25", [])
    graph_hits   = detective_report.get("graph", [])
    entities     = detective_report.get("query_entities", [])
    query_tokens = detective_report.get("query_tokens", [])

    with st.expander("🧠 How did the system find this answer? (Step-by-step walkthrough)", expanded=False):

        # ── Overview diagram ──────────────────────────────────────────────────
        st.markdown("### The 3-Detective Pipeline")
        st.markdown(f"""
Your question **"{query}"** was processed by 3 independent detectives.
Each one searches the resume differently. Their findings are combined before the LLM answers.

```
"{query}"
       │
       ├── 🔵 Detective A (Vector)  → found {len(vector_hits)} chunks
       ├── 🟡 Detective B (BM25)    → found {len(bm25_hits)} chunks
       └── 🟢 Detective C (Graph)   → found {len(graph_hits)} chunks
                     │
                     ▼
             ⚖️  Reciprocal Rank Fusion
                     │
                     ▼
             📋 Best chunks → GPT-4o-mini → Answer
```
""")
        st.divider()

        # ── STEP 1: Vector ────────────────────────────────────────────────────
        st.markdown("### 🔵 Step 1 — Detective A: Vector Search")
        st.markdown("""
**What it does:**
Think of every piece of text in your resume as a location on a map — but the map
measures *meaning*, not geography. "Python developer" and "skilled in Python" land
very close together. "loves pizza" lands far away.

When you ask a question, it gets its own location on this map.
Detective A finds the resume pieces *closest to your question's location*.

**Why this is powerful:** It understands synonyms and paraphrasing.
You can ask "coding experience" and it finds "software development background"
without those exact words appearing together.

**The math behind it:**
```
question → [0.12, -0.45, 0.87, ...]   1536 numbers = your question's location
chunk    → [0.11, -0.43, 0.85, ...]   very close = very relevant
chunk    → [0.91,  0.22, -0.34, ...]  far away   = not relevant
```
""")
        if vector_hits:
            st.success(f"✅ Found {len(vector_hits)} semantically similar chunks")
            for i, h in enumerate(vector_hits):
                dist = h.get("score")
                relevance = "🟢 Very relevant" if dist and dist < 0.3 else "🟡 Somewhat relevant" if dist and dist < 0.6 else "🔴 Weak match"
                with st.expander(f"Result {i+1} — distance={dist}  {relevance}"):
                    st.markdown(f"**Distance from your question:** `{dist}`  *(0 = identical meaning, 1 = opposite meaning)*")
                    st.text(h["text"])
        else:
            st.warning("No vector results — ChromaDB may be empty. Try re-ingesting the PDF.")
        st.divider()

        # ── STEP 2: BM25 ─────────────────────────────────────────────────────
        st.markdown("### 🟡 Step 2 — Detective B: BM25 Keyword Search")
        st.markdown(f"""
**What it does:**
Detective B doesn't understand meaning — it reads every word literally.
But it's smarter than simple keyword counting.

**BM25 rewards words that are:**
1. **Rare** across all resume chunks (specific = valuable)
2. **Frequent** in this particular chunk (relevant = useful)

**Your question was tokenized into:** `{query_tokens if query_tokens else query.lower().split()}`

Each token is looked up across all resume chunks. Common words like "what", "is", "the"
score near zero because they appear everywhere. Specific words like "AWS" or "React"
score high because they're rare and precise.

**Why Vector alone isn't enough:**
If your resume says "AWS" and you ask "AWS", Vector might miss it
because "AWS" is a rare acronym with weak embedding signal.
BM25 locks onto exact characters instantly.
""")
        if bm25_hits:
            st.success(f"✅ Found {len(bm25_hits)} keyword-matching chunks")
            for i, h in enumerate(bm25_hits):
                score = h.get("score")
                with st.expander(f"Result {i+1} — BM25 score={score}"):
                    st.markdown(f"**BM25 score:** `{score}`  *(higher = better keyword match weighted by rarity)*")
                    st.text(h["text"])
        else:
            st.info("No BM25 results — none of your question's keywords appeared in any chunk.")
        st.divider()

        # ── STEP 3: Graph ─────────────────────────────────────────────────────
        st.markdown("### 🟢 Step 3 — Detective C: Knowledge Graph Traversal")
        st.markdown("""
**What it does:**
At ingest time, we asked an LLM to read every chunk and extract:
- **Entities:** people, companies, skills, dates, roles, projects
- **Relationships:** who worked where, which skills were used at which company

This built a knowledge graph — like a family tree but for resume concepts:
```
"Python" ──[USED_AT]──▶ "Company X"
"Company X" ──[EMPLOYED]──▶ "Anand"
"Company X" ──[DURING]──▶ "2020-2023"
"React" ──[USED_AT]──▶ "Company X"
```

**At query time:** Detective C looks for entities from your question in the graph,
then walks outward up to 2 hops to collect all connected chunks.

**Why this matters:**
You ask "what did he build?" — the graph knows:
`build → Project Y → Company X → other context about Company X`
...even if those chunks share zero words with "build".
""")
        if entities:
            st.success(f"✅ Matched {len(entities)} entities from your question in the graph")
            st.markdown(f"**Entities found:** {' · '.join([f'`{e}`' for e in entities])}")
            st.markdown("*These became the starting nodes for the graph walk (2 hops outward)*")
        else:
            st.warning("⚠️ No entities from your question matched graph nodes. Try using specific names, company names, or skill names.")

        if graph_hits:
            st.success(f"✅ Graph traversal returned {len(graph_hits)} connected chunks")
            for i, h in enumerate(graph_hits):
                with st.expander(f"Result {i+1} — found via graph relationship"):
                    st.markdown("*This chunk was reached by following edges from a matched entity, not by text similarity.*")
                    st.text(h["text"])
        else:
            st.info("No graph chunks returned for this query.")
        st.divider()

        # ── STEP 4: RRF ───────────────────────────────────────────────────────
        st.markdown("### ⚖️ Step 4 — Reciprocal Rank Fusion")
        st.markdown(f"""
**The problem:** Each detective returned a ranked list. How do we combine 3 lists fairly?

**Naive approach (wrong):** Concatenate → duplicates, no re-ranking.
**Score averaging (fragile):** Different scales — BM25 score of 4.2 vs cosine distance of 0.3 can't be averaged.

**RRF (right):** Convert every result to a *rank-based score*, then add.
```
rrf_score += 1 / (60 + rank)   ← applied once per detective that found this chunk
```

**Example with your query:**
- Chunk found by Vector at rank 1: `1/(60+1) = 0.0164`
- Chunk found by BM25 at rank 2:  `1/(60+2) = 0.0161`
- Same chunk found by BOTH:        `0.0164 + 0.0161 = 0.0325` → **rises to top**
- Chunk found by all 3 detectives: even better → **almost certainly the answer**

**Why 60?** The constant smooths the curve so rank-1 doesn't completely
dominate rank-2. Without it, only rank-1 results from each detective would matter.

**Total unique chunks sent to LLM:** `{len(vector_hits) + len(bm25_hits) + len(graph_hits)}` raw → merged & re-ranked by RRF
""")
        st.divider()

        # ── STEP 5: LLM ──────────────────────────────────────────────────────
        st.markdown("### 🤖 Step 5 — LLM Generation")
        st.markdown("""
**What happens now:**
The top-ranked chunks from RRF are assembled into a prompt:
```
System: You are a helpful assistant. Answer ONLY from the context below.
        [Context: top chunks from RRF, tagged VECTOR/BM25/GRAPH]

User:   [your question]
```

**Why "answer ONLY from context"?**
Without this instruction, GPT might hallucinate — invent plausible-sounding
details that aren't in the resume. By restricting it to the retrieved chunks,
the answer is *grounded* in your actual document.

**This is the core promise of RAG:**
> Don't trust the model's memory. Give it the evidence and make it reason from that.
""")
        st.success("✅ Answer generated from the merged context above")


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
                    "query":          question,
                    "query_tokens":   question.lower().split(),
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
