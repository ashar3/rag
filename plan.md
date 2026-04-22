# PDF Knowledge Base RAG — Complete Build Plan

## What Are We Building?

A chat application where you upload a PDF (e.g., your resume) and ask questions about it in natural language.  
The app uses **Graph RAG** — meaning it doesn't just find relevant text, it understands *relationships* between concepts and answers intelligently.

---

## The Big Picture (Mental Model)

Think of it in 3 phases:

```
PDF Upload → [INGESTION PIPELINE] → Knowledge Graph + Vector Store
                                            ↓
User Question → [RETRIEVAL PIPELINE] → Relevant Context
                                            ↓
                              [LLM GENERATION] → Answer
```

---

## Component-by-Component Explanation (For Beginners)

### Phase 1: INGESTION — Turning Your PDF into Knowledge

---

#### Component 1: PDF Parsing (`pdfplumber` / `PyMuPDF`)

**What it does:** Reads the raw PDF and extracts text.  
**Why it's needed:** PDFs are not plain text — they store text as visual coordinates. We need a library to extract readable strings.  
**For your resume:** Extracts your name, skills, experience, education as raw text.

```
resume.pdf → "Anand Sharma | Software Engineer | Skills: Python, React..."
```

---

#### Component 2: Chunking (`LangChain RecursiveCharacterTextSplitter`)

**What it does:** Splits the large text into smaller overlapping pieces called "chunks".  
**Why it's needed:** LLMs have a context window limit. Also, smaller chunks = more precise retrieval.  
**Overlap:** Each chunk shares ~200 characters with the next, so context at boundaries isn't lost.

```
Full Resume Text (5000 chars)
→ Chunk 1: "...Skills: Python, React, Node..." (500 chars)
→ Chunk 2: "...React, Node, Experience at..." (500 chars, 200 overlap)
→ Chunk 3: ...
```

---

#### Component 3: Embeddings (`text-embedding-3-small` via OpenAI)

**What it does:** Converts each text chunk into a list of numbers (a "vector") that captures semantic meaning.  
**Why it's needed:** Computers can't understand text directly. Vectors let us do math — "which chunk is most similar to this question?"  
**Key idea:** Semantically similar text → numerically close vectors.

```
"Python developer" → [0.12, -0.45, 0.87, ...]   (1536 numbers)
"skilled in Python" → [0.11, -0.43, 0.85, ...]   (very close!)
"loves pizza"       → [0.91, 0.22, -0.34, ...]   (far away)
```

---

#### Component 4: Vector Store (`ChromaDB` — local, no server needed)

**What it does:** Stores all chunk embeddings in a database built for similarity search.  
**Why it's needed:** Regular databases can't do "find me the 5 most similar vectors". Chroma can.  
**For your resume:** All resume chunks are stored here, queryable by semantic similarity.

---

#### Component 5: Graph Construction (`NetworkX`) ← THE GRAPH RAG PART

**What it does:** Extracts entities (people, skills, companies, dates) and builds a knowledge graph of *relationships* between them.  
**Why it's needed:** Vector search finds similar text. Graph search understands *structure*. 

**Example for resume:**
```
Node: "Python"      — type: SKILL
Node: "Company X"   — type: EMPLOYER  
Node: "2020-2023"   — type: DATE_RANGE
Edge: "Python" → USED_AT → "Company X"
Edge: "Company X" → EMPLOYED_DURING → "2020-2023"
```

**The Calvin Cycle analogy:** If your resume says "used React at Company X" and you ask "what frontend work did you do?", Graph RAG knows React → Frontend → Company X without needing those exact words in one chunk.

**Entity Extraction:** Done with an LLM prompt asking it to pull out entities and their relationships from each chunk.

---

### Phase 2: RETRIEVAL — Finding What's Relevant

---

#### Component 6: Hybrid Retrieval

**What it does:** Combines two search strategies — vector similarity + graph traversal.  
**Why hybrid?** Vector alone misses structural relationships. Graph alone misses semantics. Together = powerful.

```
User Question: "What companies did Anand work at and what did he build there?"

Step A — Vector Search:
  → Finds chunks about work experience (semantic match)

Step B — Graph Traversal:
  → Finds EMPLOYER nodes → traverses to PROJECT edges → gets related skills/tech
  → Returns structurally connected context

Step C — Merge & Deduplicate:
  → Combines both results into a rich context window
```

---

#### Component 7: Query Understanding (Optional Enhancement)

**What it does:** Rewrites ambiguous questions to be more precise before retrieval.  
**Why it helps:** "tell me about his work" → "What positions did Anand hold, at which companies, during what time periods?"

---

### Phase 3: GENERATION — Answering the Question

---

#### Component 8: Prompt Assembly

**What it does:** Builds the final prompt sent to the LLM:
```
System: You are a helpful assistant answering questions about a resume.
Context: [retrieved chunks + graph context]
Question: [user's question]
Answer only from the context provided.
```

---

#### Component 9: LLM Response (`gpt-4o-mini` / `claude-haiku`)

**What it does:** Takes the assembled prompt and generates a natural language answer.  
**Why not just search?** Search gives you raw text. LLM synthesizes, summarizes, and answers conversationally.

---

#### Component 10: Chat History (Memory)

**What it does:** Maintains conversation turns so follow-up questions work.  
**Without it:** Each question is independent. "What did he do there?" would have no "there" reference.  
**With it:** The last N turns are included in each prompt as context.

---

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| PDF Parsing | `pdfplumber` | Clean text extraction, handles layout |
| Chunking | `LangChain` | Battle-tested, configurable overlap |
| Embeddings | `text-embedding-3-small` (OpenAI) | Fast, cheap, high quality |
| Vector DB | `ChromaDB` | Local, no infra needed, persistent |
| Graph DB | `NetworkX` | Python-native, visualizable, no server |
| Entity Extraction | `gpt-4o-mini` | Smart NER via prompting |
| LLM | `gpt-4o-mini` | Fast + cheap for Q&A |
| API Layer | `FastAPI` | Async, auto docs, easy to extend |
| Frontend | `Streamlit` | Fastest path to working UI |
| Env Management | `python-dotenv` | API keys out of code |

---

## Project File Structure

```
Rag/
├── plan.md                    ← This file
├── .env                       ← API keys (gitignored)
├── .gitignore
├── requirements.txt
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py          ← Component 1: Extract text from PDF
│   ├── chunker.py             ← Component 2: Split into chunks
│   ├── embedder.py            ← Component 3: Generate embeddings
│   ├── vector_store.py        ← Component 4: Store/query in ChromaDB
│   └── graph_builder.py       ← Component 5: Build knowledge graph
│
├── retrieval/
│   ├── __init__.py
│   ├── hybrid_retriever.py    ← Component 6: Vector + Graph retrieval
│   └── query_processor.py    ← Component 7: Query rewriting
│
├── generation/
│   ├── __init__.py
│   ├── prompt_builder.py      ← Component 8: Assemble final prompt
│   └── llm_client.py          ← Component 9: Call LLM
│
├── memory/
│   ├── __init__.py
│   └── chat_history.py        ← Component 10: Conversation memory
│
├── api/
│   ├── __init__.py
│   └── main.py                ← FastAPI endpoints
│
├── ui/
│   └── app.py                 ← Streamlit chat interface
│
├── data/
│   └── uploads/               ← Uploaded PDFs stored here
│
└── tests/
    ├── test_ingestion.py
    └── test_retrieval.py
```

---

## Build Order (Step by Step)

```
Step 1: Setup — requirements.txt, .env, .gitignore
Step 2: PDF Parsing — can we read the resume?
Step 3: Chunking — can we split it intelligently?
Step 4: Embeddings + ChromaDB — can we store and query?
Step 5: Graph Building — can we extract entities + relationships?
Step 6: Hybrid Retrieval — can we combine both search types?
Step 7: LLM Generation — can we get a coherent answer?
Step 8: Chat Memory — do follow-up questions work?
Step 9: FastAPI — expose as an API
Step 10: Streamlit UI — make it usable
Step 11: Upload your resume → test questions
```

---

## Test Questions for Your Resume

Once built, try these to validate each component:

| Question | Tests |
|---|---|
| "What programming languages does Anand know?" | Basic vector retrieval |
| "What did he build at his most recent company?" | Graph traversal (company → projects) |
| "How many years of experience does he have?" | Reasoning over structured data |
| "What's his educational background?" | Section-specific retrieval |
| "Is he a good fit for a backend role?" | Synthesis across multiple chunks |
| "What did you just tell me?" | Chat memory |

---

## Key Concepts Glossary

- **RAG** (Retrieval Augmented Generation): Instead of the LLM guessing, we *give* it relevant text to answer from. Reduces hallucination.
- **Embedding**: Turning text into numbers that capture meaning. Similar meaning = similar numbers.
- **Vector Store**: A database optimized for finding numerically similar vectors fast.
- **Graph RAG**: Enhancing RAG with a knowledge graph so the system understands *how concepts relate*, not just *what is similar*.
- **Chunk**: A small piece of your document. Smaller = more precise retrieval but loses broader context.
- **Hybrid Retrieval**: Using both semantic similarity (vectors) and structural relationships (graph) together.
- **Context Window**: The maximum amount of text an LLM can read at once. RAG solves this by only passing *relevant* text.
