"""
STEP 6a — PROMPT BUILDER
Analogy: A lawyer arranging evidence before presenting it to a judge.
We take raw retrieved chunks and format them into a structured prompt
so the LLM has clear context boundaries — it knows exactly what it can
and cannot use to answer.
"""


SYSTEM_PROMPT = """You are a helpful assistant that answers questions about a resume/CV.

Rules:
1. Answer ONLY using the context provided below. Do not use outside knowledge.
2. If the context doesn't contain enough information, say "I don't have enough information about that in the resume."
3. Be specific — quote details like company names, dates, technologies when available.
4. Keep answers concise and professional.
5. If asked about suitability for a role, reason from the evidence in the context."""


def build_prompt(
    query: str,
    retrieved_chunks: list[dict],
    chat_history: list[dict],
) -> list[dict]:
    """
    Assembles the full messages list for the LLM API call.

    Structure:
      [system]  → instructions + context (retrieved chunks)
      [user]    → message 1
      [assistant] → reply 1
      ...        (history)
      [user]    → current question

    Why context goes in system prompt: it applies to ALL turns,
    so follow-up questions also have access to it.
    """

    # Format retrieved chunks into readable context block
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        source_tag = f"[{chunk.get('retrieval_source', 'unknown').upper()} RESULT {i+1}]"
        context_parts.append(f"{source_tag}\n{chunk['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    system_content = f"""{SYSTEM_PROMPT}

--- RESUME CONTEXT ---
{context_block}
--- END CONTEXT ---"""

    messages = [{"role": "system", "content": system_content}]

    # Add previous conversation turns (last 6 turns = 3 exchanges)
    for turn in chat_history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})

    # Add current question
    messages.append({"role": "user", "content": query})

    return messages
