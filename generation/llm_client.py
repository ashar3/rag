"""
STEP 6b — LLM CLIENT
Analogy: The judge delivers a verdict based only on submitted evidence.
We send the assembled prompt and get back a natural language answer.
The LLM synthesizes, summarizes, and reasons — it doesn't just copy-paste text.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def generate_answer(messages: list[dict]) -> str:
    """
    Sends the full messages list to gpt-4o-mini and returns the answer string.
    temperature=0 → deterministic answers (better for factual Q&A).
    """
    client = _get_client()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0,
        max_tokens=1000,
    )
    return response.choices[0].message.content.strip()
