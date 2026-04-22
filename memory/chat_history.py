"""
STEP 7 — CHAT HISTORY
Analogy: A notepad the assistant reads before answering each question.
Without it, every question is the first question — no "he", "there", "that role"
can be resolved because there's no prior context.

We keep it in memory (a Python list) for simplicity.
In production you'd store this in Redis or a database per session.
"""


class ChatHistory:
    def __init__(self, max_turns: int = 10):
        # max_turns = how many exchanges to keep
        # older turns are dropped to keep prompt size manageable
        self.max_turns = max_turns
        self._history: list[dict] = []

    def add(self, role: str, content: str):
        """Add a turn. role = 'user' or 'assistant'."""
        self._history.append({"role": role, "content": content})
        # trim oldest turns if we exceed limit (keep last max_turns * 2 messages)
        cap = self.max_turns * 2
        if len(self._history) > cap:
            self._history = self._history[-cap:]

    def get(self) -> list[dict]:
        """Returns all stored turns as list of {role, content}."""
        return list(self._history)

    def clear(self):
        """Reset — called when user uploads a new PDF."""
        self._history = []

    def __len__(self):
        return len(self._history)
