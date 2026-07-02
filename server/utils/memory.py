"""
utils/memory.py — Rolling conversation history (last N turns).
"""


class Memory:
    def __init__(self, max_turns: int = 5) -> None:
        self.history:   list[dict] = []
        self.max_turns: int        = max_turns

    def add(self, user: str, assistant: str) -> None:
        self.history.append({"user": user, "assistant": assistant})
        if len(self.history) > self.max_turns:
            self.history.pop(0)

    def get(self) -> list[dict]:
        return self.history

    def clear(self) -> None:
        self.history = []

    def format(self) -> str:
        if not self.history:
            return ""
        lines = []
        for i, turn in enumerate(self.history, 1):
            lines.append(f"[Turn {i}] User asked: {turn['user']}")
            # Truncate long answers in history to avoid prompt bloat
            ans = turn['assistant']
            if len(ans) > 300:
                ans = ans[:300] + "... [truncated for brevity]"
            lines.append(f"[Turn {i}] Assistant answered: {ans}")
        return "\n".join(lines)
