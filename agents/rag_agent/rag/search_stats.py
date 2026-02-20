"""Tracks retrieval stats across tool calls within a single query."""


class SearchStats:
    def __init__(self):
        self.searches: list[tuple[str, int]] = []

    def record(self, query: str, num_chunks: int):
        self.searches.append((query, num_chunks))

    def format(self, elapsed: float = 0.0) -> str:
        parts = []
        for query, count in self.searches:
            parts.append(f"_{query}_ ({count} chunks)")
        if not parts and not elapsed:
            return ""
        timer = f"{elapsed:.1f}s"
        if not parts:
            return f"\n\n> :clock1: {timer}"
        return "\n\n> :mag: " + " | ".join(parts) + f"  :clock1: {timer}"
