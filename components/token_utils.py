from __future__ import annotations

"""
Token Budget Utilities — tokenizer 기반 길이 측정 + 자르기

Usage:
    tb = TokenBudget(processor.tokenizer)  # or TokenBudget(None) for char fallback
    n = tb.count("some text")
    trimmed = tb.truncate("long text", max_tokens=500)
"""


class TokenBudget:
    """Token-based length measurement and truncation.

    Args:
        tokenizer: HuggingFace tokenizer (e.g., processor.tokenizer).
                   None → character-based fallback (1 token ≈ 4 chars).
    """

    CHARS_PER_TOKEN = 4  # fallback ratio

    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def count(self, text: str) -> int:
        """Count tokens in text."""
        if not text:
            return 0
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return len(text) // self.CHARS_PER_TOKEN

    def truncate(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within max_tokens."""
        if not text or max_tokens <= 0:
            return ""
        if self.count(text) <= max_tokens:
            return text

        if self.tokenizer:
            ids = self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            return self.tokenizer.decode(ids, skip_special_tokens=True)

        # Char fallback
        return text[:max_tokens * self.CHARS_PER_TOKEN]

    def truncate_keep_tail(self, text: str, max_tokens: int, prefix: str = "") -> str:
        """Truncate from the head, keeping the tail (most recent)."""
        if not text or max_tokens <= 0:
            return ""
        if self.count(text) <= max_tokens:
            return text

        prefix_tokens = self.count(prefix) if prefix else 0
        available = max_tokens - prefix_tokens

        if self.tokenizer:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            tail_ids = ids[-available:]
            return prefix + self.tokenizer.decode(tail_ids, skip_special_tokens=True)

        # Char fallback
        char_budget = available * self.CHARS_PER_TOKEN
        return prefix + text[-char_budget:]
