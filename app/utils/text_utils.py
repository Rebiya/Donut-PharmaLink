"""Text utility helpers."""

import re
from typing import Iterable, List, Set


STOPWORDS: Set[str] = {
    "take",
    "tablet",
    "tablets",
    "capsule",
    "capsules",
    "tab",
    "cap",
    "mg",
    "ml",
    "po",
    "bid",
    "tid",
    "qhs",
    "doctor",
    "patient",
    "rx",
    "with",
    "after",
    "before",
    "daily",
    "night",
    "morning",
}


def clean_token(token: str) -> str:
    """Normalize token by stripping punctuation and lowercasing."""
    return re.sub(r"[^A-Za-z0-9\-]", "", token).strip().lower()


def unique_preserve_order(tokens: Iterable[str]) -> List[str]:
    """Return unique values while preserving insertion order."""
    seen: Set[str] = set()
    output: List[str] = []
    for token in tokens:
        key = token.lower()
        if key not in seen:
            seen.add(key)
            output.append(token)
    return output
