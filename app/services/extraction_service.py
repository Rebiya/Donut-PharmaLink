"""Candidate drug token extraction service."""

import re
from typing import List

from app.utils.text_utils import STOPWORDS, clean_token, unique_preserve_order


class ExtractionService:
    """Extract likely medication tokens from OCR text."""

    DOSAGE_PATTERN = re.compile(
        r"\b([A-Za-z][A-Za-z\-]{2,})\s*\d+(?:\.\d+)?\s*(?:mg|ml|mcg|g|tab|caps?|iu)\b",
        flags=re.IGNORECASE,
    )
    WORD_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z\-]{2,}\b")

    def extract_candidates(self, text: str) -> List[str]:
        """Return unique candidate drug tokens using dosage and capitalization cues."""
        if not text.strip():
            return []

        candidates: List[str] = []

        for match in self.DOSAGE_PATTERN.findall(text):
            candidates.append(match)

        for token in self.WORD_PATTERN.findall(text):
            if token[0].isupper() or token.lower() not in STOPWORDS:
                candidates.append(token)

        cleaned = [
            cleaned_token
            for token in candidates
            if (cleaned_token := clean_token(token)) and cleaned_token not in STOPWORDS
        ]
        return unique_preserve_order(cleaned)
