"""Text preprocessing for student support questions."""

import re


def clean_text(text: str) -> str:
    """Basic cleaning: lowercase, trim, collapse whitespace.

    Punctuation is left in place so TF-IDF can still use word boundaries;
    only repeated internal spaces are normalized.
    """
    if text is None:
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s
