from __future__ import annotations

"""Sentiment scoring for Russian text using a lexicon-based approach.

This module provides a simple sentiment score function that counts positive and
negative words from a lexicon and returns a normalized score in the range [-1, 1].
"""

try:
    from .sentiment_lexicon import POSITIVE, NEGATIVE
except ImportError:
    # Fallback to empty sets if the lexicon hasn't been generated yet
    POSITIVE = set()
    NEGATIVE = set()


def score(text: str) -> float:
    """Return a sentiment score for the given text in the range [-1, 1].

    The score is calculated as (positive_count - negative_count) / total_count,
    where total_count is the number of words found in either the positive or
    negative lexicon. If no lexicon words are found, returns 0.0.

    Args:
        text: Input string to score.

    Returns:
        A float between -1 (all negative) and 1 (all positive).
    """
    if not text:
        return 0.0

    # Casefold for case-insensitive matching and split on whitespace
    words = text.casefold().split()
    positive_count = 0
    negative_count = 0

    for word in words:
        if word in POSITIVE:
            positive_count += 1
        elif word in NEGATIVE:
            negative_count += 1

    total = positive_count + negative_count
    if total == 0:
        return 0.0

    return (positive_count - negative_count) / total