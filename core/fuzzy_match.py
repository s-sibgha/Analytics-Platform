"""
core/fuzzy_match.py

Lightweight fuzzy matcher (stdlib-only, no external dependency required)
used to auto-suggest role mappings from raw column headers. Falls back to
difflib's SequenceMatcher; if rapidfuzz is installed it is used instead for
better quality, but it is never a hard dependency — the platform must
never crash because an optional library is missing.
"""
from __future__ import annotations
import re
from difflib import SequenceMatcher
from typing import Dict, List, Tuple

try:
    from rapidfuzz import fuzz as _rapidfuzz
    _HAS_RAPIDFUZZ = True
except ImportError:
    _HAS_RAPIDFUZZ = False


def _normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[_\-\.]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _similarity(a: str, b: str) -> float:
    """Returns a 0-100 similarity score."""
    a_n, b_n = _normalize(a), _normalize(b)
    if not a_n or not b_n:
        return 0.0
    if a_n == b_n:
        return 100.0
    if _HAS_RAPIDFUZZ:
        return float(_rapidfuzz.token_set_ratio(a_n, b_n))
    # difflib fallback, scaled to 0-100, boosted slightly for substring hits
    base = SequenceMatcher(None, a_n, b_n).ratio() * 100
    if a_n in b_n or b_n in a_n:
        base = max(base, 85.0)
    return base


def suggest_roles_for_header(
    header: str,
    role_synonyms: Dict[str, List[str]],
    min_score: float = 70.0,
    top_n: int = 3,
) -> List[Tuple[str, float]]:
    """
    Given a raw column header, return up to `top_n` (role, score) pairs
    ranked by descending fuzzy match score, filtered to >= min_score.
    """
    scored: List[Tuple[str, float]] = []
    for role, synonyms in role_synonyms.items():
        best_for_role = max((_similarity(header, syn) for syn in synonyms), default=0.0)
        # Also compare directly against the role identifier itself.
        best_for_role = max(best_for_role, _similarity(header, role.replace("_", " ")))
        if best_for_role >= min_score:
            scored.append((role, round(best_for_role, 1)))
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_n]


def suggest_role_mapping_for_headers(
    headers: List[str],
    role_synonyms: Dict[str, List[str]],
    min_score: float = 70.0,
) -> Dict[str, List[Tuple[str, float]]]:
    """Convenience batch wrapper: header -> ranked (role, score) suggestions."""
    return {h: suggest_roles_for_header(h, role_synonyms, min_score) for h in headers}