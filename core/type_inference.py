"""
core/type_inference.py — ★ FOUNDATION ENGINE ★

Per-column cascade type detector. Order of detection:
    ID -> Datetime (incl. Excel serials) -> Boolean -> Numeric
    (incl. currency / percentage) -> Categorical -> Text

Every column receives a ColumnProfile with an explicit confidence score.
Columns below config.settings.MIN_AUTO_CONFIDENCE are flagged
`needs_manual_review = True` and must be routed to the manual mapping UI —
they are NEVER silently guessed and passed downstream.

This module must never raise an unhandled exception: any column that
causes a detection error degrades to InferredType.UNKNOWN with confidence
0.0 and a detection note explaining why, rather than crashing ingestion.
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

from core.roles import ROLE_SYNONYMS
from core.settings import MIN_AUTO_CONFIDENCE, MIN_FUZZY_ROLE_SCORE
from core.schema_models import ColumnProfile, InferredType
from core.fuzzy_match import suggest_roles_for_header

_CURRENCY_PATTERN = re.compile(r"^[^\d\-]*[-\d,]+\.?\d*\s*[^\d]*$")
_PERCENT_PATTERN = re.compile(r"^\s*-?\d+(\.\d+)?\s*%\s*$")
_ID_HEADER_HINTS = ("id", "no", "number", "code", "ref")
_BOOLEAN_TOKENS = {
    "true", "false", "yes", "no", "y", "n", "0", "1", "t", "f"
}

# Excel serial-date plausible range (roughly 1970-2100) used to avoid
# misclassifying small integers as dates.
_EXCEL_SERIAL_MIN = 25569      # 1970-01-01
_EXCEL_SERIAL_MAX = 73050      # ~2100-01-01


def _safe_series_sample(series: pd.Series, n: int = 8) -> List[Any]:
    try:
        non_null = series.dropna()
        sample = non_null.sample(min(n, len(non_null)), random_state=42) if len(non_null) else non_null
        return [_jsonable(v) for v in sample.tolist()]
    except Exception:
        return []


def _jsonable(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return str(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _looks_like_id_header(header: str) -> bool:
    h = header.lower()
    return any(hint in h for hint in _ID_HEADER_HINTS)


def _detect_id(series: pd.Series, header: str) -> Optional[Tuple[float, List[str]]]:
    """High-cardinality, mostly-unique column with an ID-like header."""
    notes: List[str] = []
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    uniqueness = non_null.nunique() / max(len(non_null), 1)
    header_hint = _looks_like_id_header(header)
    if uniqueness > 0.95 and header_hint:
        notes.append(f"Uniqueness ratio {uniqueness:.2f} with ID-like header '{header}'.")
        return 0.95, notes
    if uniqueness > 0.99:
        notes.append(f"Uniqueness ratio {uniqueness:.2f} (near-unique values).")
        return 0.7, notes
    return None


def _detect_datetime(series: pd.Series, header: str) -> Optional[Tuple[float, List[str]]]:
    notes: List[str] = []
    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    # Already a real datetime dtype.
    if pd.api.types.is_datetime64_any_dtype(series):
        notes.append("Column dtype is already datetime64.")
        return 0.99, notes

    # Excel serial-number dates: numeric column in plausible date-serial range.
    if pd.api.types.is_numeric_dtype(non_null):
        in_range = non_null.between(_EXCEL_SERIAL_MIN, _EXCEL_SERIAL_MAX)
        ratio = in_range.mean() if len(in_range) else 0.0
        if ratio > 0.9:
            notes.append(f"{ratio:.0%} of numeric values fall in plausible Excel date-serial range.")
            return 0.8, notes

    # String parse attempt.
    try:
        sample = non_null.astype(str).sample(min(50, len(non_null)), random_state=42)
        pure_digit_ratio = sample.str.fullmatch(r"\d+").fillna(False).mean()
        has_delimiter_ratio = sample.str.contains(r"[-/.\s]", regex=True).fillna(False).mean()
        if pure_digit_ratio > 0.5 and has_delimiter_ratio < 0.5:
            notes.append(
                f"{pure_digit_ratio:.0%} of sampled values are bare digit strings without date "
                "delimiters — treated as non-date to avoid misclassifying ID/serial codes."
            )
            return None
        
        try:
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        except (ValueError, TypeError):
    # format="mixed" needs pandas>=2.0; degrade to the dateutil fallback
    # rather than raising, matching this module's "never raise" contract.
            parsed = pd.to_datetime(sample, errors="coerce")
        success_ratio = parsed.notna().mean()
        if success_ratio > 0.85:
            notes.append(f"{success_ratio:.0%} of sampled values parse as valid dates.")
            return min(0.95, 0.6 + success_ratio * 0.4), notes
    except Exception as exc:  # noqa: BLE001
        notes.append(f"Datetime parse attempt raised: {exc}")
    return None


def _detect_boolean(series: pd.Series) -> Optional[Tuple[float, List[str]]]:
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    try:
        as_str = non_null.astype(str).str.strip().str.lower()
    except Exception:
        return None
    distinct = set(as_str.unique())
    if distinct and distinct.issubset(_BOOLEAN_TOKENS) and len(distinct) <= 2:
        return 0.9, [f"All {len(distinct)} distinct values are boolean-like tokens."]
    return None


def _detect_numeric(series: pd.Series, header: str) -> Optional[Tuple[float, List[str]]]:
    notes: List[str] = []
    non_null = series.dropna()
    if len(non_null) == 0:
        return None

    if pd.api.types.is_numeric_dtype(series):
        notes.append("Column dtype is already numeric.")
        return 0.97, notes

    as_str = non_null.astype(str).str.strip()
    pct_ratio = as_str.str.match(_PERCENT_PATTERN).mean()
    if pct_ratio > 0.85:
        notes.append(f"{pct_ratio:.0%} of values match a percentage pattern.")
        return 0.9, notes

    currency_like = as_str.str.contains(r"[₹$€£]") | as_str.str.match(r"^[\d,]+\.?\d*$")
    cleaned = as_str.str.replace(r"[₹$€£,\s]", "", regex=True)
    numeric_ratio = pd.to_numeric(cleaned, errors="coerce").notna().mean()
    if numeric_ratio > 0.85:
        if currency_like.mean() > 0.5:
            notes.append(f"{numeric_ratio:.0%} parse as numeric after stripping currency symbols.")
        else:
            notes.append(f"{numeric_ratio:.0%} of values parse as plain numeric.")
        return min(0.95, 0.55 + numeric_ratio * 0.4), notes
    return None


def _detect_categorical(series: pd.Series) -> Optional[Tuple[float, List[str]]]:
    non_null = series.dropna()
    if len(non_null) == 0:
        return None
    distinct_ratio = non_null.nunique() / max(len(non_null), 1)
    if non_null.nunique() <= 50 and distinct_ratio < 0.3:
        return 0.85, [f"Low cardinality ({non_null.nunique()} distinct, ratio {distinct_ratio:.2f})."]
    return None


def infer_column(series: pd.Series, header: str) -> ColumnProfile:
    """
    Run the full cascade detector on a single column and return a
    ColumnProfile. This function is exception-safe: any internal failure
    degrades to UNKNOWN rather than propagating.
    """
    try:
        null_count = int(series.isna().sum())
        total = len(series)
        null_pct = (null_count / total * 100) if total else 0.0
        distinct_count = int(series.nunique(dropna=True))
        sample_values = _safe_series_sample(series)

        cascade = [
            (InferredType.ID, _detect_id),
            (InferredType.DATETIME, lambda s, h=header: _detect_datetime(s, h)),
            (InferredType.BOOLEAN, lambda s, h=header: _detect_boolean(s)),
            (InferredType.NUMERIC, lambda s, h=header: _detect_numeric(s, h)),
            (InferredType.CATEGORICAL, lambda s, h=header: _detect_categorical(s)),
        ]

        inferred_type = InferredType.TEXT
        confidence = 0.4
        notes: List[str] = []

        for itype, detector in cascade:
            try:
                result = detector(series, header)
            except Exception as exc:  # noqa: BLE001
                notes.append(f"{itype.value} detector raised: {exc}")
                result = None
            if result:
                conf, det_notes = result
                inferred_type, confidence, notes = itype, conf, det_notes
                break
        else:
            notes.append("No specific pattern matched strongly; defaulted to free text.")

        # Currency / percentage refinement on top of numeric detection.
        if inferred_type == InferredType.NUMERIC:
            try:
                sample_str = series.dropna().astype(str).head(20)
                if sample_str.str.contains(r"[₹$€£]").mean() > 0.5:
                    inferred_type = InferredType.CURRENCY
                elif sample_str.str.match(_PERCENT_PATTERN).mean() > 0.5:
                    inferred_type = InferredType.PERCENTAGE
            except Exception:
                pass

        needs_review = confidence < MIN_AUTO_CONFIDENCE

        role_suggestions = suggest_roles_for_header(
            header, ROLE_SYNONYMS, min_score=MIN_FUZZY_ROLE_SCORE
        )
        suggested_roles = [r for r, _ in role_suggestions]
        suggested_scores = {r: s for r, s in role_suggestions}

        return ColumnProfile(
            original_name=header,
            inferred_type=inferred_type,
            confidence=round(float(confidence), 3),
            sample_values=sample_values,
            null_count=null_count,
            null_pct=round(null_pct, 2),
            distinct_count=distinct_count,
            needs_manual_review=needs_review,
            detection_notes=notes,
            suggested_roles=suggested_roles,
            suggested_role_scores=suggested_scores,
        )
    except Exception as exc:  # noqa: BLE001 — absolute final safety net
        return ColumnProfile(
            original_name=header,
            inferred_type=InferredType.UNKNOWN,
            confidence=0.0,
            needs_manual_review=True,
            detection_notes=[f"Type inference failed unexpectedly: {exc}"],
        )


def infer_dataframe(df: pd.DataFrame) -> List[ColumnProfile]:
    """Run infer_column across every column in the dataframe."""
    profiles: List[ColumnProfile] = []
    for col in df.columns:
        profiles.append(infer_column(df[col], str(col)))
    return profiles