from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover — optional dependency, never a hard requirement
    pa = None  # type: ignore
    _HAS_PYARROW = False

from core.roles import ROLE_RECORD_ID, ROLE_STATUS
from core.settings import NULL_PLACEHOLDERS, MIN_AUTO_CONFIDENCE
from utils.audit_log import AuditLog
from core.column_registry import ColumnRegistry
from core.schema_models import AuditEntry, ColumnProfile, InferredType

_CURRENCY_RE = re.compile(r"[₹\$€£¥,\s]")
_PERCENT_RE = re.compile(r"\s*%\s*$")
_EXCEL_SERIAL_ORIGIN = pd.Timestamp("1899-12-30")
_EXCEL_SERIAL_MIN = 25569
_EXCEL_SERIAL_MAX = 73050
_IQR_MULTIPLIER = 3.0


@dataclass
class FlaggedDuplicateGroup:
    record_id_value: Any
    occurrences: int
    rows: pd.DataFrame
    business_note: str


@dataclass
class OutlierSummary:
    role: str
    column_name: str
    outlier_count: int
    lower_fence: float
    upper_fence: float
    outlier_indices: List[int]


@dataclass
class CleaningResult:
    original_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    analytics_df: pd.DataFrame
    audit_entries: List[AuditEntry]
    flagged_business_key_duplicates: List[FlaggedDuplicateGroup]
    flagged_outlier_summaries: List[OutlierSummary]
    cleaning_summary: Dict[str, int]

    @property
    def rows_original(self) -> int:
        return len(self.original_df)

    @property
    def rows_cleaned(self) -> int:
        return len(self.cleaned_df)

    @property
    def rows_removed(self) -> int:
        return self.rows_original - self.rows_cleaned


class SafeCleaningEngine:

    _NULL_REPLACE_MAP: Dict[str, float] = {v: np.nan for v in NULL_PLACEHOLDERS if v != ""}

    def __init__(self, registry: ColumnRegistry, profiles: List[ColumnProfile]) -> None:
        self._registry = registry
        self._profiles: Dict[str, ColumnProfile] = {p.original_name: p for p in profiles}
        self._audit = AuditLog()
        self._summary: Dict[str, int] = {
            "rows_removed_exact_duplicates": 0,
            "rows_removed_fully_empty": 0,
            "values_null_normalized": 0,
            "values_whitespace_trimmed": 0,
            "columns_dtype_converted": 0,
            "date_columns_standardized": 0,
            "business_key_duplicate_groups": 0,
            "outlier_columns_flagged": 0,
            # ── Parquet/Arrow ingestion-safety counters (Refactor Phase 1) ──
            "timezone_columns_normalized": 0,
            "decimal_columns_coerced": 0,
            "null_type_columns_coerced": 0,
        }

    def clean(self, df: pd.DataFrame) -> CleaningResult:
        original = df.copy(deep=True)
        work = df.copy(deep=True)

        # ── Arrow/Parquet ingestion-safety passes — MUST run first, before
        # any pandas string/numeric/datetime operation below is permitted to
        # touch a column, since tz-aware timestamps, Arrow decimal128
        # columns, and Arrow null-typed / fully-empty columns each corrupt
        # or crash downstream pandas logic if left unnormalized.
        work = self._normalize_timezone(work)
        work = self._coerce_null_type_columns(work)
        work = self._coerce_decimal_columns(work)

        # Clean values natively but preserve original operational indexes during profiling checks
        work = self._trim_and_collapse_whitespace(work)
        work = self._normalize_null_placeholders(work)
        work = self._convert_high_confidence_numerics(work)
        work = self._standardize_high_confidence_dates(work)

        # Identify tracking anomalies while the base data structure retains index integrity
        flagged_dups = self._flag_business_key_duplicates(work)
        flagged_outliers = self._flag_outliers(work)

        # Structural reductions are applied strictly at the boundary end of the pipeline
        work = self._remove_fully_empty_rows(work)
        work = self._remove_exact_duplicate_rows(work)

        analytics = self._build_analytics_df(work)

        return CleaningResult(
            original_df=original,
            cleaned_df=work,
            analytics_df=analytics,
            audit_entries=self._audit.entries(),
            flagged_business_key_duplicates=flagged_dups,
            flagged_outlier_summaries=flagged_outliers,
            cleaning_summary=dict(self._summary),
        )

    # ── Arrow/Parquet ingestion-safety passes (Refactor Phase 1: A3, A4, B2) ──

    def _normalize_timezone(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Strips timezone metadata from every tz-aware datetime64 column,
        unconditionally, before any other cleaning step runs. This closes
        Audit Finding A3: `_standardize_high_confidence_dates`'s
        `is_datetime64_any_dtype` guard passes for tz-aware columns exactly
        as it does for tz-naive ones, silently skipping standardization and
        leaving tz metadata to propagate into every downstream duration/
        comparison operation in analytics.py and chart_factory.py. Running
        this pass first means every column arrives tz-naive before any
        other function in this class ever inspects it. Never raises.
        """
        normalized: List[str] = []
        for col in df.columns:
            try:
                if not pd.api.types.is_datetime64_any_dtype(df[col]):
                    continue
                tz = getattr(df[col].dt, "tz", None)
                if tz is not None:
                    df[col] = df[col].dt.tz_localize(None)
                    normalized.append(col)
            except Exception:
                pass
        if normalized:
            self._summary["timezone_columns_normalized"] += len(normalized)
            self._audit.log(
                "cleaning",
                f"Normalized timezone metadata (tz-aware → tz-naive) on {len(normalized)} "
                f"datetime column(s): {', '.join(normalized)}.",
                details={"step": "normalize_timezone", "columns": normalized},
            )
        return df

    def _coerce_decimal_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects Arrow-backed decimal128(p,s) columns (a plausible dtype for
        a currency/Amount field sourced from a Parquet file with an
        explicit DECIMAL schema) and casts them to float64 before any
        downstream `pd.to_numeric` call. Closes Audit Finding A4:
        `_convert_high_confidence_numerics` and `_flag_outliers` both call
        `pd.to_numeric` without a decimal-type guard, which has
        inconsistent, version-dependent behavior against `ArrowDtype`
        decimal columns (silent coercion on some pandas/pyarrow version
        pairs, `TypeError`/`ArrowNotImplementedError` on others). Never
        raises; a cast failure on one column is skipped, not fatal to the
        pipeline.
        """
        if not _HAS_PYARROW:
            return df
        coerced: List[str] = []
        for col in df.columns:
            try:
                dtype = df[col].dtype
                pyarrow_dtype = getattr(dtype, "pyarrow_dtype", None)
                if pyarrow_dtype is None or not pa.types.is_decimal(pyarrow_dtype):
                    continue
                try:
                    df[col] = df[col].astype("float64")
                except Exception:
                    df[col] = pd.to_numeric(df[col].astype(str), errors="coerce")
                coerced.append(col)
            except Exception:
                pass
        if coerced:
            self._summary["decimal_columns_coerced"] += len(coerced)
            self._audit.log(
                "cleaning",
                f"Coerced {len(coerced)} Arrow decimal128 column(s) to float64 prior to "
                f"numeric processing: {', '.join(coerced)}.",
                details={"step": "coerce_decimal_columns", "columns": coerced},
            )
        return df

    def _coerce_null_type_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects columns with an Arrow `null` type (a valid but ambiguous
        Arrow dtype produced when serializing a fully-empty object column)
        or a fully-empty legacy `object` column, and force-casts both to an
        explicit, empty `pandas.StringDtype()` column. Closes Audit
        Finding B2: an all-null column has an ambiguous Arrow type at
        Parquet write/read time and is rejected by many Parquet
        readers/writers or by schema-unification across a multi-file scan
        (`ArrowNotImplementedError: Unsupported cast from null to X`).
        Normalizing to a concrete, empty string column here guarantees a
        stable, homogeneous schema before any Parquet write path
        introduced upstream of this class. Never raises.
        """
        coerced: List[str] = []
        for col in df.columns:
            try:
                dtype = df[col].dtype
                pyarrow_dtype = getattr(dtype, "pyarrow_dtype", None) if _HAS_PYARROW else None
                is_arrow_null = bool(
                    _HAS_PYARROW and pyarrow_dtype is not None and pa.types.is_null(pyarrow_dtype)
                )
                is_fully_empty_object = (
                    dtype == object and len(df[col]) > 0 and bool(df[col].isna().all())
                )
                if not (is_arrow_null or is_fully_empty_object):
                    continue
                df[col] = pd.Series([pd.NA] * len(df), index=df.index, dtype="string")
                coerced.append(col)
            except Exception:
                pass
        if coerced:
            self._summary["null_type_columns_coerced"] += len(coerced)
            self._audit.log(
                "cleaning",
                f"Coerced {len(coerced)} Arrow null-typed / fully-empty column(s) to an "
                f"explicit empty string dtype: {', '.join(coerced)}.",
                details={"step": "coerce_null_type_columns", "columns": coerced},
            )
        return df

    def _remove_fully_empty_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = df.isnull().all(axis=1)
        count = int(mask.sum())
        if count:
            df = df.loc[~mask].reset_index(drop=True)
            self._summary["rows_removed_fully_empty"] += count
            self._audit.log(
                "cleaning",
                f"Removed {count} fully-empty row(s) (all values NaN/blank).",
                rows_affected=count,
                details={"step": "remove_fully_empty_rows"},
            )
        return df

    def _remove_exact_duplicate_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        n_before = len(df)
        df = df.drop_duplicates(keep="first").reset_index(drop=True)
        removed = n_before - len(df)
        if removed:
            self._summary["rows_removed_exact_duplicates"] += removed
            self._audit.log(
                "cleaning",
                f"Removed {removed} exact duplicate row(s) (all columns identical).",
                rows_affected=removed,
                details={"step": "remove_exact_duplicates"},
            )
        return df

    # ── String-safe cleaning passes (Refactor Phase 1: A1, A2) ──

    def _trim_and_collapse_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Trims and collapses internal whitespace across every string-typed
        column. Closes Audit Finding A1: iterating via
        `df.select_dtypes(include="object")` matches only legacy numpy
        `object` dtype and silently skips `pandas.StringDtype` /
        `ArrowDtype(pa.string())` columns (the dtypes a Parquet/pyarrow
        ingestion path can legitimately produce), turning this function
        into dead code with no error raised. `pd.api.types.is_string_dtype`
        matches `object`, `StringDtype`, and Arrow-backed string dtypes
        uniformly. Never raises.
        """
        total_affected = 0
        string_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        for col in string_cols:
            try:
                original_values = df[col].copy()
                df[col] = df[col].apply(
                    lambda v: re.sub(r"\s+", " ", str(v)).strip()
                    if isinstance(v, str) else v
                )
                changed = int((df[col] != original_values).sum())
                total_affected += changed
            except Exception:
                pass
        if total_affected:
            self._summary["values_whitespace_trimmed"] += total_affected
            self._audit.log(
                "cleaning",
                f"Trimmed and collapsed internal whitespace across {total_affected} cell(s).",
                rows_affected=total_affected,
                details={"step": "trim_whitespace"},
            )
        return df

    def _normalize_null_placeholders(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalizes placeholder-null tokens ("NA", "-", "NULL", etc.) to true
        NaN across every string-typed column. Closes Audit Findings A1 and
        A2: column selection now uses `is_string_dtype` (see
        `_trim_and_collapse_whitespace` above) instead of
        `select_dtypes(include="object")`, and the strip step no longer
        calls `.astype(str)` directly — on a nullable `StringDtype` /
        `ArrowDtype(string)` column, `.astype(str)` converts `pd.NA` into
        the literal four-character string `"<NA>"`, permanently escaping
        null detection downstream. The `.where(~mask, ...)` pattern only
        ever transforms already-non-null cells, leaving true nulls
        untouched. Never raises.
        """
        total_changed = 0
        string_cols = [c for c in df.columns if pd.api.types.is_string_dtype(df[c])]
        for col in string_cols:
            try:
                original_nulls = int(df[col].isnull().sum())

                df[col] = df[col].replace(self._NULL_REPLACE_MAP)

                mask = df[col].notna()
                stripped = df[col].astype("string").str.strip()
                df[col] = df[col].where(~mask, stripped)

                df[col] = df[col].replace("", pd.NA)
                df[col] = df[col].where(df[col].notna(), np.nan)

                new_nulls = int(df[col].isnull().sum())
                total_changed += (new_nulls - original_nulls)
            except Exception:
                pass
        if total_changed:
            self._summary["values_null_normalized"] += total_changed
            self._audit.log(
                "cleaning",
                f"Normalized {total_changed} placeholder-null token(s) (NA, -, NULL, etc.) to NaN.",
                rows_affected=total_changed,
                details={"step": "normalize_null_placeholders", "tokens": NULL_PLACEHOLDERS},
            )
        return df

    def _convert_high_confidence_numerics(self, df: pd.DataFrame) -> pd.DataFrame:
        converted: List[str] = []
        for col, profile in self._profiles.items():
            if col not in df.columns:
                continue
            if profile.inferred_type not in (
                InferredType.NUMERIC, InferredType.CURRENCY, InferredType.PERCENTAGE
            ):
                continue
            if profile.confidence < MIN_AUTO_CONFIDENCE:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue
            try:
                original_non_null_mask = df[col].notna()
                if int(original_non_null_mask.sum()) == 0:
                    continue
                cleaned_series = df[col].astype(str).apply(self._strip_numeric_formatting)
                numeric_series = pd.to_numeric(cleaned_series, errors="coerce")
                # Success ratio scored only against cells that carried a value prior to
                # conversion — pre-existing nulls must never count as parse failures,
                # matching type_inference.py's non-null-based confidence methodology.
                success_ratio = numeric_series[original_non_null_mask].notna().mean()
                if success_ratio >= 0.85:
                    df[col] = numeric_series
                    converted.append(col)
                    self._audit.log(
                        "cleaning",
                        f"Converted column '{col}' to numeric "
                        f"(type={profile.inferred_type.value}, "
                        f"confidence={profile.confidence:.2f}, "
                        f"parse_success={success_ratio:.0%}).",
                        details={"step": "convert_numeric", "column": col},
                    )
            except Exception as exc:
                self._audit.log(
                    "cleaning",
                    f"Skipped numeric conversion for '{col}': {exc}",
                    details={"step": "convert_numeric", "column": col, "error": str(exc)},
                )
        self._summary["columns_dtype_converted"] += len(converted)
        return df

    @staticmethod
    def _strip_numeric_formatting(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        v = _CURRENCY_RE.sub("", value)
        v = _PERCENT_RE.sub("", v)
        return v.strip() or np.nan

    def _standardize_high_confidence_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        standardized: List[str] = []
        for col, profile in self._profiles.items():
            if col not in df.columns:
                continue
            if profile.inferred_type != InferredType.DATETIME:
                continue
            if profile.confidence < MIN_AUTO_CONFIDENCE:
                continue
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                continue
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    non_null = df[col].dropna()
                    if len(non_null) and non_null.between(_EXCEL_SERIAL_MIN, _EXCEL_SERIAL_MAX).mean() > 0.9:
                        df[col] = pd.to_datetime(
                            df[col], unit="D", origin=_EXCEL_SERIAL_ORIGIN, errors="coerce"
                        )
                        standardized.append(col)
                        self._audit.log(
                            "cleaning",
                            f"Converted column '{col}' from Excel date serial to datetime.",
                            details={"step": "standardize_dates", "column": col, "method": "excel_serial"},
                        )
                        continue
                original_non_null_mask = df[col].notna()
                if int(original_non_null_mask.sum()) == 0:
                   continue
                parsed = pd.to_datetime(df[col], errors="coerce", format="mixed")
                parse_rate = parsed[original_non_null_mask].notna().mean()
                if parse_rate >= 0.80:
                    df[col] = parsed
                    standardized.append(col)
                    self._audit.log(
                        "cleaning",
                        f"Standardized datetime column '{col}' "
                        f"(parse_success={parse_rate:.0%}, confidence={profile.confidence:.2f}).",
                        details={"step": "standardize_dates", "column": col, "method": "infer"},
                    )
            except Exception as exc:
                self._audit.log(
                    "cleaning",
                    f"Skipped date standardization for '{col}': {exc}",
                    details={"step": "standardize_dates", "column": col, "error": str(exc)},
                )
        self._summary["date_columns_standardized"] += len(standardized)
        return df

    def _flag_business_key_duplicates(self, df: pd.DataFrame) -> List[FlaggedDuplicateGroup]:
        groups: List[FlaggedDuplicateGroup] = []
        id_col = self._registry.resolve(ROLE_RECORD_ID)
        if not id_col or id_col not in df.columns:
            return groups
        try:
            dup_mask = df.duplicated(subset=[id_col], keep=False)
            if not dup_mask.any():
                return groups
            dup_df = df[dup_mask].copy()
            for key_val, group in dup_df.groupby(id_col, sort=False):
                count = len(group)
                all_identical = group.duplicated(keep=False).all()
                note = (
                    "Exact row duplicates sharing the same Record ID — "
                    "already removed in safe cleaning step."
                    if all_identical
                    else f"Business-key duplicate: Record ID '{key_val}' appears {count} times "
                    f"with differing values — retained in dataset as potential repeat activity "
                    f"(e.g. repeated complaint, multi-stage record, or duplicate submission)."
                )
                if not all_identical:
                    groups.append(
                        FlaggedDuplicateGroup(
                            record_id_value=key_val,
                            occurrences=count,
                            rows=group.reset_index(drop=True),
                            business_note=note,
                        )
                    )
            self._summary["business_key_duplicate_groups"] += len(groups)
            if groups:
                self._audit.log(
                    "cleaning",
                    f"Flagged {len(groups)} business-key duplicate group(s) on column '{id_col}' "
                    f"(NOT removed — see Flagged Duplicates panel for context).",
                    rows_affected=sum(g.occurrences for g in groups),
                    details={"step": "flag_business_key_duplicates", "id_column": id_col},
                )
        except Exception as exc:
            self._audit.log(
                "cleaning",
                f"Business-key duplicate check skipped: {exc}",
                details={"step": "flag_business_key_duplicates", "error": str(exc)},
            )
        return groups

    def _flag_outliers(self, df: pd.DataFrame) -> List[OutlierSummary]:
        summaries: List[OutlierSummary] = []
        numeric_cols: Dict[str, str] = {}
        for col, profile in self._profiles.items():
            if col not in df.columns:
                continue
            if profile.inferred_type in (InferredType.NUMERIC, InferredType.CURRENCY):
                if profile.confidence >= MIN_AUTO_CONFIDENCE:
                    for role, mapping in self._registry.mappings.items():
                        if mapping.column_name == col and mapping.confirmed:
                            numeric_cols[col] = role
                            break

        for col, role in numeric_cols.items():
            try:
                series = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(series) < 10:
                    continue
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                if iqr == 0:
                    continue
                lower = q1 - _IQR_MULTIPLIER * iqr
                upper = q3 + _IQR_MULTIPLIER * iqr
                outlier_idx = series[(series < lower) | (series > upper)].index.tolist()
                if outlier_idx:
                    summaries.append(OutlierSummary(
                        role=role,
                        column_name=col,
                        outlier_count=len(outlier_idx),
                        lower_fence=float(round(lower, 4)),
                        upper_fence=float(round(upper, 4)),
                        outlier_indices=outlier_idx,
                    ))
                    self._audit.log(
                        "cleaning",
                        f"Flagged {len(outlier_idx)} outlier(s) in column '{col}' "
                        f"(IQR×{_IQR_MULTIPLIER} fence: [{lower:.2f}, {upper:.2f}]) "
                        f"— NOT removed per non-destructive compliance policy.",
                        rows_affected=len(outlier_idx),
                        details={
                            "step": "flag_outliers",
                            "column": col,
                            "role": role,
                            "lower_fence": lower,
                            "upper_fence": upper,
                        },
                    )
            except Exception as exc:
                self._audit.log(
                    "cleaning",
                    f"Outlier detection skipped for '{col}': {exc}",
                    details={"step": "flag_outliers", "column": col, "error": str(exc)},
                )
        self._summary["outlier_columns_flagged"] += len(summaries)
        return summaries

    def _build_analytics_df(self, cleaned_df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies status value canonicalization to build the Analytics-Ready
        dataset. Closes Audit Finding A5: the prior `.apply(lambda v:
        table.get(...))` implementation forces a silent dictionary/category
        → object dtype widening whenever `status_col` arrives Arrow
        dictionary-encoded (the expected outcome for a low-cardinality
        Status field read via DuckDB/PyArrow), directly undermining the
        memory-footprint benefit of the Parquet pivot. This implementation
        resolves canonicalization via a fully vectorized `.map(table)` /
        `.fillna(...)` chain and explicitly casts the result back to
        pandas `category` dtype, restoring dictionary-encoding-equivalent
        memory characteristics. True nulls are preserved exactly (never
        canonicalized to a string). Never raises.
        """
        analytics = cleaned_df.copy(deep=True)
        status_col = self._registry.resolve(ROLE_STATUS)
        if status_col and status_col in analytics.columns:
            try:
                table = self._registry.value_canonicalization.get("status", {})
                original_notna = analytics[status_col].notna()

                normalized = analytics[status_col].astype(str).str.strip().str.lower()
                canonical = normalized.map(table)
                fallback = analytics[status_col].astype(str).str.strip()
                resolved = canonical.fillna(fallback)

                resolved = resolved.where(original_notna, np.nan)
                analytics[status_col] = resolved.astype("category")

                self._audit.log(
                    "cleaning",
                    f"Applied status value canonicalization to column '{status_col}' in analytics "
                    f"dataset (category-dtype preserved to avoid dictionary-encoding memory widening).",
                    details={"step": "build_analytics_df", "column": status_col},
                )
            except Exception as exc:
                self._audit.log(
                    "cleaning",
                    f"Status canonicalization skipped: {exc}",
                    details={"step": "build_analytics_df", "error": str(exc)},
                )
        return analytics