"""
engine/analytics_duckdb_accelerator.py — Refactor Phase 4: Force-DuckDB
thread-safety fix + group-key TRIM() normalization parity fix.

Mirrors engine/duckdb_executor.py's exact singleton/fallback contract.
Never imported by anything that assumes it exists — analytics.py calls
into this module defensively and falls back to its own unmodified pandas
groupby implementation on any None return.
"""
from __future__ import annotations

from typing import List, Optional

import pandas as pd

from engine.duckdb_executor import (
    _get_connection_for_thread,
    should_use_duckdb,
    _sanitize_identifier,
    _quote_identifier,
    _is_aggregatable_column,
    _is_aggregatable_duckdb_type,
    _fetch_df_arrow_backed,
)

from engine.duckdb_executor import duckdb_group_aggregate_from_parquet_filtered
from utils.error_logging import log_exception


def duckdb_officer_or_category_productivity(
    df: pd.DataFrame,
    group_col: str,
    closed_mask: pd.Series,
    pending_mask: pd.Series,
    parquet_path: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Out-of-core equivalent of the per-group total/closed/pending counting
    loop inside ComplaintKPIEngine.compute_officer_productivity /
    compute_category_breakdown. Returns [group_col, "total", "closed",
    "pending"], or None on any failure. Never raises.

    Refactor Phase 4 / Audit Finding — Thread Safety: both branches now
    call `_get_connection_for_thread()` instead of the raw shared
    connection, since this accelerator is now invoked on every group-by
    KPI computation regardless of dataset size.

    Refactor Phase 4 / Audit Finding — Group-Key Normalization Parity: the
    Parquet-native branch now wraps the group column in SQL TRIM() before
    both the SELECT and GROUP BY clauses, mirroring the
    `.astype(str).str.strip()` normalization ComplaintKPIEngine already
    applies before its pandas groupby. Without this, a group value with
    leading/trailing whitespace would produce a DuckDB-computed key that
    never matches the caller's already-trimmed lookup key, silently
    defeating acceleration for exactly the noisy data it exists to help
    with (the per-officer loop still falls back correctly to the pandas
    count in that case — no wrong numbers ever reached the UI — but this
    fix restores full acceleration coverage and guarantees byte-identical
    grouping keys between the SQL and pandas paths).
    """
    try:
        connection = _get_connection_for_thread()
        if connection is None:
            return None

        if parquet_path:
            safe_col = _sanitize_identifier(str(group_col))
            if safe_col is None:
                return None
            try:
                describe_df = connection.execute(
                    "DESCRIBE SELECT * FROM read_parquet(?)", [parquet_path]
                ).fetch_df()
            except Exception as exc:  # noqa: BLE001
                log_exception(
                    "engine.analytics_duckdb_accelerator.duckdb_officer_or_category_productivity.describe",
                    exc, severity="warning",
                    context={"parquet_path": parquet_path, "group_col": group_col},
                )
                return None

            if describe_df is None or describe_df.empty or "column_name" not in describe_df.columns:
                return None
            column_types = dict(zip(describe_df["column_name"], describe_df["column_type"]))
            if safe_col not in column_types or not _is_aggregatable_duckdb_type(column_types[safe_col]):
                return None

            quoted = _quote_identifier(safe_col)
            trimmed_expr = f"TRIM(CAST({quoted} AS VARCHAR))"
            query = (
                f"SELECT {trimmed_expr} AS group_key, "
                f"COUNT(*) AS total, "
                f"0 AS closed, "
                f"0 AS pending "
                f"FROM read_parquet(?) WHERE {quoted} IS NOT NULL "
                f"GROUP BY {trimmed_expr}"
            )
            result = _fetch_df_arrow_backed(connection.execute(query, [parquet_path]))
            if result is None or result.empty:
                return None
            result = result.rename(columns={"group_key": group_col})
            return result

        if not should_use_duckdb(df):
            return None
        safe_col = _sanitize_identifier(str(group_col))
        if safe_col is None or safe_col not in df.columns:
            return None
        if not _is_aggregatable_column(df, safe_col):
            return None

        working_frame = df.assign(__closed=closed_mask.values, __pending=pending_mask.values)  # noqa: F841
        quoted = _quote_identifier(safe_col)
        trimmed_expr = f"TRIM(CAST({quoted} AS VARCHAR))"
        query = (
            f"SELECT {trimmed_expr} AS group_key, "
            f"COUNT(*) AS total, "
            f"SUM(CASE WHEN __closed THEN 1 ELSE 0 END) AS closed, "
            f"SUM(CASE WHEN __pending THEN 1 ELSE 0 END) AS pending "
            f"FROM working_frame WHERE {quoted} IS NOT NULL "
            f"GROUP BY {trimmed_expr}"
        )
        result = _fetch_df_arrow_backed(connection.execute(query))
        if result is None or result.empty:
            return None
        result = result.rename(columns={"group_key": group_col})
        return result
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "engine.analytics_duckdb_accelerator.duckdb_officer_or_category_productivity",
            exc, severity="warning",
            context={"group_col": group_col, "parquet_path": parquet_path},
        )
        return None
    


def duckdb_officer_or_category_productivity_filtered(
    df: pd.DataFrame,
    group_col: str,
    closed_mask: pd.Series,
    pending_mask: pd.Series,
    parquet_path: Optional[str] = None,
    filters: Optional[List[Tuple[str, str, Any]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Filter-aware sibling of `duckdb_officer_or_category_productivity`.
    Purely additive — the original function is completely untouched and
    remains the code path for the unfiltered/zero-filter case. When
    `parquet_path` AND `filters` are both supplied, `total` is computed
    directly against the Parquet file with the active filter/drill
    predicates pushed into the WHERE clause via
    `duckdb_group_aggregate_from_parquet_filtered`. As with the original
    function, `closed`/`pending` on this Parquet-native path remain 0
    (they are tied to the already-materialized pandas masks, not
    portable to SQL); callers must reconcile those the same way the
    original function's callers already do. Never raises — any failure
    or unavailability falls through to None, signaling the caller to use
    the existing in-memory pandas/replacement-scan path.
    """
    try:
        connection = _get_connection_for_thread()
        if connection is None or not parquet_path:
            return None
        safe_col = _sanitize_identifier(str(group_col))
        if safe_col is None:
            return None

        result = duckdb_group_aggregate_from_parquet_filtered(
            parquet_path, [safe_col], None, "count", filters=filters,
        )
        if result is None or result.empty or safe_col not in result.columns:
            return None
        result = result.rename(columns={safe_col: group_col, "Count": "total"})
        result["closed"] = 0
        result["pending"] = 0
        return result[[group_col, "total", "closed", "pending"]]
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "engine.analytics_duckdb_accelerator.duckdb_officer_or_category_productivity_filtered",
            exc, severity="warning",
            context={"group_col": group_col, "parquet_path": parquet_path, "filters": filters},
        )
        return None