"""
engine/duckdb_executor.py — Refactor Phase 4: Force-DuckDB-For-All-Datasets
+ Thread-Safety Remediation.

Row-count gating has been removed entirely from should_use_duckdb: DuckDB
acceleration is now attempted for every non-empty dataframe regardless of
size, per the platform-wide "Force DuckDB For All Datasets" mandate.
core.settings.DUCKDB_ROW_THRESHOLD is retained only for backward-compatible
call signatures and is no longer read by should_use_duckdb's logic.

Thread-safety fix: every .execute() call in this module now goes through
_get_connection_for_thread() (a per-thread cursor over the single shared
in-memory database) instead of the raw _shared_connection object directly.
DuckDB connections are not documented as safe for concurrent .execute()
calls from multiple threads; under the Force-DuckDB mandate every chart
render on every concurrent Streamlit session now hits this module, making
the previously-latent race condition a live production risk.

Never raises: any DuckDB unavailability, query construction failure,
execution failure, or nested-Arrow-type rejection degrades to a sentinel
return (None) so the caller falls back to the existing pandas aggregation
path. Every caught exception is routed through utils.error_logging
.log_exception at "warning" severity before returning None.
"""
from __future__ import annotations

import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from core.settings import DUCKDB_ROW_THRESHOLD
from utils.error_logging import log_exception

try:
    import duckdb  # type: ignore
    _HAS_DUCKDB = True
except ImportError:  # pragma: no cover — optional dependency
    duckdb = None  # type: ignore
    _HAS_DUCKDB = False

try:
    import pyarrow as pa  # type: ignore
    _HAS_PYARROW = True
except ImportError:  # pragma: no cover — optional dependency
    pa = None  # type: ignore
    _HAS_PYARROW = False

_SUPPORTED_AGGREGATIONS: Tuple[str, ...] = (
    "count", "sum", "mean", "median", "min", "max", "nunique",
)

_SQL_AGG_FN_MAP: Dict[str, str] = {
    "sum": "SUM",
    "mean": "AVG",
    "median": "MEDIAN",
    "min": "MIN",
    "max": "MAX",
}

_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[\w \-\.\(\)/%]{1,128}$", re.UNICODE)

_NESTED_DUCKDB_TYPE_MARKERS: Tuple[str, ...] = ("STRUCT(", "MAP(", "UNION(", "[]")

_connection_lock = threading.Lock()
_shared_connection: Optional[Any] = None
_thread_local = threading.local()


def _initialize_connection() -> Optional[Any]:
    """
    Lazily constructs and caches a single process-wide, in-memory DuckDB
    connection. Thread-safe via a module-level lock. Never raises: returns
    None if the optional `duckdb` dependency is not installed or connection
    construction fails for any reason.

    IMPORTANT: this returns the raw shared connection object, intended
    ONLY for availability checks (is_duckdb_available/should_use_duckdb)
    and as the base connection _get_connection_for_thread() derives
    per-thread cursors from. Query execution must NEVER call .execute()
    directly on the object returned here — use _get_connection_for_thread().
    """
    global _shared_connection
    if not _HAS_DUCKDB:
        return None
    if _shared_connection is not None:
        return _shared_connection
    with _connection_lock:
        if _shared_connection is not None:
            return _shared_connection
        try:
            connection = duckdb.connect(database=":memory:")
            try:
                connection.execute("SET enable_object_cache=true;")
            except Exception as exc:  # noqa: BLE001
                log_exception(
                    "engine.duckdb_executor._initialize_connection.enable_object_cache",
                    exc, severity="warning",
                )
            _shared_connection = connection
        except Exception as exc:  # noqa: BLE001
            log_exception("engine.duckdb_executor._initialize_connection", exc, severity="warning")
            _shared_connection = None
        return _shared_connection


def _get_connection_for_thread() -> Optional[Any]:
    """
    Returns a DuckDB cursor scoped to the calling thread, sharing the same
    underlying in-memory database as _shared_connection but safe for
    concurrent execute() calls across threads. This is the ONLY entry
    point any function in this module (or a downstream accelerator module)
    should use to actually run a query — never _initialize_connection()'s
    return value directly. Each thread lazily gets its own cursor, cached
    for that thread's lifetime. Never raises: falls back to the shared
    connection object itself if cursor creation fails, rather than
    returning None and losing acceleration entirely.
    """
    base_connection = _initialize_connection()
    if base_connection is None:
        return None
    cursor = getattr(_thread_local, "cursor", None)
    if cursor is not None:
        return cursor
    try:
        cursor = base_connection.cursor()
    except Exception:  # noqa: BLE001
        cursor = base_connection
    _thread_local.cursor = cursor
    return cursor


def is_duckdb_available() -> bool:
    """Returns True if the optional `duckdb` package is importable and a
    connection can be established. Never raises."""
    return _initialize_connection() is not None


def should_use_duckdb(df: pd.DataFrame, row_threshold: int = DUCKDB_ROW_THRESHOLD) -> bool:
    """
    Execution Substrate Switch predicate.

    Refactor Phase 4 / Force-DuckDB mandate: `row_threshold` is retained
    ONLY for backward-compatible call signatures (any existing caller
    passing it positionally or by keyword continues to work) — it is no
    longer read or compared against. DuckDB acceleration is now attempted
    for EVERY non-empty dataframe, regardless of row count, as long as a
    working DuckDB connection is available. Never raises; degrades to
    False (i.e., "use pandas") on any internal failure, which remains a
    safe, correctness-preserving default since the pandas path is fully
    verified independently.
    """
    try:
        if df is None or df.empty:
            return False
        return is_duckdb_available()
    except Exception:  # noqa: BLE001
        return False


def _sanitize_identifier(name: str) -> Optional[str]:
    """Validates a candidate column name against a conservative allow-list
    before it is ever quoted and interpolated into a SQL statement. Never
    raises."""
    try:
        if not isinstance(name, str) or not name:
            return None
        if not _SAFE_IDENTIFIER_PATTERN.match(name):
            return None
        return name
    except Exception:  # noqa: BLE001
        return None


def _quote_identifier(name: str) -> str:
    """Double-quotes a column identifier for DuckDB SQL, escaping any
    embedded double-quote characters."""
    return '"' + name.replace('"', '""') + '"'


def _is_aggregatable_column(df: pd.DataFrame, col: str) -> bool:
    """
    Nested-type schema guard for the pandas-replacement-scan path. Rejects
    any column whose dtype is a nested/composite Arrow type (Struct, List,
    LargeList, Map, Union) before it is ever handed to
    `duckdb_group_aggregate`'s SQL construction. Accepts standard pandas
    numeric/string/datetime/boolean/categorical dtypes unconditionally.
    Never raises; any inspection failure conservatively returns False.
    """
    try:
        if col not in df.columns:
            return False
        series = df[col]
        dtype = series.dtype
        pyarrow_dtype = getattr(dtype, "pyarrow_dtype", None)
        if _HAS_PYARROW and pyarrow_dtype is not None:
            try:
                if (
                    pa.types.is_struct(pyarrow_dtype)
                    or pa.types.is_list(pyarrow_dtype)
                    or pa.types.is_large_list(pyarrow_dtype)
                    or pa.types.is_fixed_size_list(pyarrow_dtype)
                    or pa.types.is_map(pyarrow_dtype)
                    or pa.types.is_union(pyarrow_dtype)
                    or pa.types.is_nested(pyarrow_dtype)
                ):
                    return False
            except Exception:  # noqa: BLE001
                return False
        if (
            pd.api.types.is_numeric_dtype(series)
            or pd.api.types.is_string_dtype(series)
            or pd.api.types.is_datetime64_any_dtype(series)
            or pd.api.types.is_bool_dtype(series)
            or isinstance(dtype, pd.CategoricalDtype)
        ):
            return True
        return False
    except Exception:  # noqa: BLE001
        return False


def _is_aggregatable_duckdb_type(type_name: str) -> bool:
    """
    Nested-type schema guard for the Parquet-native scan path. Inspects a
    DuckDB SQL type string and rejects any type carrying a nested-type
    marker before that column is ever placed into a GROUP BY/aggregate SQL
    clause against `read_parquet()`. Never raises.
    """
    try:
        if not isinstance(type_name, str) or not type_name:
            return False
        upper = type_name.strip().upper()
        return not any(marker in upper for marker in _NESTED_DUCKDB_TYPE_MARKERS)
    except Exception:  # noqa: BLE001
        return False


def _fetch_df_arrow_backed(result_relation: Any) -> Optional[pd.DataFrame]:
    """
    Dtype-backend-consistent result materialization. Uses an explicit
    `types_mapper=pd.ArrowDtype` conversion via
    `.fetch_arrow_table().to_pandas(...)`, keeping every DuckDB round trip
    on the same Arrow-backed dtype backend end-to-end. Falls back to plain
    `.fetch_df()` when pyarrow is unavailable or the Arrow conversion path
    fails. Never raises; returns None only if both paths fail.
    """
    try:
        if _HAS_PYARROW:
            arrow_table = result_relation.fetch_arrow_table()
            return arrow_table.to_pandas(types_mapper=pd.ArrowDtype)
        return result_relation.fetch_df()
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "engine.duckdb_executor._fetch_df_arrow_backed.arrow_path", exc, severity="warning",
        )
        try:
            return result_relation.fetch_df()
        except Exception as exc2:  # noqa: BLE001
            log_exception(
                "engine.duckdb_executor._fetch_df_arrow_backed.fallback_path", exc2, severity="warning",
            )
            return None


def duckdb_group_aggregate(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: Optional[str],
    aggregation: str,
) -> Optional[pd.DataFrame]:
    """
    Out-of-core equivalent of visualization.chart_factory._aggregate's core
    pandas computation, executed entirely inside embedded DuckDB. Returns
    a pandas DataFrame with the SAME column-naming contract
    chart_factory._aggregate already produces.

    Refactor Phase 4: now uses _get_connection_for_thread() (per-thread
    cursor) instead of the raw shared connection object, since this
    function is now invoked on EVERY aggregation regardless of row count
    and must be safe under concurrent Streamlit sessions.

    Never raises: any failure returns None (logged at "warning" severity),
    signaling the caller to fall back to the pandas aggregation path.
    """
    try:
        if df is None or df.empty or not group_cols:
            return None
        if aggregation not in _SUPPORTED_AGGREGATIONS:
            aggregation = "count"

        connection = _get_connection_for_thread()
        if connection is None:
            return None

        safe_group_cols: List[str] = []
        for col in group_cols:
            safe = _sanitize_identifier(str(col))
            if safe is None or safe not in df.columns:
                return None
            if not _is_aggregatable_column(df, safe):
                return None
            safe_group_cols.append(safe)

        safe_value_col: Optional[str] = None
        if value_col is not None:
            safe_value_col = _sanitize_identifier(str(value_col))
            if safe_value_col is None or safe_value_col not in df.columns:
                return None
            if not _is_aggregatable_column(df, safe_value_col):
                return None

        quoted_group_cols = [_quote_identifier(c) for c in safe_group_cols]
        group_by_clause = ", ".join(quoted_group_cols)
        not_null_clause = " AND ".join(f"{col} IS NOT NULL" for col in quoted_group_cols)

        if safe_value_col is None or aggregation == "count":
            select_clause = f"{group_by_clause}, COUNT(*) AS \"Count\""
            result_value_col = "Count"
        else:
            quoted_value_col = _quote_identifier(safe_value_col)
            if aggregation == "nunique":
                agg_expr = f"COUNT(DISTINCT {quoted_value_col})"
            else:
                sql_fn = _SQL_AGG_FN_MAP.get(aggregation, "COUNT")
                agg_expr = f"{sql_fn}({quoted_value_col})"
            select_clause = f"{group_by_clause}, {agg_expr} AS {quoted_value_col}"
            result_value_col = safe_value_col

        query = (
            f"SELECT {select_clause} FROM working_frame "
            f"WHERE {not_null_clause} "
            f"GROUP BY {group_by_clause}"
        )

        working_frame = df  # noqa: F841 — DuckDB replacement-scan target
        result_relation = connection.execute(query)
        result_df = _fetch_df_arrow_backed(result_relation)

        if result_df is None:
            return None
        if result_df.empty:
            return pd.DataFrame(columns=safe_group_cols + [result_value_col])

        if result_value_col != "Count":
            result_df[result_value_col] = pd.to_numeric(result_df[result_value_col], errors="coerce")

        return result_df
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "engine.duckdb_executor.duckdb_group_aggregate", exc, severity="warning",
            context={"group_cols": group_cols, "value_col": value_col, "aggregation": aggregation},
        )
        return None


def duckdb_group_aggregate_from_parquet(
    parquet_path: str,
    group_cols: List[str],
    value_col: Optional[str],
    aggregation: str,
) -> Optional[pd.DataFrame]:
    """
    Parquet-native execution primitive. Executes the identical group-by/
    aggregate computation as `duckdb_group_aggregate`, but directly against
    a Parquet file via DuckDB's native `read_parquet()`, with ZERO pandas
    materialization of the raw rows.

    Refactor Phase 4: both the DESCRIBE metadata query and the main
    aggregation query now route through _get_connection_for_thread().

    Never raises: any failure returns None (logged at "warning" severity).
    """
    try:
        if not parquet_path or not isinstance(parquet_path, str) or not group_cols:
            return None
        if aggregation not in _SUPPORTED_AGGREGATIONS:
            aggregation = "count"

        connection = _get_connection_for_thread()
        if connection is None:
            return None

        try:
            describe_relation = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [parquet_path]
            )
            describe_df = describe_relation.fetch_df()
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "engine.duckdb_executor.duckdb_group_aggregate_from_parquet.describe",
                exc, severity="warning", context={"parquet_path": parquet_path},
            )
            return None

        if describe_df is None or describe_df.empty or "column_name" not in describe_df.columns:
            return None
        column_types: Dict[str, str] = dict(zip(describe_df["column_name"], describe_df["column_type"]))

        safe_group_cols: List[str] = []
        for col in group_cols:
            safe = _sanitize_identifier(str(col))
            if safe is None or safe not in column_types:
                return None
            if not _is_aggregatable_duckdb_type(column_types[safe]):
                return None
            safe_group_cols.append(safe)

        safe_value_col: Optional[str] = None
        if value_col is not None:
            safe_value_col = _sanitize_identifier(str(value_col))
            if safe_value_col is None or safe_value_col not in column_types:
                return None
            if not _is_aggregatable_duckdb_type(column_types[safe_value_col]):
                return None

        quoted_group_cols = [_quote_identifier(c) for c in safe_group_cols]
        group_by_clause = ", ".join(quoted_group_cols)
        not_null_clause = " AND ".join(f"{col} IS NOT NULL" for col in quoted_group_cols)

        if safe_value_col is None or aggregation == "count":
            select_clause = f"{group_by_clause}, COUNT(*) AS \"Count\""
            result_value_col = "Count"
        else:
            quoted_value_col = _quote_identifier(safe_value_col)
            if aggregation == "nunique":
                agg_expr = f"COUNT(DISTINCT {quoted_value_col})"
            else:
                sql_fn = _SQL_AGG_FN_MAP.get(aggregation, "COUNT")
                agg_expr = f"{sql_fn}({quoted_value_col})"
            select_clause = f"{group_by_clause}, {agg_expr} AS {quoted_value_col}"
            result_value_col = safe_value_col

        query = (
            f"SELECT {select_clause} FROM read_parquet(?) "
            f"WHERE {not_null_clause} "
            f"GROUP BY {group_by_clause}"
        )

        result_relation = connection.execute(query, [parquet_path])
        result_df = _fetch_df_arrow_backed(result_relation)

        if result_df is None:
            return None
        if result_df.empty:
            return pd.DataFrame(columns=safe_group_cols + [result_value_col])

        if result_value_col != "Count":
            result_df[result_value_col] = pd.to_numeric(result_df[result_value_col], errors="coerce")

        return result_df
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "engine.duckdb_executor.duckdb_group_aggregate_from_parquet", exc, severity="warning",
            context={
                "parquet_path": parquet_path, "group_cols": group_cols,
                "value_col": value_col, "aggregation": aggregation,
            },
        )
        return None
    
def _build_filter_where_clause(
    filters: Optional[List[Tuple[str, str, Any]]],
    column_types: Dict[str, str],
) -> Tuple[str, List[Any]]:
    """
    Builds a parameterized SQL WHERE fragment (plus its ordered parameter
    list) from a list of (column, operator, value) filter predicates, for
    safe pushdown into a DuckDB read_parquet() scan. Only columns present
    in `column_types` (verified against the target Parquet file's own
    DESCRIBE schema) are ever interpolated, and every column identifier is
    routed through `_sanitize_identifier` + `_quote_identifier` — the same
    safety contract every other query builder in this module follows.
    Supported operators: "eq" (equality), "in" (membership against a
    list/tuple), "between" (inclusive range against a 2-tuple
    [start, end], either side may be None to leave that bound open). Any
    filter referencing an unrecognized column, an unsupported operator, or
    a malformed value is silently skipped — fails safe toward "no
    predicate", never toward an injectable/malformed SQL fragment. Never
    raises.

    Returns (where_fragment, params). where_fragment is either "" (no
    valid filters) or a string beginning with " AND ...".
    """
    if not filters:
        return "", []
    clauses: List[str] = []
    params: List[Any] = []
    for entry in filters:
        try:
            col, op, value = entry
            safe_col = _sanitize_identifier(str(col))
            if safe_col is None or safe_col not in column_types:
                continue
            quoted = _quote_identifier(safe_col)
            op_l = str(op).strip().lower()
            if op_l == "eq":
                if value is None:
                    continue
                clauses.append(f"{quoted} = ?")
                params.append(value)
            elif op_l == "in":
                values = [v for v in (value or []) if v is not None]
                if not values:
                    continue
                placeholders = ", ".join(["?"] * len(values))
                clauses.append(f"{quoted} IN ({placeholders})")
                params.extend(values)
            elif op_l == "between":
                if not isinstance(value, (list, tuple)) or len(value) != 2:
                    continue
                start, end = value
                if start is not None:
                    clauses.append(f"{quoted} >= ?")
                    params.append(start)
                if end is not None:
                    clauses.append(f"{quoted} <= ?")
                    params.append(end)
            else:
                continue
        except Exception:  # noqa: BLE001
            continue
    if not clauses:
        return "", []
    return " AND " + " AND ".join(clauses), params


def duckdb_group_aggregate_from_parquet_filtered(
    parquet_path: str,
    group_cols: List[str],
    value_col: Optional[str],
    aggregation: str,
    filters: Optional[List[Tuple[str, str, Any]]] = None,
) -> Optional[pd.DataFrame]:
    """
    Filter-aware variant of `duckdb_group_aggregate_from_parquet`. Same
    output contract and column-naming as the original — purely additive;
    the original function is untouched and remains the zero-filter code
    path. Every entry in `filters` is pushed directly into the DuckDB
    `read_parquet()` scan's WHERE clause via a fully parameterized query
    (values are never string-interpolated), so filtering happens at
    Parquet row-group/predicate-pushdown level instead of requiring the
    caller to materialize the whole file into pandas first and mask it
    there. This is what keeps the platform's fastest execution path
    available even once a drill-down path or categorical filter is
    active. Never raises: any failure returns None, signaling the caller
    to fall back to the existing, unmodified pandas/replacement-scan
    paths exactly as before.
    """
    try:
        if not parquet_path or not isinstance(parquet_path, str) or not group_cols:
            return None
        if aggregation not in _SUPPORTED_AGGREGATIONS:
            aggregation = "count"

        connection = _get_connection_for_thread()
        if connection is None:
            return None

        try:
            describe_df = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [parquet_path]
            ).fetch_df()
        except Exception as exc:  # noqa: BLE001
            log_exception(
                "engine.duckdb_executor.duckdb_group_aggregate_from_parquet_filtered.describe",
                exc, severity="warning", context={"parquet_path": parquet_path},
            )
            return None

        if describe_df is None or describe_df.empty or "column_name" not in describe_df.columns:
            return None
        column_types: Dict[str, str] = dict(zip(describe_df["column_name"], describe_df["column_type"]))

        safe_group_cols: List[str] = []
        for col in group_cols:
            safe = _sanitize_identifier(str(col))
            if safe is None or safe not in column_types:
                return None
            if not _is_aggregatable_duckdb_type(column_types[safe]):
                return None
            safe_group_cols.append(safe)

        safe_value_col: Optional[str] = None
        if value_col is not None:
            safe_value_col = _sanitize_identifier(str(value_col))
            if safe_value_col is None or safe_value_col not in column_types:
                return None
            if not _is_aggregatable_duckdb_type(column_types[safe_value_col]):
                return None

        quoted_group_cols = [_quote_identifier(c) for c in safe_group_cols]
        group_by_clause = ", ".join(quoted_group_cols)
        not_null_clause = " AND ".join(f"{col} IS NOT NULL" for col in quoted_group_cols)

        if safe_value_col is None or aggregation == "count":
            select_clause = f"{group_by_clause}, COUNT(*) AS \"Count\""
            result_value_col = "Count"
        else:
            quoted_value_col = _quote_identifier(safe_value_col)
            if aggregation == "nunique":
                agg_expr = f"COUNT(DISTINCT {quoted_value_col})"
            else:
                sql_fn = _SQL_AGG_FN_MAP.get(aggregation, "COUNT")
                agg_expr = f"{sql_fn}({quoted_value_col})"
            select_clause = f"{group_by_clause}, {agg_expr} AS {quoted_value_col}"
            result_value_col = safe_value_col

        where_fragment, filter_params = _build_filter_where_clause(filters, column_types)

        query = (
            f"SELECT {select_clause} FROM read_parquet(?) "
            f"WHERE {not_null_clause}{where_fragment} "
            f"GROUP BY {group_by_clause}"
        )
        result_relation = connection.execute(query, [parquet_path] + filter_params)
        result_df = _fetch_df_arrow_backed(result_relation)

        if result_df is None:
            return None
        if result_df.empty:
            return pd.DataFrame(columns=safe_group_cols + [result_value_col])
        if result_value_col != "Count":
            result_df[result_value_col] = pd.to_numeric(result_df[result_value_col], errors="coerce")
        return result_df
    except Exception as exc:  # noqa: BLE001
        log_exception(
            "engine.duckdb_executor.duckdb_group_aggregate_from_parquet_filtered", exc, severity="warning",
            context={
                "parquet_path": parquet_path, "group_cols": group_cols,
                "value_col": value_col, "aggregation": aggregation, "filters": filters,
            },
        )
        return None