"""
tests/test_duckdb_pandas_parity.py — Differential Testing Suite for the
Execution Substrate Switch (engine/duckdb_executor.py vs the pandas
aggregation path in visualization/chart_factory.py::_aggregate).

This suite verifies that engine.duckdb_executor.duckdb_group_aggregate
produces results that are schema- and value-identical to the equivalent
pandas groupby().agg() computation, across:
    - direct engine-vs-engine parity (duckdb_group_aggregate vs a manual
      pandas reference implementation)
    - integration parity through visualization.chart_factory._aggregate,
      forcing each engine explicitly via monkeypatched dispatch flags
    - edge cases: empty DataFrame, single-row DataFrame, large (1M+ row)
      DataFrame threshold-triggering behavior
    - graceful fallback to pandas when the optional `duckdb` dependency is
      unavailable

Never assumes duckdb is installed in every test environment: DuckDB-
specific assertions are skipped (not failed) via pytest.importorskip when
the optional dependency is absent, consistent with engine/duckdb_executor
.py's own "never a hard dependency" design contract.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytest

from core.settings import DUCKDB_ROW_THRESHOLD
from engine import duckdb_executor
from visualization import chart_factory

# ── Skip DuckDB-specific tests gracefully when the optional dependency is
# not installed in this environment, matching the project's "never a hard
# dependency" policy for engine/duckdb_executor.py. ─────────────────────
duckdb = pytest.importorskip(
    "duckdb", reason="Optional 'duckdb' package is not installed in this environment."
)


# ══════════════════════════════════════════════════════════════════════════════
# FIXTURES — SAMPLE DATASETS
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def moderate_df() -> pd.DataFrame:
    """A small, deterministic dataset with a categorical group column and a
    numeric value column, exercising sum/mean/count/nunique aggregations."""
    rng = np.random.default_rng(seed=42)
    n = 500
    categories = ["Zone A", "Zone B", "Zone C", "Zone D", None]
    return pd.DataFrame({
        "zone": rng.choice(categories, size=n, p=[0.25, 0.25, 0.2, 0.2, 0.1]),
        "amount": rng.uniform(low=10.0, high=5000.0, size=n).round(2),
        "consumer_id": rng.integers(low=1000, high=1050, size=n),
    })


@pytest.fixture
def empty_df() -> pd.DataFrame:
    """An empty DataFrame with the expected schema but zero rows."""
    return pd.DataFrame({"zone": pd.Series(dtype="object"), "amount": pd.Series(dtype="float64")})


@pytest.fixture
def single_row_df() -> pd.DataFrame:
    """A single-row DataFrame — the minimal non-empty edge case."""
    return pd.DataFrame({"zone": ["Zone A"], "amount": [1234.56]})


@pytest.fixture(scope="module")
def large_df() -> pd.DataFrame:
    """
    A 1,000,001-row DataFrame — exactly one row above
    core.settings.DUCKDB_ROW_THRESHOLD — used to verify the Execution
    Substrate Switch actually triggers the DuckDB path rather than merely
    being defined-but-unreferenced. Module-scoped so it is only
    constructed once across the test module.
    """
    n = DUCKDB_ROW_THRESHOLD + 1
    rng = np.random.default_rng(seed=7)
    categories = np.array(["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"])
    return pd.DataFrame({
        "zone": rng.choice(categories, size=n),
        "amount": rng.uniform(low=1.0, high=10000.0, size=n),
    })


# ══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS — PANDAS REFERENCE IMPLEMENTATION & NORMALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def _pandas_reference_aggregate(
    df: pd.DataFrame,
    group_cols: List[str],
    value_col: Optional[str],
    aggregation: str,
) -> pd.DataFrame:
    """
    Independent, minimal pandas reference implementation mirroring the
    exact column-naming contract of engine.duckdb_executor
    .duckdb_group_aggregate: a "Count" column when value_col is None or
    aggregation == "count", otherwise a column named exactly `value_col`
    holding the aggregated numeric result. Deliberately reimplemented here
    (rather than calling chart_factory._aggregate directly) so this
    reference is fully independent of the production dispatch logic being
    tested, per differential-testing best practice.
    """
    if df.empty or not group_cols:
        return pd.DataFrame(columns=group_cols + (["Count"] if value_col is None else [value_col]))

    if value_col is None or aggregation == "count":
        result = df.groupby(group_cols, dropna=True).size().reset_index(name="Count")
        result["Count"] = result["Count"].astype("int64")
        return result

    if aggregation == "nunique":
        result = (
            df.groupby(group_cols, dropna=True)[value_col]
            .nunique()
            .reset_index(name=value_col)
        )
        result[value_col] = result[value_col].astype("int64")
        return result

    agg_fn_map: Dict[str, str] = {
        "sum": "sum", "mean": "mean", "median": "median", "min": "min", "max": "max",
    }
    agg_fn = agg_fn_map.get(aggregation, "count")
    numeric_value = pd.to_numeric(df[value_col], errors="coerce")
    tmp = df[group_cols].copy()
    tmp["__value"] = numeric_value
    result = (
        tmp.groupby(group_cols, dropna=True)["__value"]
        .agg(agg_fn)
        .reset_index()
        .rename(columns={"__value": value_col})
    )
    result[value_col] = pd.to_numeric(result[value_col], errors="coerce")
    return result


def _normalize_for_comparison(
    df: Optional[pd.DataFrame],
    group_cols: List[str],
    value_col: str,
) -> pd.DataFrame:
    """
    Normalizes a result DataFrame for cross-engine comparison: sorts by
    group columns (stable, deterministic ordering independent of each
    engine's internal execution order), resets the index, and casts the
    value column to float64 so integer/float dtype divergences between
    pandas and DuckDB's result marshaling never produce a false-positive
    parity failure. Never raises: returns an empty, correctly-shaped
    DataFrame on a None/empty input.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=group_cols + [value_col])
    normalized = df.copy(deep=True)
    for col in group_cols:
        normalized[col] = normalized[col].astype(str)
    normalized[value_col] = pd.to_numeric(normalized[value_col], errors="coerce").astype("float64")
    normalized = normalized.sort_values(by=group_cols).reset_index(drop=True)
    return normalized[group_cols + [value_col]]


# ══════════════════════════════════════════════════════════════════════════════
# DIRECT ENGINE-VS-ENGINE PARITY — duckdb_group_aggregate vs pandas reference
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("aggregation", ["count", "sum", "mean", "median", "min", "max", "nunique"])
def test_duckdb_vs_pandas_parity(moderate_df: pd.DataFrame, aggregation: str) -> None:
    """
    Core differential test: for every supported aggregation verb, the
    DuckDB out-of-core execution path (duckdb_group_aggregate) must
    produce a schema- and value-identical result to the independent
    pandas reference implementation, for the same grouping and value
    column, on the same input dataset.
    """
    group_cols = ["zone"]
    value_col = None if aggregation == "count" else "amount"

    pandas_result = _pandas_reference_aggregate(moderate_df, group_cols, value_col, aggregation)
    duckdb_result = duckdb_executor.duckdb_group_aggregate(
        moderate_df, group_cols, value_col, aggregation
    )

    assert duckdb_result is not None, (
        f"duckdb_group_aggregate returned None for aggregation='{aggregation}'; "
        "expected a valid DataFrame given DuckDB is confirmed available in this test session."
    )

    result_value_col = "Count" if (value_col is None or aggregation == "count") else value_col

    pandas_norm = _normalize_for_comparison(pandas_result, group_cols, result_value_col)
    duckdb_norm = _normalize_for_comparison(duckdb_result, group_cols, result_value_col)

    pd.testing.assert_frame_equal(
        pandas_norm, duckdb_norm, check_dtype=False, check_exact=False, rtol=1e-9, atol=1e-9,
    )


def test_duckdb_vs_pandas_parity_multi_group_columns(moderate_df: pd.DataFrame) -> None:
    """Parity must hold across a multi-column grouping key as well, mirroring
    chart_factory._aggregate's group_keys assembly for facet/group_by
    combinations."""
    group_cols = ["zone", "consumer_id"]
    value_col = "amount"
    aggregation = "sum"

    pandas_result = _pandas_reference_aggregate(moderate_df, group_cols, value_col, aggregation)
    duckdb_result = duckdb_executor.duckdb_group_aggregate(
        moderate_df, group_cols, value_col, aggregation
    )
    assert duckdb_result is not None

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy(deep=True)
        out["zone"] = out["zone"].astype(str)
        out["consumer_id"] = out["consumer_id"].astype(str)
        out[value_col] = pd.to_numeric(out[value_col], errors="coerce").astype("float64")
        return out.sort_values(by=group_cols).reset_index(drop=True)[group_cols + [value_col]]

    pd.testing.assert_frame_equal(
        _norm(pandas_result), _norm(duckdb_result), check_dtype=False, check_exact=False, rtol=1e-9,
    )


# ══════════════════════════════════════════════════════════════════════════════
# EDGE CASES — EMPTY / SINGLE-ROW DATAFRAMES
# ══════════════════════════════════════════════════════════════════════════════

def test_empty_dataframe_parity(empty_df: pd.DataFrame) -> None:
    """
    An empty DataFrame must degrade gracefully and identically on both
    engines: duckdb_group_aggregate returns None per its documented
    contract (df is None or df.empty short-circuit), and the pandas
    reference implementation returns an empty-but-correctly-shaped
    DataFrame. Neither path should raise.
    """
    duckdb_result = duckdb_executor.duckdb_group_aggregate(empty_df, ["zone"], "amount", "sum")
    assert duckdb_result is None, "duckdb_group_aggregate must return None for an empty input DataFrame."

    pandas_result = _pandas_reference_aggregate(empty_df, ["zone"], "amount", "sum")
    assert pandas_result.empty
    assert list(pandas_result.columns) == ["zone", "amount"]

    # Integration-level check: chart_factory._aggregate must also degrade
    # gracefully (empty result, never an unhandled exception) regardless
    # of which engine it would have dispatched to.
    agg_df, value_label = chart_factory._aggregate(
        empty_df, x_col="zone", y_col="amount", aggregation="sum", group_col=None, facet_col=None,
    )
    assert agg_df.empty
    assert value_label in ("Count", "amount")


def test_single_row_dataframe_parity(single_row_df: pd.DataFrame) -> None:
    """A single-row DataFrame is the minimal non-empty case: both engines
    must resolve to the same one-row aggregation result."""
    group_cols = ["zone"]
    value_col = "amount"
    aggregation = "sum"

    pandas_result = _pandas_reference_aggregate(single_row_df, group_cols, value_col, aggregation)
    duckdb_result = duckdb_executor.duckdb_group_aggregate(
        single_row_df, group_cols, value_col, aggregation
    )
    assert duckdb_result is not None
    assert len(duckdb_result) == 1

    pandas_norm = _normalize_for_comparison(pandas_result, group_cols, value_col)
    duckdb_norm = _normalize_for_comparison(duckdb_result, group_cols, value_col)
    pd.testing.assert_frame_equal(pandas_norm, duckdb_norm, check_dtype=False, check_exact=False)


# ══════════════════════════════════════════════════════════════════════════════
# LARGE DATAFRAME — DUCKDB_ROW_THRESHOLD TRIGGER VERIFICATION
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
def test_large_dataframe_triggers_duckdb_threshold(large_df: pd.DataFrame) -> None:
    """
    Verifies core.settings.DUCKDB_ROW_THRESHOLD actually gates the
    Execution Substrate Switch: a DataFrame with more rows than the
    threshold must cause should_use_duckdb() to return True (given
    DuckDB is available in this test session), and the resulting
    aggregation via duckdb_group_aggregate must match the pandas
    reference computation exactly.
    """
    assert len(large_df) > DUCKDB_ROW_THRESHOLD

    assert duckdb_executor.should_use_duckdb(large_df) is True, (
        "should_use_duckdb() must return True once row count exceeds "
        "DUCKDB_ROW_THRESHOLD and a working DuckDB connection is available."
    )

    group_cols = ["zone"]
    value_col = "amount"
    aggregation = "mean"

    duckdb_result = duckdb_executor.duckdb_group_aggregate(
        large_df, group_cols, value_col, aggregation
    )
    assert duckdb_result is not None

    pandas_result = _pandas_reference_aggregate(large_df, group_cols, value_col, aggregation)

    pandas_norm = _normalize_for_comparison(pandas_result, group_cols, value_col)
    duckdb_norm = _normalize_for_comparison(duckdb_result, group_cols, value_col)
    pd.testing.assert_frame_equal(
        pandas_norm, duckdb_norm, check_dtype=False, check_exact=False, rtol=1e-6, atol=1e-6,
    )


def test_below_threshold_does_not_trigger_duckdb(moderate_df: pd.DataFrame) -> None:
    """A DataFrame at or below DUCKDB_ROW_THRESHOLD must never route to the
    DuckDB path, preserving the platform's 'pandas for datasets under
    ~1M rows' execution substrate contract."""
    assert len(moderate_df) <= DUCKDB_ROW_THRESHOLD
    assert duckdb_executor.should_use_duckdb(moderate_df) is False


# ══════════════════════════════════════════════════════════════════════════════
# INTEGRATION PARITY — chart_factory._aggregate FORCED THROUGH EACH ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def test_chart_factory_aggregate_forced_duckdb_matches_forced_pandas(
    moderate_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Integration-level differential test: forces
    visualization.chart_factory._aggregate down the DuckDB path (by
    monkeypatching the `should_use_duckdb` name bound inside
    chart_factory's own module namespace to always return True) and
    separately down the pandas path (forcing it to always return False),
    then asserts the two dispatch outcomes are value-identical. This
    directly exercises the production dispatch branch inside _aggregate,
    not just the underlying engines in isolation.
    """
    group_col = None
    facet_col = None
    x_col = "zone"
    y_col = "amount"
    aggregation = "sum"

    # ── Force pandas path ──
    monkeypatch.setattr(chart_factory, "should_use_duckdb", lambda df: False)
    pandas_agg_df, pandas_value_label = chart_factory._aggregate(
        moderate_df, x_col, y_col, aggregation, group_col, facet_col
    )

    # ── Force DuckDB path ──
    monkeypatch.setattr(chart_factory, "should_use_duckdb", lambda df: True)
    duckdb_agg_df, duckdb_value_label = chart_factory._aggregate(
        moderate_df, x_col, y_col, aggregation, group_col, facet_col
    )

    assert pandas_value_label == duckdb_value_label == y_col

    pandas_norm = _normalize_for_comparison(pandas_agg_df, [x_col], y_col)
    duckdb_norm = _normalize_for_comparison(duckdb_agg_df, [x_col], y_col)
    pd.testing.assert_frame_equal(
        pandas_norm, duckdb_norm, check_dtype=False, check_exact=False, rtol=1e-9,
    )


def test_chart_factory_aggregate_duckdb_forced_but_engine_fails_falls_back(
    moderate_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Even when should_use_duckdb() reports True, if duckdb_group_aggregate
    itself returns None (any internal execution failure), _aggregate must
    unconditionally fall through to its original, unmodified pandas
    implementation rather than propagating an error or returning an empty
    result. This verifies the 'never crash on unknown input' contract at
    the dispatch boundary.
    """
    monkeypatch.setattr(chart_factory, "should_use_duckdb", lambda df: True)
    monkeypatch.setattr(chart_factory, "duckdb_group_aggregate", lambda *args, **kwargs: None)

    agg_df, value_label = chart_factory._aggregate(
        moderate_df, "zone", "amount", "sum", None, None
    )

    assert value_label == "amount"
    assert not agg_df.empty

    expected = _pandas_reference_aggregate(moderate_df, ["zone"], "amount", "sum")
    pd.testing.assert_frame_equal(
        _normalize_for_comparison(agg_df, ["zone"], "amount"),
        _normalize_for_comparison(expected, ["zone"], "amount"),
        check_dtype=False, check_exact=False, rtol=1e-9,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GRACEFUL FALLBACK — DUCKDB LIBRARY UNAVAILABLE
# ══════════════════════════════════════════════════════════════════════════════

def test_graceful_fallback_when_duckdb_unavailable(
    moderate_df: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Simulates the optional `duckdb` dependency being unavailable at
    runtime (e.g. not installed in a constrained deployment container) by
    monkeypatching engine.duckdb_executor's internal availability flag and
    shared connection singleton. Verifies:
        1. is_duckdb_available() correctly reports False.
        2. should_use_duckdb() correctly reports False regardless of row
           count (the safe, correctness-preserving default).
        3. duckdb_group_aggregate() returns None rather than raising.
        4. The full chart_factory._aggregate dispatch path still produces
           a correct result via its original, unmodified pandas
           implementation — parity with the pandas reference is preserved
           end-to-end even with DuckDB entirely unavailable.
    """
    monkeypatch.setattr(duckdb_executor, "_HAS_DUCKDB", False)
    monkeypatch.setattr(duckdb_executor, "_shared_connection", None)

    assert duckdb_executor.is_duckdb_available() is False
    assert duckdb_executor.should_use_duckdb(moderate_df, row_threshold=0) is False

    fallback_result = duckdb_executor.duckdb_group_aggregate(moderate_df, ["zone"], "amount", "sum")
    assert fallback_result is None

    # Integration check: chart_factory imported should_use_duckdb by value
    # at module load time, so it must also be patched on chart_factory's
    # own namespace to simulate the same unavailability at the dispatch
    # call site used by _aggregate.
    monkeypatch.setattr(chart_factory, "should_use_duckdb", duckdb_executor.should_use_duckdb)
    monkeypatch.setattr(chart_factory, "duckdb_group_aggregate", duckdb_executor.duckdb_group_aggregate)

    agg_df, value_label = chart_factory._aggregate(
        moderate_df, "zone", "amount", "sum", None, None
    )
    assert value_label == "amount"

    expected = _pandas_reference_aggregate(moderate_df, ["zone"], "amount", "sum")
    pd.testing.assert_frame_equal(
        _normalize_for_comparison(agg_df, ["zone"], "amount"),
        _normalize_for_comparison(expected, ["zone"], "amount"),
        check_dtype=False, check_exact=False, rtol=1e-9,
    )


def test_duckdb_group_aggregate_invalid_column_returns_none_not_raise(
    moderate_df: pd.DataFrame,
) -> None:
    """duckdb_group_aggregate must degrade to None (never raise) when a
    group or value column does not exist in the input DataFrame, matching
    the identifier-safety contract documented in its docstring."""
    result = duckdb_executor.duckdb_group_aggregate(
        moderate_df, ["nonexistent_column"], "amount", "sum"
    )
    assert result is None

    result_bad_value_col = duckdb_executor.duckdb_group_aggregate(
        moderate_df, ["zone"], "nonexistent_value_column", "sum"
    )
    assert result_bad_value_col is None

def test_null_group_key_parity(moderate_df: pd.DataFrame) -> None:
    """Regression test for the NULL-vs-dropna group-key divergence: pandas'
    groupby(..., dropna=True) excludes NaN-keyed rows entirely, while
    DuckDB's GROUP BY treats NULL as a distinct group by default. Verifies
    duckdb_group_aggregate's WHERE ... IS NOT NULL filter keeps both
    engines' row counts and group-key sets identical, even when the group
    column is guaranteed to contain nulls."""
    df_with_nulls = moderate_df.copy()
    df_with_nulls.loc[df_with_nulls.index[:10], "zone"] = None
    assert df_with_nulls["zone"].isna().any()

    pandas_result = _pandas_reference_aggregate(df_with_nulls, ["zone"], "amount", "sum")
    duckdb_result = duckdb_executor.duckdb_group_aggregate(df_with_nulls, ["zone"], "amount", "sum")

    assert duckdb_result is not None
    assert len(duckdb_result) == len(pandas_result)
    assert set(duckdb_result["zone"].astype(str)) == set(pandas_result["zone"].astype(str))
    assert not duckdb_result["zone"].isna().any()    