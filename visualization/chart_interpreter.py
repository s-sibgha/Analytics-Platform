"""
visualization/chart_interpreter.py — PHASE 3B: PART 2 — Automated Analytical Interpreter

Stateless, rule-based translation layer that converts the (figure, aggregated_df,
metadata) triple already produced by visualization/chart_factory.py render
functions into a structured, natural-language interpretation payload consumable
by the Smart Narrative Engine, KPI cards, and executive dashboard panels.

This module performs NO rendering and holds NO mutable state — it is pure
calculation over already-aggregated data, matching the project's Universal
Analytics Engine philosophy. It never raises to the UI: every failure path
(ineligible chart, empty dataframe, missing columns, non-numeric payloads,
unrecognized chart_type) degrades to a structured, safe fallback payload
conforming to the mandated schema rather than propagating an exception.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Module-level constants ──────────────────────────────────────────────────

_SEVERITY_CRITICAL: str = "critical"
_SEVERITY_WARNING: str = "warning"
_SEVERITY_INFO: str = "info"
_SEVERITY_POSITIVE: str = "positive"

_INSIGHT_CONCENTRATION: str = "volumetric_concentration"
_INSIGHT_TREND: str = "directional_trend"
_INSIGHT_ANOMALY: str = "statistical_anomaly"
_INSIGHT_DISTRIBUTION: str = "distribution_profile"
_INSIGHT_ELIGIBILITY: str = "eligibility_notice"
_INSIGHT_EXTREME: str = "extreme_value"
_INSIGHT_FLOW: str = "flow_dominance"
_INSIGHT_RISK: str = "risk_concentration"
_INSIGHT_COVERAGE: str = "data_coverage"

_ANOMALY_Z_THRESHOLD: float = 2.5
_ANOMALY_IQR_MULTIPLIER: float = 1.5
_HIGH_CONCENTRATION_THRESHOLD_PCT: float = 40.0
_MODERATE_CONCENTRATION_THRESHOLD_PCT: float = 25.0
_MIN_POINTS_FOR_ANOMALY_CHECK: int = 4

_NA_TEXT: str = "N/A — insufficient data"

# Language-domain lexicons (extensible: add new domains without touching
# any interpretation logic below).
_DOMAIN_LEXICON: Dict[str, Dict[str, str]] = {
    "power_utility": {
        "record_singular": "complaint",
        "record_plural": "complaints",
        "unit_label": "cases",
    },
    "revenue_analytics": {
        "record_singular": "transaction",
        "record_plural": "transactions",
        "unit_label": "records",
    },
    "hr_analytics": {
        "record_singular": "employee record",
        "record_plural": "employee records",
        "unit_label": "records",
    },
    "generic": {
        "record_singular": "record",
        "record_plural": "records",
        "unit_label": "records",
    },
}
_DEFAULT_DOMAIN: str = "power_utility"

# Extensible dictionary-mapped switch framework: maps the project's
# CHART_REGISTRY chart_type strings (visualization/chart_factory.py) onto an
# internal interpretation category. Onboarding a new chart type requires
# only a single new entry here — no interpretation logic needs to change.
_CHART_TYPE_CATEGORY_MAP: Dict[str, str] = {
    "bar": "categorical_bar",
    "bar_horizontal": "categorical_bar",
    "bar_grouped": "categorical_bar",
    "bar_stacked": "categorical_bar",
    "pie": "categorical_bar",
    "donut": "categorical_bar",
    "line": "time_series_line",
    "area": "time_series_line",
    "sparkline": "time_series_line",
    "calendar_heatmap": "time_series_line",
    "rolling_average_trend": "time_series_line",
    "peak_complaint_hour": "peak_hour",
    "histogram": "distribution_profile",
    "box": "distribution_profile",
    "pending_age_distribution": "distribution_profile",
    "treemap": "hierarchical_concentration",
    "sunburst": "hierarchical_concentration",
    "heatmap": "matrix_correlation",
    "growth_heatmap": "matrix_correlation",
    "officer_performance_matrix": "matrix_correlation",
    "pareto": "pareto_distribution",
    "sankey": "flow_network",
    "scatter": "risk_scatter",
    "bubble": "risk_scatter",
    "risk_matrix": "risk_scatter",
    "reopened_complaint_analysis": "reopen_analysis",
    "geographical_concentration_map": "geospatial_concentration",
}

_INTERPRETABLE_STATUS_OK: frozenset = frozenset({"eligible", "ok", "success"})


# ── Private generic helpers ──────────────────────────────────────────────────

def _resolve_lexicon(language_domain: Any) -> Dict[str, str]:
    key = str(language_domain or _DEFAULT_DOMAIN).strip().lower()
    return _DOMAIN_LEXICON.get(key, _DOMAIN_LEXICON[_DEFAULT_DOMAIN])


def _format_number(value: Any, decimals: int = 2) -> str:
    """Deterministic, precision-safe numeric formatter. Never raises; always
    degrades to 'N/A' on non-finite or non-numeric input."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "N/A"
    if not np.isfinite(v):
        return "N/A"
    if abs(v - round(v)) < 1e-9:
        return f"{v:,.0f}"
    return f"{v:,.{decimals}f}"


def _safe_label(value: Any, max_len: int = 60) -> str:
    text = str(value).strip()
    if not text or text.lower() in ("nan", "none", "nat", "<na>"):
        return "Unlabeled"
    if len(text) > max_len:
        return text[: max_len - 1] + "…"
    return text


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    return df.select_dtypes(include=[np.number]).columns.tolist()


def _categorical_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = set(_numeric_columns(df))
    return [c for c in df.columns if c not in numeric_cols]


def _detect_anomaly_series(series: pd.Series) -> bool:
    """Dual-method (Z-score + IQR) outlier detector. Returns False (never
    raises) whenever there is insufficient data or zero variance."""
    clean = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < _MIN_POINTS_FOR_ANOMALY_CHECK:
        return False
    std = float(clean.std(ddof=0))
    if std and np.isfinite(std) and std > 0:
        mean = float(clean.mean())
        z = (clean - mean).abs() / std
        if bool((z > _ANOMALY_Z_THRESHOLD).any()):
            return True
    q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
    iqr = q3 - q1
    if iqr and iqr > 0:
        lower = q1 - _ANOMALY_IQR_MULTIPLIER * iqr
        upper = q3 + _ANOMALY_IQR_MULTIPLIER * iqr
        return bool(((clean < lower) | (clean > upper)).any())
    return False


def _peak_trough_from_series(
    labels: pd.Series,
    values: pd.Series,
) -> Tuple[str, str, Optional[float], Optional[float]]:
    """Vectorized peak/trough resolver. Returns safe fallback text and None
    values when no computable numeric pair exists."""
    combined = pd.DataFrame({"__label": labels.astype(str), "__value": pd.to_numeric(values, errors="coerce")})
    combined["__value"] = combined["__value"].replace([np.inf, -np.inf], np.nan)
    combined = combined.dropna(subset=["__value"])
    if combined.empty:
        return _NA_TEXT, _NA_TEXT, None, None
    max_row = combined.loc[combined["__value"].idxmax()]
    min_row = combined.loc[combined["__value"].idxmin()]
    peak_val = float(max_row["__value"])
    trough_val = float(min_row["__value"])
    peak = f"{_safe_label(max_row['__label'])}: {_format_number(peak_val)}"
    trough = f"{_safe_label(min_row['__label'])}: {_format_number(trough_val)}"
    return peak, trough, peak_val, trough_val


def _build_insight(insight_type: str, headline: str, body_text: str, severity: str) -> Dict[str, str]:
    return {
        "insight_type": insight_type,
        "headline": headline,
        "body_text": body_text,
        "severity": severity,
    }


def _empty_statistical_summary() -> Dict[str, Any]:
    return {
        "peak_coordinate": _NA_TEXT,
        "trough_coordinate": _NA_TEXT,
        "is_anomaly_detected": False,
    }


def _no_data_insight(headline: str, body_text: str) -> List[Dict[str, str]]:
    return [_build_insight(_INSIGHT_ELIGIBILITY, headline, body_text, _SEVERITY_INFO)]


# ── Category interpretation handlers ─────────────────────────────────────────

def _interpret_categorical_bar(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    cat_cols = _categorical_columns(df)
    num_cols = _numeric_columns(df)
    if not cat_cols or not num_cols:
        return _empty_statistical_summary(), _no_data_insight(
            "Insufficient Categorical Data",
            "The aggregated payload does not contain both a categorical dimension and a "
            "numeric measure required to compute concentration insights.",
        )

    category_col = cat_cols[0]
    value_col = num_cols[-1]
    grouped = df.groupby(category_col, dropna=False)[value_col].sum(numeric_only=True).reset_index()
    grouped[value_col] = pd.to_numeric(grouped[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    grouped = grouped.dropna(subset=[value_col])
    if grouped.empty:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Values",
            "All aggregated values were non-numeric or null after cleaning.",
        )

    peak, trough, peak_val, trough_val = _peak_trough_from_series(grouped[category_col], grouped[value_col])
    anomaly = _detect_anomaly_series(grouped[value_col])
    total = float(grouped[value_col].sum())
    share_pct = (peak_val / total * 100.0) if total > 0 and peak_val is not None else 0.0
    plural = lexicon["record_plural"]

    if share_pct >= _HIGH_CONCENTRATION_THRESHOLD_PCT:
        severity, headline = _SEVERITY_CRITICAL, "High Volumetric Concentration Detected"
    elif share_pct >= _MODERATE_CONCENTRATION_THRESHOLD_PCT:
        severity, headline = _SEVERITY_WARNING, "Moderate Concentration Across Categories"
    else:
        severity, headline = _SEVERITY_INFO, "Balanced Distribution Across Categories"

    body = (
        f"The leading category accounts for {_format_number(share_pct)}% of total {plural} "
        f"({_format_number(peak_val)} of {_format_number(total)}), while the lowest-volume "
        f"category recorded only {_format_number(trough_val)}."
    )
    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_CONCENTRATION, headline, body, severity)]
    if anomaly:
        insights.append(_build_insight(
            _INSIGHT_ANOMALY,
            "Statistical Outlier Category Identified",
            "One or more categories deviate significantly (beyond 2.5 standard deviations or the "
            "1.5x interquartile range) from the group's typical volume, warranting root-cause review.",
            _SEVERITY_WARNING,
        ))

    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_time_series(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    cols = list(df.columns)
    num_cols = _numeric_columns(df)
    if not cols or not num_cols:
        return _empty_statistical_summary(), _no_data_insight(
            "No Time Series Data",
            "The aggregated payload contains no numeric time-indexed values to interpret.",
        )

    period_col = cols[0]
    raw_candidates = [c for c in num_cols if "rolling" not in str(c).lower()]
    value_col = raw_candidates[0] if raw_candidates else num_cols[0]

    work = df[[period_col, value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[value_col])
    if work.empty:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Values",
            "All period values were non-numeric or null after cleaning.",
        )

    peak, trough, peak_val, trough_val = _peak_trough_from_series(work[period_col], work[value_col])
    anomaly = _detect_anomaly_series(work[value_col])

    n = len(work)
    slope = 0.0
    if n >= 2:
        x = np.arange(n, dtype=float)
        y = work[value_col].to_numpy(dtype=float)
        try:
            slope, _intercept = np.polyfit(x, y, 1)
        except (np.linalg.LinAlgError, ValueError):
            slope = 0.0

    mean_val = float(work[value_col].mean()) if n else 0.0
    relative_slope = (slope / abs(mean_val) * 100.0) if mean_val != 0.0 and np.isfinite(mean_val) else 0.0
    plural = lexicon["record_plural"]

    if relative_slope > 5.0:
        severity, headline = _SEVERITY_WARNING, "Upward Trend Detected"
        body = (
            f"{plural.capitalize()} volume is trending upward at an average rate of "
            f"~{_format_number(abs(relative_slope))}% per period."
        )
    elif relative_slope < -5.0:
        severity, headline = _SEVERITY_POSITIVE, "Downward Trend Detected"
        body = (
            f"{plural.capitalize()} volume is trending downward at an average rate of "
            f"~{_format_number(abs(relative_slope))}% per period, indicating improving conditions."
        )
    else:
        severity, headline = _SEVERITY_INFO, "Stable Trend Across Periods"
        body = (
            f"{plural.capitalize()} volume remains broadly stable across the observed periods "
            f"with no material directional shift."
        )

    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_TREND, headline, body, severity)]
    if anomaly:
        insights.append(_build_insight(
            _INSIGHT_ANOMALY,
            "Period-Level Anomaly Detected",
            "At least one period recorded a value that deviates sharply from the surrounding trend, "
            "suggesting a possible data-entry issue, seasonal spike, or operational event.",
            _SEVERITY_WARNING,
        ))

    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_peak_hour(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    summary, insights = _interpret_time_series(df, metadata, lexicon, kwargs)
    for insight in insights:
        if insight["insight_type"] == _INSIGHT_TREND:
            insight["insight_type"] = _INSIGHT_EXTREME
            insight["headline"] = "Peak Intraday Activity Window Identified"
            insight["severity"] = _SEVERITY_INFO
            insight["body_text"] = (
                f"Intake peaks at {summary['peak_coordinate']}, while the quietest window is "
                f"{summary['trough_coordinate']}. Staffing and escalation workflows should align "
                f"with this intraday pattern."
            )
    return summary, insights


def _interpret_distribution(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    cols = list(df.columns)
    plural = lexicon["record_plural"]

    mean_v = std_v = min_v = max_v = median_v = np.nan

    if "Statistic" in cols:
        value_col = next((c for c in cols if c != "Statistic"), None)
        if value_col is None:
            return _empty_statistical_summary(), _no_data_insight(
                "No Measure Column",
                "The distribution table has no measurable value column.",
            )
        stats_series = pd.to_numeric(df.set_index("Statistic")[value_col], errors="coerce")
        mean_v = float(stats_series.get("mean", np.nan))
        std_v = float(stats_series.get("std", np.nan))
        min_v = float(stats_series.get("min", np.nan))
        max_v = float(stats_series.get("max", np.nan))
        median_v = float(stats_series.get("50%", np.nan))
    elif {"count", "mean"}.issubset(set(cols)):
        weights = pd.to_numeric(df["count"], errors="coerce").fillna(0.0)
        total_weight = float(weights.sum())
        mean_series = pd.to_numeric(df["mean"], errors="coerce")
        mean_v = float((mean_series * weights).sum() / total_weight) if total_weight > 0 else float(mean_series.mean())
        std_v = float(pd.to_numeric(df["std"], errors="coerce").mean()) if "std" in cols else np.nan
        min_v = float(pd.to_numeric(df["min"], errors="coerce").min()) if "min" in cols else np.nan
        max_v = float(pd.to_numeric(df["max"], errors="coerce").max()) if "max" in cols else np.nan
        median_v = float(pd.to_numeric(df["50%"], errors="coerce").mean()) if "50%" in cols else np.nan
    else:
        numeric_block = df.select_dtypes(include=[np.number])
        if numeric_block.empty:
            return _empty_statistical_summary(), _no_data_insight(
                "No Measurable Columns",
                "The distribution table has no numeric statistics to interpret.",
            )
        stacked = numeric_block.stack()
        if stacked.empty:
            return _empty_statistical_summary(), _no_data_insight(
                "No Computable Statistics",
                "Distribution statistics could not be computed from the aggregated payload.",
            )
        mean_v = float(stacked.mean())
        std_v = float(stacked.std(ddof=0))
        min_v = float(stacked.min())
        max_v = float(stacked.max())
        median_v = float(stacked.median())

    finite_vals = [v for v in (mean_v, std_v, min_v, max_v, median_v) if v is not None and np.isfinite(v)]
    if not finite_vals:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Statistics",
            "Distribution statistics could not be computed from the aggregated payload.",
        )

    peak = f"Maximum: {_format_number(max_v)}" if np.isfinite(max_v) else _NA_TEXT
    trough = f"Minimum: {_format_number(min_v)}" if np.isfinite(min_v) else _NA_TEXT

    anomaly = False
    if np.isfinite(std_v) and std_v > 0 and np.isfinite(mean_v):
        if np.isfinite(max_v) and abs(max_v - mean_v) / std_v > _ANOMALY_Z_THRESHOLD:
            anomaly = True
        if np.isfinite(min_v) and abs(mean_v - min_v) / std_v > _ANOMALY_Z_THRESHOLD:
            anomaly = True

    spread_pct = (std_v / mean_v * 100.0) if np.isfinite(std_v) and np.isfinite(mean_v) and mean_v != 0 else 0.0
    if spread_pct >= 75.0:
        severity, headline = _SEVERITY_WARNING, "High Variability in Distribution"
    elif spread_pct >= 35.0:
        severity, headline = _SEVERITY_INFO, "Moderate Variability in Distribution"
    else:
        severity, headline = _SEVERITY_POSITIVE, "Tightly Clustered Distribution"

    body = (
        f"The {plural} distribution has a mean of {_format_number(mean_v)} "
        f"(median {_format_number(median_v)}) with values spanning from "
        f"{_format_number(min_v)} to {_format_number(max_v)}."
    )
    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_DISTRIBUTION, headline, body, severity)]
    if anomaly:
        insights.append(_build_insight(
            _INSIGHT_ANOMALY,
            "Extreme Value Detected in Distribution",
            "The observed maximum or minimum deviates by more than 2.5 standard deviations from the "
            "mean, indicating a potential long-tail case requiring individual review.",
            _SEVERITY_WARNING,
        ))

    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_hierarchical(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    num_cols = _numeric_columns(df)
    if not num_cols:
        return _empty_statistical_summary(), _no_data_insight(
            "No Measure Column",
            "The hierarchy table has no numeric value column to rank nodes by.",
        )

    value_col = num_cols[-1]
    path_cols = [c for c in df.columns if c != value_col and c not in num_cols]
    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[value_col])
    if work.empty or not path_cols:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Nodes",
            "No leaf nodes in the hierarchy contained a computable numeric value.",
        )

    labels = work[path_cols].astype(str).agg(" › ".join, axis=1)
    peak, trough, peak_val, trough_val = _peak_trough_from_series(labels, work[value_col])
    total = float(work[value_col].sum())
    share_pct = (peak_val / total * 100.0) if total > 0 and peak_val is not None else 0.0
    anomaly = _detect_anomaly_series(work[value_col])
    plural = lexicon["record_plural"]

    if share_pct >= _HIGH_CONCENTRATION_THRESHOLD_PCT:
        severity, headline = _SEVERITY_CRITICAL, "Severe Hierarchical Concentration"
    elif share_pct >= _MODERATE_CONCENTRATION_THRESHOLD_PCT:
        severity, headline = _SEVERITY_WARNING, "Notable Hierarchical Concentration"
    else:
        severity, headline = _SEVERITY_INFO, "Distributed Load Across Hierarchy"

    body = (
        f"The deepest-contributing branch accounts for {_format_number(share_pct)}% of total {plural} "
        f"({_format_number(peak_val)} of {_format_number(total)}) at '{peak.split(':')[0]}'."
    )
    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_CONCENTRATION, headline, body, severity)]
    if anomaly:
        insights.append(_build_insight(
            _INSIGHT_ANOMALY,
            "Outlier Branch Identified",
            "One or more hierarchy branches carry disproportionately high or low volume relative to peers.",
            _SEVERITY_WARNING,
        ))

    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_matrix(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    num_cols = _numeric_columns(df)
    label_cols = [c for c in df.columns if c not in num_cols]
    if not num_cols or not label_cols:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Matrix",
            "The matrix table lacks either a row-label column or numeric metric columns.",
        )

    row_label_col = label_cols[0]
    numeric_block = df[num_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    stacked = numeric_block.stack()
    if stacked.empty:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Cells",
            "All matrix cells were non-numeric or null after cleaning.",
        )

    max_idx = stacked.idxmax()
    min_idx = stacked.idxmin()
    max_row_pos, max_col = max_idx
    min_row_pos, min_col = min_idx
    max_row_label = _safe_label(df.loc[max_row_pos, row_label_col])
    min_row_label = _safe_label(df.loc[min_row_pos, row_label_col])
    peak_val = float(stacked.loc[max_idx])
    trough_val = float(stacked.loc[min_idx])
    peak = f"{max_row_label} × {_safe_label(max_col)}: {_format_number(peak_val)}"
    trough = f"{min_row_label} × {_safe_label(min_col)}: {_format_number(trough_val)}"

    anomaly = _detect_anomaly_series(stacked)
    mean_v = float(stacked.mean())

    if anomaly:
        severity, headline = _SEVERITY_WARNING, "Significant Cell-Level Deviation Identified"
        body = (
            f"The matrix cell '{peak}' deviates substantially from the average cell value of "
            f"{_format_number(mean_v)}, warranting a targeted review of that intersection."
        )
        insight_type = _INSIGHT_ANOMALY
    else:
        severity, headline = _SEVERITY_INFO, "Matrix Values Within Expected Range"
        body = (
            f"Matrix values are broadly consistent, ranging from {_format_number(trough_val)} to "
            f"{_format_number(peak_val)} around an average of {_format_number(mean_v)}."
        )
        insight_type = _INSIGHT_DISTRIBUTION

    insights: List[Dict[str, str]] = [_build_insight(insight_type, headline, body, severity)]
    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_pareto(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    cols = list(df.columns)
    cat_col = cols[0]
    cum_col = next((c for c in cols if "cumulative" in str(c).lower()), None)
    class_col = next((c for c in cols if str(c).lower() == "pareto class"), None)
    value_col = next((c for c in _numeric_columns(df) if c != cum_col), None)

    if value_col is None:
        return _empty_statistical_summary(), _no_data_insight(
            "No Measure Column",
            "The Pareto table has no numeric value column to rank contributors by.",
        )

    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[value_col])
    if work.empty:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Values",
            "All Pareto contributor values were non-numeric or null after cleaning.",
        )

    peak, trough, peak_val, trough_val = _peak_trough_from_series(work[cat_col], work[value_col])
    anomaly = _detect_anomaly_series(work[value_col])
    plural = lexicon["record_plural"]

    vital_few_count = int((work[class_col] == "Vital Few").sum()) if class_col and class_col in work.columns else 0
    total_count = int(len(work))
    vital_pct = (vital_few_count / total_count * 100.0) if total_count else 0.0

    concentrated = bool(vital_few_count > 0 and vital_pct <= 40.0)
    if concentrated:
        headline, severity = "80/20 Pareto Concentration Confirmed", _SEVERITY_WARNING
    else:
        headline, severity = "Contributor Base Broadly Distributed", _SEVERITY_INFO

    body = (
        f"Just {vital_few_count} of {total_count} contributor categor"
        f"{'y' if vital_few_count == 1 else 'ies'} ({_format_number(vital_pct)}% of all categories) "
        f"account for the majority of total {plural}, led by '{peak.split(':')[0]}'."
    )
    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_CONCENTRATION, headline, body, severity)]
    if anomaly:
        insights.append(_build_insight(
            _INSIGHT_ANOMALY,
            "Disproportionate Contributor Identified",
            "One contributor category carries a statistically disproportionate share of total volume "
            "relative to its peers.",
            _SEVERITY_WARNING,
        ))

    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_flow(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    required = {"Source", "Target", "Flow"}
    if not required.issubset(set(df.columns)):
        return _empty_statistical_summary(), _no_data_insight(
            "No Flow Data",
            "The aggregated payload does not contain a resolvable Source/Target/Flow structure.",
        )

    work = df.copy()
    work["Flow"] = pd.to_numeric(work["Flow"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=["Flow"])
    if work.empty:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Flows",
            "All flow values were non-numeric or null after cleaning.",
        )

    labels = work["Source"].astype(str) + " → " + work["Target"].astype(str)
    peak, trough, peak_val, trough_val = _peak_trough_from_series(labels, work["Flow"])
    anomaly = _detect_anomaly_series(work["Flow"])
    total = float(work["Flow"].sum())
    share_pct = (peak_val / total * 100.0) if total > 0 and peak_val is not None else 0.0
    plural = lexicon["record_plural"]

    if share_pct >= _HIGH_CONCENTRATION_THRESHOLD_PCT:
        severity, headline = _SEVERITY_WARNING, "Dominant Workflow Path Identified"
    else:
        severity, headline = _SEVERITY_INFO, "Workflow Paths Reasonably Distributed"

    body = (
        f"The dominant transition '{peak.split(':')[0]}' carries {_format_number(share_pct)}% of "
        f"total tracked {plural} flow ({_format_number(peak_val)} of {_format_number(total)})."
    )
    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_FLOW, headline, body, severity)]
    if anomaly:
        insights.append(_build_insight(
            _INSIGHT_ANOMALY,
            "Irregular Flow Volume Detected",
            "At least one workflow transition carries a volume that is statistically disproportionate "
            "relative to other tracked paths.",
            _SEVERITY_WARNING,
        ))

    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_risk_scatter(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    num_cols = _numeric_columns(df)
    label_cols = [c for c in df.columns if c not in num_cols]
    if not num_cols:
        return _empty_statistical_summary(), _no_data_insight(
            "No Numeric Measures",
            "The scatter/risk table has no numeric measures to rank by.",
        )

    rank_col = "Risk Score" if "Risk Score" in num_cols else num_cols[-1]
    label_col = label_cols[0] if label_cols else None

    work = df.copy()
    work[rank_col] = pd.to_numeric(work[rank_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    work = work.dropna(subset=[rank_col])
    if work.empty:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Values",
            "All ranking values were non-numeric or null after cleaning.",
        )

    labels = work[label_col].astype(str) if label_col else pd.Series(
        [f"Row {i}" for i in range(len(work))], index=work.index
    )
    peak, trough, peak_val, trough_val = _peak_trough_from_series(labels, work[rank_col])
    anomaly = _detect_anomaly_series(work[rank_col])
    plural = lexicon["record_plural"]

    if anomaly:
        severity, headline = _SEVERITY_CRITICAL, "High-Risk Concentration Identified"
        body = (
            f"'{peak.split(':')[0]}' shows a materially elevated {rank_col.lower()} relative to peers, "
            f"placing it in the highest-priority risk quadrant for {plural} management."
        )
    else:
        severity, headline = _SEVERITY_INFO, "Risk Levels Broadly Comparable Across Groups"
        body = (
            f"{rank_col} values range from {_format_number(trough_val)} to {_format_number(peak_val)} "
            f"without any group showing extreme statistical deviation."
        )

    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_RISK, headline, body, severity)]
    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_reopen_analysis(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    if "Reopen Rate (%)" not in df.columns:
        return _empty_statistical_summary(), _no_data_insight(
            "No Reopen Data",
            "The aggregated payload does not contain a Reopen Rate (%) column to interpret.",
        )

    label_cols = [c for c in df.columns if c not in _numeric_columns(df)]
    label_col = label_cols[0] if label_cols else df.columns[0]

    work = df.copy()
    work["Reopen Rate (%)"] = pd.to_numeric(work["Reopen Rate (%)"], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    work = work.dropna(subset=["Reopen Rate (%)"])
    if work.empty:
        return _empty_statistical_summary(), _no_data_insight(
            "No Computable Rates",
            "All reopen-rate values were non-numeric or null after cleaning.",
        )

    peak, trough, peak_val, trough_val = _peak_trough_from_series(work[label_col], work["Reopen Rate (%)"])
    anomaly = _detect_anomaly_series(work["Reopen Rate (%)"])
    plural = lexicon["record_plural"]

    if peak_val is not None and peak_val >= 20.0:
        severity, headline = _SEVERITY_CRITICAL, "Elevated Reopen Rate Requires Intervention"
    elif peak_val is not None and peak_val >= 10.0:
        severity, headline = _SEVERITY_WARNING, "Reopen Rate Above Comfortable Threshold"
    else:
        severity, headline = _SEVERITY_POSITIVE, "Reopen Rates Within Acceptable Range"

    body = (
        f"'{peak.split(':')[0]}' shows the highest reopen incidence at {_format_number(peak_val)}%, "
        f"signaling potential first-time resolution quality gaps for that group's {plural}."
    )
    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_RISK, headline, body, severity)]
    if anomaly:
        insights.append(_build_insight(
            _INSIGHT_ANOMALY,
            "Reopen Rate Outlier Group",
            "One group's reopen rate deviates sharply from the rest, warranting a targeted quality audit.",
            _SEVERITY_WARNING,
        ))

    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


def _interpret_geospatial(
    df: pd.DataFrame,
    metadata: Dict[str, Any],
    lexicon: Dict[str, str],
    kwargs: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    plural = lexicon["record_plural"]
    total = int(len(df))
    resolved = total

    status_col = next((c for c in df.columns if str(c).lower() == "geolocation status"), None)
    if status_col:
        resolved = int(df[status_col].astype(str).str.startswith("Resolved").sum())

    resolution_pct = (resolved / total * 100.0) if total else 0.0
    peak = f"Resolved: {_format_number(resolved)}"
    trough = f"Unresolved: {_format_number(total - resolved)}"
    anomaly = resolution_pct < 50.0

    if resolution_pct >= 90.0:
        severity, headline = _SEVERITY_POSITIVE, "High Geospatial Coverage Achieved"
    elif resolution_pct >= 50.0:
        severity, headline = _SEVERITY_INFO, "Moderate Geospatial Coverage"
    else:
        severity, headline = _SEVERITY_WARNING, "Low Geospatial Resolution Coverage"

    body = (
        f"{_format_number(resolution_pct)}% of {plural} ({_format_number(resolved)} of "
        f"{_format_number(total)}) were successfully geolocated for mapping; the remainder lacked "
        f"resolvable coordinates."
    )
    insights: List[Dict[str, str]] = [_build_insight(_INSIGHT_COVERAGE, headline, body, severity)]
    summary = {"peak_coordinate": peak, "trough_coordinate": trough, "is_anomaly_detected": anomaly}
    return summary, insights


# ── Category dispatch registry ───────────────────────────────────────────────

_CATEGORY_HANDLERS: Dict[
    str,
    Callable[[pd.DataFrame, Dict[str, Any], Dict[str, str], Dict[str, Any]], Tuple[Dict[str, Any], List[Dict[str, str]]]],
] = {
    "categorical_bar": _interpret_categorical_bar,
    "time_series_line": _interpret_time_series,
    "peak_hour": _interpret_peak_hour,
    "distribution_profile": _interpret_distribution,
    "hierarchical_concentration": _interpret_hierarchical,
    "matrix_correlation": _interpret_matrix,
    "pareto_distribution": _interpret_pareto,
    "flow_network": _interpret_flow,
    "risk_scatter": _interpret_risk_scatter,
    "reopen_analysis": _interpret_reopen_analysis,
    "geospatial_concentration": _interpret_geospatial,
}


# ── Public entry point ───────────────────────────────────────────────────────

def interpret_chart_output(
    chart_type: str,
    aggregated_df: Optional[pd.DataFrame],
    metadata: Dict[str, Any],
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Translates the (aggregated_df, metadata) output of a
    visualization/chart_factory.py render function into a structured,
    natural-language interpretation payload.

    Args:
        chart_type: The CHART_REGISTRY key that produced the payload (e.g.
            "bar", "line", "pareto", "risk_matrix", "sankey", ...).
        aggregated_df: The export-ready DataFrame returned alongside the
            figure/map object. May be None or empty for ineligible charts.
        metadata: The metadata dict returned by the same render call
            (either the Part 1/2 `_build_metadata` contract or the Part 3
            `_build_universal_metadata` contract — both expose a "status"
            and "reason" key, which is all this module relies on).
        **kwargs:
            language_domain (str): Tailors phrasing vocabulary. One of
                "power_utility" (default), "revenue_analytics",
                "hr_analytics", or "generic". Unknown values fall back to
                the default domain rather than raising.

    Returns:
        A dict strictly matching:
            {
                "chart_type": str,
                "statistical_summary": {
                    "peak_coordinate": str,
                    "trough_coordinate": str,
                    "is_anomaly_detected": bool,
                },
                "insights": [
                    {"insight_type": str, "headline": str, "body_text": str, "severity": str},
                    ...
                ],
            }

    This function never raises: every failure path (ineligible chart, empty
    dataframe, missing columns, non-numeric payloads, unrecognized
    chart_type) degrades to a structured, safe fallback payload.
    """
    safe_metadata: Dict[str, Any] = metadata if isinstance(metadata, dict) else {}
    lexicon = _resolve_lexicon(kwargs.get("language_domain"))

    status = str(safe_metadata.get("status", "Eligible")).strip().lower()
    if status not in _INTERPRETABLE_STATUS_OK:
        reason = str(
            safe_metadata.get("reason")
            or "The requested visualization could not be computed with the currently mapped roles."
        )
        return {
            "chart_type": chart_type,
            "statistical_summary": _empty_statistical_summary(),
            "insights": _no_data_insight("Visualization Not Currently Eligible", reason),
        }

    if not isinstance(aggregated_df, pd.DataFrame) or aggregated_df.empty:
        return {
            "chart_type": chart_type,
            "statistical_summary": _empty_statistical_summary(),
            "insights": _no_data_insight(
                "No Aggregated Data Available",
                "The visualization returned no rows to analyze — this may indicate an empty dataset "
                "after filtering, or a role mapping producing zero matches.",
            ),
        }

    category = _CHART_TYPE_CATEGORY_MAP.get(chart_type, "categorical_bar")
    handler = _CATEGORY_HANDLERS.get(category, _interpret_categorical_bar)

    try:
        df_local = aggregated_df.copy(deep=True)
        summary, insights = handler(df_local, safe_metadata, lexicon, kwargs)
    except Exception as exc:  # noqa: BLE001 — absolute final safety net
        summary = _empty_statistical_summary()
        insights = _no_data_insight(
            "Interpretation Engine Encountered an Issue",
            f"The automated interpreter could not fully analyze this visualization: {exc}",
        )

    if not isinstance(summary, dict) or "peak_coordinate" not in summary:
        summary = _empty_statistical_summary()
    if not insights:
        insights = _no_data_insight(
            "No Notable Patterns Detected",
            "The visualization computed successfully but did not surface any statistically "
            "significant concentration, trend, or anomaly patterns.",
        )

    return {
        "chart_type": chart_type,
        "statistical_summary": summary,
        "insights": insights,
    }