from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew, zscore

from core.models import (
    CategoricalProfile,
    ColumnSchema,
    CorrelationPair,
    DataSummary,
    GlobalAnomalies,
    HistogramBin,
    NumericProfile,
    NumericStats,
    TimeMetricProfile,
    TimeProfile,
)

MAX_CORR_PAIRS = 20
HIST_BINS = 10
MAX_CATEGORIES = 20

SAMPLE_DATA_CSV = """BatchID,Temperature,Pressure,Humidity,Yield,DefectRate
1,150,4500,45,98.5,1.5
2,151,4510,46,98.2,1.8
3,149,4490,44,99.0,1.0
4,152,4520,45,97.8,2.2
5,155,4600,48,96.5,3.5
6,158,4650,50,92.0,8.0
7,160,4700,52,85.0,15.0
8,162,4750,55,80.0,20.0
9,150,4500,45,98.0,2.0
10,149,4495,44,98.8,1.2
"""

def _to_native(val):
    if hasattr(val, "item"):
        return val.item()
    if isinstance(val, (np.floating, np.integer)):
        return val.astype(float) if isinstance(val, np.floating) else int(val)
    return val


def _numeric_stats(values: pd.Series) -> NumericStats:
    arr = values.dropna().to_numpy()
    stats = NumericStats(
        mean=float(arr.mean()),
        std=float(arr.std(ddof=0)),
        min=float(arr.min()),
        max=float(arr.max()),
        p01=float(np.percentile(arr, 1)),
        p25=float(np.percentile(arr, 25)),
        p50=float(np.percentile(arr, 50)),
        p75=float(np.percentile(arr, 75)),
        p99=float(np.percentile(arr, 99)),
        skew=float(skew(arr, bias=False)),
        kurtosis=float(kurtosis(arr, bias=False)),
    )
    return stats


def _histogram(values: pd.Series) -> List[HistogramBin]:
    arr = values.dropna().to_numpy()
    if arr.size == 0:
        return []
    counts, edges = np.histogram(arr, bins=HIST_BINS)
    bins: List[HistogramBin] = []
    for i in range(len(counts)):
        bins.append(
            HistogramBin(
                range=f"{edges[i]:.1f} - {edges[i+1]:.1f}",
                count=int(counts[i]),
                min=float(edges[i]),
                max=float(edges[i + 1]),
            )
        )
    return bins


def _categorical_profile(series: pd.Series) -> CategoricalProfile:
    counts = series.astype(str).value_counts(dropna=False)
    top_values = []
    for val, cnt in counts.head(MAX_CATEGORIES).items():
        top_values.append(
            {"value": str(val), "count": int(cnt), "fraction": float(cnt / len(series))}
        )
    other_fraction = float(counts.iloc[MAX_CATEGORIES:].sum() / len(series)) if len(counts) > MAX_CATEGORIES else 0.0
    return CategoricalProfile(
        n_unique=int(counts.size),
        top_values=top_values,
        other_fraction=other_fraction,
        missing={"count": int(series.isna().sum()), "fraction": float(series.isna().mean())},
    )


def _correlations(df: pd.DataFrame, numeric_cols: List[str]) -> List[CorrelationPair]:
    pairs: List[CorrelationPair] = []
    if len(numeric_cols) < 2:
        return pairs
    corr_matrix = df[numeric_cols].corr().abs()
    seen: set[Tuple[str, str]] = set()
    for c1 in numeric_cols:
        for c2 in numeric_cols:
            if c1 == c2 or (c2, c1) in seen:
                continue
            seen.add((c1, c2))
            pearson = corr_matrix.loc[c1, c2]
            if pd.notna(pearson):
                pairs.append(
                    CorrelationPair(pair=(c1, c2), pearson=float(pearson), n_samples=len(df))
                )
    pairs.sort(key=lambda p: p.pearson, reverse=True)
    return pairs[:MAX_CORR_PAIRS]


def _time_profile(df: pd.DataFrame, index_col: str, numeric_cols: List[str]) -> TimeProfile:
    index_values = df[index_col] if index_col in df.columns else pd.Series(range(len(df)))
    metrics = {}
    for col in numeric_cols:
        if col == index_col:
            continue
        values = df[col]
        slope, r2 = _simple_regression(index_values, values)
        metrics[col] = TimeMetricProfile(
            trend={
                "slope": slope,
                "r2": r2,
                "direction": "flat" if abs(slope) < 1e-6 else ("up" if slope > 0 else "down"),
            },
            change_points=[
                {
                    "index": int(values.sub(values.mean()).abs().idxmax()),
                    "value": _to_native(values.iloc[values.sub(values.mean()).abs().argmax()]),
                    "delta_mean": float(values.sub(values.mean()).abs().max()),
                }
            ],
        )
    return TimeProfile(
        index_column=index_col,
        index_range={"min": _to_native(index_values.min()), "max": _to_native(index_values.max())},
        metrics=metrics,
    )


def _simple_regression(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
    x_arr = x.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)
    if len(x_arr) != len(y_arr) or len(x_arr) == 0:
        return 0.0, 0.0
    slope = np.polyfit(x_arr, y_arr, 1)[0]
    y_pred = slope * x_arr + (y_arr.mean() - slope * x_arr.mean())
    ss_tot = ((y_arr - y_arr.mean()) ** 2).sum()
    ss_res = ((y_arr - y_pred) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    return float(slope), float(r2)


def analyze_dataset(df: pd.DataFrame, file_name: str) -> DataSummary:
    if df.empty:
        raise ValueError("Empty dataset")

    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df.insert(0, "id", range(1, len(df) + 1))

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != "id"]

    schema = {}
    index_col = "id"
    for col in df.columns:
        if col == "id":
            continue
        dtype = str(df[col].dtype)
        role = "numeric" if col in numeric_cols else "categorical"
        if "time" in col.lower() or "date" in col.lower() or "cycle" in col.lower():
            index_col = col if col in numeric_cols else index_col
        schema[col] = ColumnSchema(dtype=dtype, role=role)

    numeric_profile = {}
    for col in numeric_cols:
        series = df[col]
        stats = _numeric_stats(series)
        hist = _histogram(series)
        missing_count = int(series.isna().sum())
        zero_count = int((series == 0).sum())
        outlier_mask = np.abs(zscore(series.fillna(series.mean()))) > 3
        outlier_count = int(outlier_mask.sum())
        numeric_profile[col] = NumericProfile(
            stats=stats,
            histogram=hist,
            missing={"count": missing_count, "fraction": missing_count / len(df)},
            zeros={"count": zero_count, "fraction": zero_count / len(df)},
            outliers={"count": outlier_count, "fraction": outlier_count / len(df)},
        )

    categorical_profile = {}
    for col in categorical_cols:
        categorical_profile[col] = _categorical_profile(df[col])

    top_correlations = _correlations(df, numeric_cols)
    time_profiles = _time_profile(df, index_col, numeric_cols) if numeric_cols else None

    outlier_rows = (np.abs(zscore(df[numeric_cols].fillna(df[numeric_cols].mean()))) > 3).any(
        axis=1
    ) if numeric_cols else pd.Series([False] * len(df))
    flagged = int(outlier_rows.sum())
    anomalies = GlobalAnomalies(
        total_flagged_rows=flagged,
        fraction=float(flagged / len(df)),
        rules=[{"rule": "Value > 3 Sigma", "count": flagged}],
    )

    sample_rows = df.head(5).to_dict(orient="records")

    return DataSummary(
        fileName=file_name,
        n_rows=len(df),
        n_cols=len(df.columns) - 1,
        schema=schema,
        numeric_profile=numeric_profile,
        categorical_profile=categorical_profile,
        top_correlations=top_correlations,
        time_profiles=time_profiles,
        anomalies=anomalies,
        sample_rows=sample_rows,
        raw_data_subset=df.to_dict(orient="records"),
    )
