from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union


Role = Literal["numeric", "categorical", "datetime", "index", "unknown"]


@dataclass
class ColumnSchema:
    dtype: str
    role: Role


@dataclass
class NumericStats:
    mean: float
    std: float
    min: float
    max: float
    p01: float
    p25: float
    p50: float
    p75: float
    p99: float
    skew: float
    kurtosis: float


@dataclass
class HistogramBin:
    range: str
    count: int
    min: float
    max: float


@dataclass
class NumericProfile:
    stats: NumericStats
    histogram: List[HistogramBin]
    missing: Dict[str, float]
    zeros: Dict[str, float]
    outliers: Dict[str, float]


@dataclass
class CategoricalProfile:
    n_unique: int
    top_values: List[Dict[str, Union[str, int, float]]]
    other_fraction: float
    missing: Dict[str, float]


@dataclass
class CorrelationPair:
    pair: Tuple[str, str]
    pearson: float
    n_samples: int


@dataclass
class TimeMetricProfile:
    trend: Dict[str, Union[float, str]]
    change_points: List[Dict[str, Union[int, float]]]


@dataclass
class TimeProfile:
    index_column: str
    index_range: Dict[str, Union[int, float, str]]
    metrics: Dict[str, TimeMetricProfile]


@dataclass
class GlobalAnomalies:
    total_flagged_rows: int
    fraction: float
    rules: List[Dict[str, Union[str, int]]]


@dataclass
class DataSummary:
    fileName: str
    n_rows: int
    n_cols: int
    schema: Dict[str, ColumnSchema]
    numeric_profile: Dict[str, NumericProfile]
    categorical_profile: Dict[str, CategoricalProfile]
    top_correlations: List[CorrelationPair]
    time_profiles: Optional[TimeProfile]
    anomalies: GlobalAnomalies
    sample_rows: List[Dict[str, Union[int, float, str]]]
    raw_data_subset: List[Dict[str, Union[int, float, str]]] = field(default_factory=list)


@dataclass
class AnalysisReport:
    technicalReport: str
    executiveSummary: str
    anomalies: List[str]
