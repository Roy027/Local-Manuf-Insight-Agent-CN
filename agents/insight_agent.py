import json
from typing import Any

import pandas as pd
from google import genai

from core.models import DataSummary


def _prune_profile(summary: DataSummary, max_cols: int = 15) -> Any:
    numeric_keys = list(summary.numeric_profile.keys())[:max_cols]
    categorical_keys = list(summary.categorical_profile.keys())[:max_cols]

    # Convert nested dataclasses to dicts to ensure JSON serializability
    def _flatten(obj: Any) -> Any:
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        if hasattr(obj, "__dict__"):
            return {key: _flatten(value) for key, value in obj.__dict__.items()}
        if isinstance(obj, list):
            return [_flatten(item) for item in obj]
        if isinstance(obj, dict):
            return {key: _flatten(value) for key, value in obj.items()}
        return obj

    numeric = {k: _flatten(summary.numeric_profile[k]) for k in numeric_keys}
    categorical = {k: _flatten(summary.categorical_profile[k]) for k in categorical_keys}

    return {
        "n_rows": summary.n_rows,
        "n_cols": summary.n_cols,
        "numeric_profile": numeric,
        "categorical_profile": categorical,
        "top_correlations": [_flatten(c) for c in summary.top_correlations],
        "time_profiles": _flatten(summary.time_profiles) if summary.time_profiles else None,
        "anomalies": _flatten(summary.anomalies),
        "sample": [_flatten(row) for row in summary.sample_rows],
    }


def generate_insights(client: genai.Client, summary: DataSummary) -> str:
    context = _prune_profile(summary)
    prompt = f"""
You are the InsightAgent, an expert in manufacturing data analytics.
You are provided with a statistical profile of the dataset.
Do not request raw data. Use the provided statistics, trends, and correlations.

Input Data Profile:
{json.dumps(context, indent=2)}

Task:
1. Analyze trends in the time_profiles. Are there degradations?
2. Interpret top_correlations. Do they indicate physical relationships (e.g. Temp vs Pressure)?
3. Evaluate anomalies. Is the dataset stable or noisy?

Output:
Provide a list of key technical findings, hypotheses, and potential root causes.
Focus on deviations from normality.
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
    )
    return response.text or "No insights generated."
