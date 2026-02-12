import json
from typing import Any

import pandas as pd
from core.llm_client import BaseLLMClient

from core.models import DataSummary


def _prune_profile(summary: DataSummary, max_cols: int = 15) -> Any:
    # ... (unchanged)
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


def generate_insights(client: BaseLLMClient, summary: DataSummary) -> str:
    context = _prune_profile(summary)
    system_instruction = """
你是 InsightAgent（洞察智能体），一位制造业数据分析专家。
你必须仅使用中文进行回答。
任务:
1. 分析 time_profiles 中的趋势。
2. 解释 top_correlations（强相关性）。
3. 评估 anomalies（异常）。
输出:
提供一份关键技术发现、假设和潜在根本原因的列表。
任何分析结果都必须用中文表述。
"""
    prompt = f"""
输入数据概况 (Input Data Profile):
{json.dumps(context, indent=2)}
"""
    response = client.generate_content(prompt, system_instruction=system_instruction)
    return response.text or "No insights generated."
