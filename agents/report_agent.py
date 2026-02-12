import json

from core.llm_client import BaseLLMClient

from core.models import AnalysisReport, DataSummary


def generate_report(client: BaseLLMClient, summary: DataSummary, insights: str, knowledge: str) -> AnalysisReport:
    key_metrics = {
        "rows": summary.n_rows,
        "anomalies": summary.anomalies.total_flagged_rows,
        "correlations": [
            f"{c.pair[0]} vs {c.pair[1]} ({c.pearson:.2f})"
            for c in summary.top_correlations[:3]
        ],
    }

    prompt = f"""
你是 ReportAgent（报告智能体）。负责整合最终交付物。

输入 (Inputs):
1. 关键指标 (Key Metrics): {json.dumps(key_metrics)}
2. 专家洞察 (Expert Insights): {insights}
3. 知识背景 (Knowledge Context): {knowledge}

任务: 创建一个包含以下字段的 JSON 对象 (请确保内容为中文):
- technicalReport: 给工程师的详细 Markdown 报告。章节包括：分析方法、关键发现（趋势/异常）、根本原因假设、推荐行动（引用 SOP）。
- executiveSummary: 给厂长的简明 Markdown 摘要。重点关注：良率影响、质量风险、业务决策。（请使用项目符号）。
- anomalies: 描述与其相关的短字符串列表（例如 "温度漂移 > 5%"）。
"""
    response = client.generate_content(
        prompt=prompt,
        response_mime_type="application/json"
    )
    text = response.text
    if not text:
        raise RuntimeError("Failed to generate report")
    data = json.loads(text)
    return AnalysisReport(
        technicalReport=data.get("technicalReport", ""),
        executiveSummary=data.get("executiveSummary", ""),
        anomalies=data.get("anomalies", []),
    )
