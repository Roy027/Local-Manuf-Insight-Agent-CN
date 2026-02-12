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

    system_instruction = """
你是 ReportAgent（报告智能体）。负责整合最终交付物。
你必须仅使用中文生成 JSON 内容。
任务: 创建一个包含以下字段的 JSON 对象:
- technicalReport: 给工程师的详细 Markdown 报告 (中文)。
- executiveSummary: 给厂长的简明 Markdown 摘要 (中文)。
- anomalies: 描述与其相关的短字符串列表 (中文)。
"""
    prompt = f"""
输入 (Inputs):
1. 关键指标 (Key Metrics): {json.dumps(key_metrics)}
2. 专家洞察 (Expert Insights): {insights}
3. 知识背景 (Knowledge Context): {knowledge}
"""
    response = client.generate_content(
        prompt=prompt,
        system_instruction=system_instruction,
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
