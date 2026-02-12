import json

from google import genai

from core.models import AnalysisReport, DataSummary


def generate_report(client: genai.Client, summary: DataSummary, insights: str, knowledge: str) -> AnalysisReport:
    key_metrics = {
        "rows": summary.n_rows,
        "anomalies": summary.anomalies.total_flagged_rows,
        "correlations": [
            f"{c.pair[0]} vs {c.pair[1]} ({c.pearson:.2f})"
            for c in summary.top_correlations[:3]
        ],
    }

    prompt = f"""
You are the ReportAgent. Synthesize the final deliverables.

Inputs:
1. Key Metrics: {json.dumps(key_metrics)}
2. Expert Insights: {insights}
3. Knowledge Context: {knowledge}

Task: Create a JSON object with:
- technicalReport: Detailed Markdown for engineers. Sections: Analysis Methodology, Key Findings (Trend/Anomaly), Root Cause Hypothesis, Recommended Actions (citing SOPs).
- executiveSummary: Concise Markdown for Plant Manager. Focus on: Yield Impact, Quality Risk, Business Decision. (Bullet points).
- anomalies: A list of short strings describing top detected issues (e.g. "Temp drift > 5%").
"""
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
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
