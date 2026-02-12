from core.llm_client import BaseLLMClient

from core.knowledge import KNOWLEDGE_BASE_SOPS


def retrieve_knowledge(client: BaseLLMClient, current_insights: str) -> str:
    prompt = f"""
You are the KnowledgeAgent. You have access to the company's SOPs and Historical Cases (long-term memory).

Your Task:
Read the Current Insights and find relevant documents in the Knowledge Base.
Map specific observed problems to SOPs or past cases.

Current Insights:
{current_insights}

Knowledge Base:
{KNOWLEDGE_BASE_SOPS}

Output:
A set of citations and excerpts that explain or solve the issues found in the insights.
If an insight matches a Historical Case, explicitly mention it.
"""
    response = client.generate_content(prompt)
    return response.text or "No relevant knowledge found."
