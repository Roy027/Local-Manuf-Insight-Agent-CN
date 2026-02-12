from core.llm_client import BaseLLMClient

from core.knowledge import KNOWLEDGE_BASE_SOPS


def retrieve_knowledge(client: BaseLLMClient, current_insights: str) -> str:
    prompt = f"""
你是 KnowledgeAgent（知识智能体）。你可以访问公司的 SOP 和历史案例（长期记忆）。

你的任务:
阅读当前的洞察（Current Insights），并在知识库中查找相关文档。
将观察到的具体问题映射到 SOP 或过往案例。

当前洞察 (Current Insights):
{current_insights}

知识库 (Knowledge Base):
{KNOWLEDGE_BASE_SOPS}

输出:
请**必须**用中文提供一组引文和摘录，解释或解决洞察中发现的问题。
如果洞察与历史案例匹配，请明确提及。所有解释都**必须**是中文。
"""
    response = client.generate_content(prompt)
    return response.text or "No relevant knowledge found."
