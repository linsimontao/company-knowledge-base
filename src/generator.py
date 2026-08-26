"""LLM response generation engine with prompt grounding, anti-hallucination, and citation tracking."""

import os
from typing import List, Optional
from pydantic import BaseModel, Field
from src.config import RAGConfig, default_config
from src.retriever import RetrievedDoc

SYSTEM_PROMPT = """你是一位严谨、专注的企业内部知识库问答助手。
你的任务是严格依据提供的【企业参考文档片段】直接回答员工的问题。

【核心回答规范与红线】
1. 绝对忠实性（Zero Hallucination）：回答中的每一句话、数字、金额、天数、流程必须100%直接源自参考文档。严禁引入任何未经文档提及的外部常识、法律通识或个人推断。
2. 禁止客套寒暄（No Pleasantries）：不要在开头输出“您好”、“我是小助手”、“很高兴为您解答”等闲聊或问候语，直接开门见山列出事实结论。
3. 明确未知细节：如果问题中包含多个部分，而参考文档只回答了其中一部分，请仅回答有文档依据的部分，并明确说明“关于XXX，参考文档未做具体规定”，严禁根据经验脑补。
4. 结构化呈现：使用清晰的 Markdown 列表（1. 2. 3. / -）或小标题逐条罗列核心事实。
5. 来源出处标注：回答最后必须列出【参考出处】，注明参考的文档文件名及对应章节。"""


class GeneratedAnswer(BaseModel):
    """Encapsulates the final LLM response along with retrieved references."""

    query: str
    answer: str
    sources: List[str] = Field(default_factory=list)
    retrieved_docs: List[RetrievedDoc] = Field(default_factory=list)
    model_used: str = Field(default="unknown")


class LLMGenerator:
    """Handles prompt construction and LLM inference across OpenAI, Gemini, and Mock providers."""

    def __init__(self, config: RAGConfig = default_config):
        self.config = config

    def generate(self, query: str, retrieved_docs: List[RetrievedDoc]) -> GeneratedAnswer:
        """Generates an answer using the retrieved document context."""
        context_str = self._format_context(retrieved_docs)
        user_prompt = f"""【企业参考文档片段】
{context_str}

【员工提问】
{query}

请根据上述参考文档，给出严谨、准确且格式友好的解答："""

        provider = self.config.llm_provider.lower()
        answer_text = ""
        model_name = self.config.llm_model

        # Extract unique sources
        sources = list(dict.fromkeys([f"{doc.source} ({doc.section})" for doc in retrieved_docs]))

        # Try OpenAI-compatible API
        if provider == "openai":
            api_key = self.config.openai_api_key or os.getenv("OPENAI_API_KEY")
            if api_key:
                try:
                    from openai import OpenAI

                    client = OpenAI(
                        api_key=api_key,
                        base_url=self.config.openai_base_url or os.getenv("OPENAI_BASE_URL"),
                    )
                    resp = client.chat.completions.create(
                        model=self.config.llm_model,
                        messages=[
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        temperature=self.config.llm_temperature,
                    )
                    answer_text = resp.choices[0].message.content or ""
                    model_name = self.config.llm_model
                except Exception as e:
                    print(f"⚠️ OpenAI generation error: {e}. Falling back to Local Extractive Generator.")
                    answer_text = self._mock_generate(query, retrieved_docs)
                    model_name = "local_extractive_fallback"
            else:
                answer_text = self._mock_generate(query, retrieved_docs)
                model_name = "local_extractive_offline"

        # Try Gemini API
        elif provider == "gemini":
            api_key = self.config.gemini_api_key or os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    from google import genai

                    client = genai.Client(api_key=api_key)
                    resp = client.models.generate_content(
                        model=self.config.llm_model or "gemini-2.5-flash",
                        contents=f"{SYSTEM_PROMPT}\n\n{user_prompt}",
                    )
                    answer_text = resp.text or ""
                    model_name = self.config.llm_model or "gemini-2.5-flash"
                except Exception as e:
                    print(f"⚠️ Gemini generation error: {e}. Falling back to Local Extractive Generator.")
                    answer_text = self._mock_generate(query, retrieved_docs)
                    model_name = "local_extractive_fallback"
            else:
                answer_text = self._mock_generate(query, retrieved_docs)
                model_name = "local_extractive_offline"

        else:
            answer_text = self._mock_generate(query, retrieved_docs)
            model_name = "local_extractive_mock"

        return GeneratedAnswer(
            query=query,
            answer=answer_text.strip(),
            sources=sources,
            retrieved_docs=retrieved_docs,
            model_used=model_name,
        )

    def _format_context(self, retrieved_docs: List[RetrievedDoc]) -> str:
        """Formats retrieved chunks into clean numbered context blocks."""
        if not retrieved_docs:
            return "（未检索到任何相关制度文档）"

        blocks = []
        for idx, doc in enumerate(retrieved_docs, 1):
            blocks.append(
                f"--- 片段 {idx} [来源: {doc.source} | 章节: {doc.header_path}] ---\n{doc.text}\n"
            )
        return "\n".join(blocks)

    def _mock_generate(self, query: str, docs: List[RetrievedDoc]) -> str:
        """Offline high-fidelity rule & extraction-based fallback generator."""
        if not docs:
            return (
                "抱歉，根据公司现有知识库制度，未找到关于该问题的明确规定。\n\n"
                "建议直接咨询您的 HRBP、财务接口人或 IT 服务台（分机 8008）。"
            )

        # Highlight most relevant chunk text
        top_doc = docs[0]
        summary_lines = []
        for doc in docs[:2]:
            lines = [l.strip() for l in doc.text.splitlines() if l.strip() and not l.startswith("#")]
            summary_lines.extend(lines[:6])

        extracted_body = "\n".join(summary_lines[:10])
        source_tags = "\n".join([f"- 📄 `{doc.source}` ({doc.header_path})" for doc in docs[:3]])

        return f"""根据公司内部制度规定，为您查询到以下相关信息：

{extracted_body}

---
**【参考出处】**：
{source_tags}"""

