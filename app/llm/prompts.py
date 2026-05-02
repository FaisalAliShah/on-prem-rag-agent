from typing import Any


RAG_PROMPT_TEMPLATE = """You are an on-prem document question-answering assistant.
Use the context below as the only source of truth.
If the context contains the answer, answer directly and briefly.
If the question asks for details about a person, organization, project, or topic, summarize the concrete details found in the context.
If the question asks for an email, URL, phone number, or contact detail, extract the exact value from the context.
If the context does not contain the answer, say exactly: "I could not find this in the ingested documents."
Never include that sentence when you have already provided details from the context.
Do not use outside knowledge.

<context>
{context}
</context>

<question>
{question}
</question>

Answer:
"""


def format_context(chunks: list[dict[str, Any]]) -> str:
    parts = []
    for index, chunk in enumerate(chunks, start=1):
        source = chunk.get("source", "unknown")
        page = chunk.get("page")
        page_label = f", page {page}" if page else ""
        parts.append(
            f"[{index}] Source: {source}{page_label}; Chunk: {chunk['chunk_id']}\n{chunk['text']}"
        )
    return "\n\n".join(parts)


def build_rag_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    return RAG_PROMPT_TEMPLATE.format(question=question, context=format_context(chunks))


def clean_generated_answer(answer: str) -> str:
    fallback = "I could not find this in the ingested documents."
    cleaned = answer.replace(fallback, "").strip()
    return cleaned or answer.strip()
