from app.llm.prompts import build_rag_prompt


def test_build_rag_prompt_includes_strict_instruction_and_sources():
    prompt = build_rag_prompt(
        "What is the process?",
        [
            {
                "chunk_id": "policy_chunk_001",
                "source": "data/raw/policy.md",
                "page": None,
                "text": "Identify risks and assign owners.",
            }
        ],
    )

    assert "only source of truth" in prompt
    assert "I could not find this in the ingested documents." in prompt
    assert "policy_chunk_001" in prompt
    assert "Identify risks" in prompt
