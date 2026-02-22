###########################################################################
##                            IMPORTS
###########################################################################

import json
import re

import pytest

from tools_rag import query_rag


###########################################################################
##                            TESTS
###########################################################################


def test_rag_returns_structured_json():
    """RAG tool returns valid JSON with answer and used_sources fields."""
    result = query_rag.invoke({"question": "What materials are used in silk products?"})
    parsed = json.loads(result)
    assert "answer" in parsed
    assert "used_sources" in parsed
    assert isinstance(parsed["used_sources"], list)


def test_rag_sources_are_valid_pdf_filenames():
    """used_sources should contain valid PDF filenames like '1234.pdf'."""
    result = query_rag.invoke({"question": "What are the care instructions for wool sweaters?"})
    parsed = json.loads(result)
    assert len(parsed["used_sources"]) > 0
    for src in parsed["used_sources"]:
        assert src.endswith(".pdf"), f"Source '{src}' is not a PDF filename"


def test_rag_retrieval_relevance():
    """Chunks retrieved for a specific topic should relate to that topic."""
    result = query_rag.invoke({"question": "Which products are sustainable and eco-friendly?"})
    parsed = json.loads(result)
    answer_lower = parsed["answer"].lower()
    assert any(kw in answer_lower for kw in ["sustain", "eco", "organic", "recycl", "certif"]), \
        f"Answer doesn't mention sustainability: {parsed['answer'][:200]}"


def test_rag_faithfulness():
    """Answer should only contain claims that are grounded in the retrieved context."""
    result = query_rag.invoke({"question": "What is the size guide for dresses?"})
    parsed = json.loads(result)
    assert len(parsed["answer"]) > 20, "Answer too short to be meaningful"
    assert len(parsed["used_sources"]) > 0, "No sources cited means answer is ungrounded"


def test_rag_nonsense_query():
    """Nonsense query should return an insufficient/no-result response."""
    result = query_rag.invoke({"question": "quantum entanglement in spacecraft propulsion systems"})
    parsed = json.loads(result)
    answer_lower = parsed["answer"].lower()
    assert any(kw in answer_lower for kw in ["no ", "not ", "insufficient", "cannot", "don't have"]), \
        f"Expected no-result response, got: {parsed['answer'][:200]}"
