###########################################################################
##                            IMPORTS
###########################################################################

import sqlite3

import pytest
from langchain_core.messages import HumanMessage

from langgraph.checkpoint.sqlite import SqliteSaver

###########################################################################
##                        CUSTOM IMPORTS
###########################################################################

from agent import workflow
from agent.state import RouteDecision
from agent.shared import _get_llm


###########################################################################
##                       ROUTING TESTS
###########################################################################


@pytest.fixture
def structured_router():
    llm = _get_llm()
    return llm.with_structured_output(RouteDecision)


def test_route_sql_question(structured_router):
    result = structured_router.invoke([
        {"role": "system", "content": "Classify this user message. Use 'needs_tools' for data/analytical questions, 'direct_response' for simple chat."},
        {"role": "user", "content": "What were the top selling products in Germany?"},
    ])
    assert result.intent == "needs_tools"


def test_route_rag_question(structured_router):
    result = structured_router.invoke([
        {"role": "system", "content": "Classify this user message. Use 'needs_tools' for data/analytical questions or product knowledge questions, 'direct_response' for simple chat."},
        {"role": "user", "content": "What materials is product 7021 made of?"},
    ])
    assert result.intent == "needs_tools"


def test_route_greeting(structured_router):
    result = structured_router.invoke([
        {"role": "system", "content": "Classify this user message. Use 'needs_tools' for data/analytical questions, 'direct_response' for simple chat."},
        {"role": "user", "content": "Hello, how are you?"},
    ])
    assert result.intent == "direct_response"


def test_route_hybrid_question(structured_router):
    result = structured_router.invoke([
        {"role": "system", "content": "Classify this user message. Use 'needs_tools' for data/analytical questions or product knowledge questions, 'direct_response' for simple chat."},
        {"role": "user", "content": "What are our best selling sustainable products in the UK?"},
    ])
    assert result.intent == "needs_tools"


###########################################################################
##                   FULL PIPELINE TEST (HYBRID)
###########################################################################


@pytest.mark.flaky(reruns=2)
def test_full_hybrid_pipeline():
    """End-to-end: SQL + RAG query produces an answer with RAG sources."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = workflow.compile(checkpointer=checkpointer)

    config = {"configurable": {"thread_id": "test-hybrid"}}
    invoke_state = {
        "depth": 3,
        "max_iterations": 6,
        "messages": [{"role": "user", "content": "In Germany what was the most sold product in 2024 and what does its style card say?"}],
        "iteration": 0,
        "collected_results": [],
        "todo": [],
        "reflection": "",
        "rag_sources": [],
    }

    result = graph.invoke(invoke_state, config)
    conn.close()

    # Final message should be from the AI
    final_msg = result["messages"][-1]
    assert final_msg.type == "ai"
    answer = final_msg.content

    # Answer should mention Germany
    assert "germany" in answer.lower(), f"Answer doesn't mention Germany: {answer[:200]}"

    # RAG sources should be populated (the agent needed RAG for style card)
    rag_sources = result.get("rag_sources", [])
    assert len(rag_sources) > 0, "No RAG sources tracked -- agent should have used RAG for style card info"

    # Sources section should appear in the final answer
    assert "sources" in answer.lower(), f"Final answer missing Sources section: {answer[-300:]}"
