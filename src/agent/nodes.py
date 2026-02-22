###########################################################################
##                            IMPORTS
###########################################################################

import json
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

###########################################################################
##                        CUSTOM IMPORTS
###########################################################################

from agent.state import AgentState, RouteDecision, ReflectDecision
from agent.shared import _get_llm, TOOLS, SYSTEM_PROMPT
from agent.prompts import PLAN_PROMPT, REFLECT_PROMPT, SYNTHESIZE_PROMPT


###########################################################################
##                          GRAPH NODES
###########################################################################


def router(state: AgentState) -> dict:
    """Classify whether the user question needs tools or is simple chat."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(RouteDecision)

    last_human = next(m for m in reversed(state["messages"]) if isinstance(m, HumanMessage))

    result = structured_llm.invoke([
        SystemMessage(content="Classify this user message. Use 'needs_tools' if it asks about sales data, products, revenue, customers, stores, materials, sustainability, or any analytical/product question. Use 'direct_response' for greetings, general chat, or non-data questions."),
        last_human,
    ])

    return {"reflection": result.intent}


def plan(state: AgentState) -> dict:
    """Generate tool calls based on the question, collected results, and reflection feedback."""
    llm = _get_llm()
    llm_with_tools = llm.bind_tools(TOOLS)

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    question = last_human.content if last_human else ""

    iteration = state.get("iteration", 0)
    collected = state.get("collected_results", [])
    reflection = state.get("reflection", "")
    todo = state.get("todo", [])

    # Build context for the planner
    context_parts = [SYSTEM_PROMPT, PLAN_PROMPT]
    context_parts.append(f"\nUser question: {question}")
    context_parts.append(f"\nIteration: {iteration + 1} / {state.get('max_iterations', 2)}")

    if todo:
        context_parts.append(f"\nCurrent TODO list:\n" + "\n".join(f"- {t}" for t in todo))

    if collected:
        context_parts.append(f"\nData collected so far:\n" + "\n---\n".join(collected))

    if reflection and iteration > 0:
        context_parts.append(f"\nReflection from previous iteration:\n{reflection}")

    response = llm_with_tools.invoke([
        SystemMessage(content="\n".join(context_parts)),
        HumanMessage(content=question),
    ])

    return {
        "messages": [response],
        "iteration": iteration + 1,
    }


def collect_results(state: AgentState) -> dict:
    """Extract tool results from the latest messages and append to collected_results."""
    collected = list(state.get("collected_results", []))
    rag_sources = list(state.get("rag_sources", []))

    for msg in reversed(state["messages"]):
        if hasattr(msg, "type") and msg.type == "tool":
            # Extract RAG sources from structured JSON output
            if msg.name == "query_rag":
                try:
                    parsed = json.loads(msg.content)
                    collected.append(f"[{msg.name}] {parsed['answer']}")
                    for src in parsed.get("used_sources", []):
                        rag_sources.append({"source": Path(str(src)).name, "tool": "query_rag"})
                except (json.JSONDecodeError, KeyError):
                    collected.append(f"[{msg.name}] {msg.content}")
            else:
                collected.append(f"[{msg.name}] {msg.content}")
        elif hasattr(msg, "type") and msg.type == "ai":
            break

    return {"collected_results": collected, "rag_sources": rag_sources}


def reflect(state: AgentState) -> dict:
    """Evaluate if collected data is sufficient or if more queries are needed."""
    llm = _get_llm()
    structured_llm = llm.with_structured_output(ReflectDecision)

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    question = last_human.content if last_human else ""
    collected = state.get("collected_results", [])
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    result = structured_llm.invoke([
        SystemMessage(content=REFLECT_PROMPT.format(
            question=question,
            iteration=iteration,
            max_iterations=max_iter,
            collected_data="\n---\n".join(collected) if collected else "(none yet)",
        )),
        HumanMessage(content="Evaluate the collected data and decide if more queries are needed."),
    ])

    return {
        "reflection": result.feedback,
        "todo": result.updated_todo,
        "reflection_satisfied": result.satisfied,
    }


def synthesize(state: AgentState) -> dict:
    """Produce the final comprehensive answer from all collected results."""
    llm = _get_llm()

    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    question = last_human.content if last_human else ""
    collected = state.get("collected_results", [])

    response = llm.invoke([
        SystemMessage(content=SYNTHESIZE_PROMPT.format(
            system_prompt=SYSTEM_PROMPT,
            collected_data="\n---\n".join(collected) if collected else "(no data)",
        )),
        HumanMessage(content=question),
    ])

    content = response.content
    if isinstance(content, list):
        content = " ".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()

    # Append RAG sources section if any were used
    rag_sources = state.get("rag_sources", [])
    if rag_sources:
        unique_sources = list(dict.fromkeys(s["source"] for s in rag_sources))
        sources_section = "\n\n---\n**Sources (Product Technical Sheets):**\n"
        sources_section += "\n".join(f"- {src}" for src in unique_sources)
        content += sources_section

    return {"messages": [AIMessage(content=content)]}


def respond(state: AgentState) -> dict:
    """Direct chat response without tools."""
    llm = _get_llm()

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        *[m for m in state["messages"] if isinstance(m, (HumanMessage, AIMessage))],
    ])

    content = response.content
    if isinstance(content, list):
        content = " ".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()

    return {"messages": [AIMessage(content=content)]}


###########################################################################
##                       CONDITIONAL EDGES
###########################################################################


def route_after_router(state: AgentState) -> str:
    if state.get("reflection") == "direct_response":
        return "respond"
    return "plan"


def route_after_plan(state: AgentState) -> str:
    """Check if the plan node emitted tool calls."""
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "execute"
    return "synthesize"


def route_after_reflect(state: AgentState) -> str:
    satisfied = state.get("reflection_satisfied", True)
    iteration = state.get("iteration", 0)
    max_iter = state.get("max_iterations", 2)

    if satisfied or iteration >= max_iter:
        return "synthesize"
    return "plan"
