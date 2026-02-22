# Hybrid Sales Assistant

A hybrid data retrieval system for a multinational fashion retail company, with sales data for 2024. The agent can answer questions by querying both **structured sales data** (SQL database) and **unstructured product documentation** (PDF technical sheets), and synthesize answers from both sources when needed.

The agent figures out which source to use based on the question, or uses both if the question requires it.

**Data sources:**
- `db/sales.db` - SQLite sales database with transactions
- `data/pdf/` - 200 product technical sheet PDFs (materials, care instructions etc)
- `db/rag.db` - vector embeddings of the product PDFs

---

## Quick Start

**Requirements:** Python 3.11+, UV package manager

1. Create a `.env` file in the project root with your Google API key:

```
GOOGLE_API_KEY=your_api_key_here
```

2. Install dependencies and run:

```bash
uv sync
uv run python src/gui.py
```

This opens the Textual TUI where you can select a research depth (1-3), start or resume a conversation, and watch the agent's tool calls and reasoning live as it works.

---

## Example Runs

**Simple query (Depth 1):** *"What are our top selling products?"*

```
Router:  data question -> plan
Plan:    query_sql (top 10 products by revenue with category)
Execute: SELECT p.description_en, p.category, SUM(t.line_total) ...
Reflect: sufficient data, proceed to synthesize
Synthesize: final answer with top 10 products ranked by revenue
```

**Hybrid query (Depth 3):** *"What's the material of the 5 most sold products in the USA in 2024?"*

```
Router:  data question -> plan
Plan:    query_sql (top 5 products by units sold in USA)
Execute: SELECT p.product_id, p.description_en, SUM(t.quantity) ...
Reflect: have sales ranking, but need material info -> continue
Plan:    query_rag ("what material is product 1234 made of") x5
Execute: RAG retrieves chunks from product PDFs, synthesizes material info
Reflect: have both sales ranking + materials for all 5 -> satisfied
Synthesize: final answer listing top 5 USA products with their materials, sources listed
```

---

## How It Works

The agent is a custom **LangGraph StateGraph** with a plan-execute-reflect loop. Instead of a simple ReAct agent that tries to answer in one shot, this agent:

1. **Routes** the question: simple chat or data question
2. **Plans** which tools to call (SQL, RAG, or both)
3. **Executes** the tool calls
4. **Reflects** on the results, decides if more data is needed
5. **Loops back** to plan if not satisfied
6. **Synthesizes** a final answer from all collected data

The research depth is configurable (1-3). Depth 1 gives quick answers, depth 3 does deep multi-step analysis across both data sources. The reflection node can also stop early if the answer was found before the max iterations.

## Tools

- **query_sql**: read-only SELECT queries on the sales database (200-row cap)
- **query_rag**: 2-step RAG, retrieves relevant chunks from product PDFs and synthesizes an answer with source tracking

## Project Structure

```
src/
  main.py          # Rich CLI entry point
  gui.py           # Textual TUI entry point
  models.py        # SQLModel ORM definitions
  tools_sql.py     # SQL query tool
  tools_rag.py     # RAG retrieval + synthesis tool
  ingest.py        # One-shot PDF ingestion into sqlite-vec
  agent/
    graph.py       # LangGraph StateGraph wiring
    nodes.py       # Graph nodes (router, plan, execute, reflect, synthesize)
    state.py       # AgentState TypedDict + Pydantic decision models
    shared.py      # LLM setup, tools, constants
    prompts.py     # System prompt builder, golden bucket loader
db/
  sales.db         # Sales data (products, transactions, customers, stores)
  rag.db           # Vector embeddings of product PDFs (sqlite-vec)
  application.db   # Conversation history (SqliteSaver checkpoints)
data/pdf/
  {product_id}.pdf # 200 product technical sheet PDFs
golden_bucket/
  golden_bucket.json  # 20 few-shot SQL query examples
```

## Debugging and Tracing

The agent graph is compatible with LangGraph Studio for visual debugging:

```bash
uv run langgraph dev
```

All runs are traced with **LangSmith** for observability, you can inspect every node execution, tool call, and LLM response in the LangSmith dashboard.

## Environment Variables

The `.env` file needs a `GOOGLE_API_KEY` (used for Gemini 2.5 Flash LLM and gemini-embedding-001 embeddings). LangSmith tracing keys are also expected if you want tracing enabled.

## Testing

```bash
uv run pytest tests/ -v
```

17 tests across 3 files: SQL tool validation, RAG retrieval quality, and agent workflow (routing, hybrid queries).

## More Details

See [README_Architectural_Decisions.md](README_Architectural_Decisions.md) for the full architectural decision record, why I chose each technology, trade-offs, and what I'd change for production.
