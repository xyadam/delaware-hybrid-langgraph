# Architectural Decision Record

## Tech Stack:

- **Approach:** Hybrid RAG + Agentic (Option C)
- **Framework:** LangGraph (custom StateGraph)
- **LLM:** Gemini 2.5 Flash
- **Embedding:** gemini-embedding-001 (768 dims, multilingual)
- **Databases:** SQLite (3 separate files)
  - `sales.db` - sales data (products, transactions, customers, stores)
  - `rag.db` - sqlite-vec embeddings of product technical sheet PDFs
  - `application.db` - conversation history (SqliteSaver checkpoints)
- **RAG source:** 200 product technical sheet PDFs
- **ORM:** SQLModel
- **Checkpointer:** SqliteSaver
- **Few-shot:** 20 SQL query patterns (golden bucket JSON)
- **UI:** Textual TUI / Rich CLI
- **Package manager:** UV



---

## Dataset

The project uses the [Global Fashion Retail Sales](https://www.kaggle.com/datasets/ricgomes/global-fashion-retail-stores-dataset) open-source dataset from Kaggle: a simulated 2-year sales dataset for a multinational fashion brand with stores across 7 countries (US, UK, Germany, France, Spain, Portugal, China).

For portability, I trimmed it down to **2024 only** and sampled **200 products** (67K transactions, 61K customers, 35 stores). This keeps the database at 15 MB so it can be committed to Git and the evaluator can clone, explore the schema, and run the app without any setup, which is a very important aspect.

The CSV files were converted into a SQLite database (`db/sales.db`) using a simple ingestion script. For a real-life application I'd use a proper cloud database instance (Postgres or MS SQL) hosted in the cloud (Azure, AWS, etc.).

---

## Approach: Hybrid RAG + Agentic (Option C)

I went with the hybrid approach: a research-centric LangGraph agent that decides with each query whether to search the vector knowledge base (RAG), query structured sales data (SQL), or just respond directly. The reason is simple: sales numbers live in SQLite, product knowledge lives in PDFs, and the agent figures out which source to use (or both) based on what the user asks.

---

## Why LangGraph Instead of a Simple ReAct Agent

Originally I started planning the project with LangChain's **create_agent** because it's the fastest way to get a working agent. The problem was that the agent kept trying to answer from a single query, it wouldn't dig deeper into the data on its own. I tried teaching it with few-shot examples (golden bucket), and that helped, but it still wasn't consistent.

This was the reason I decided to make a switch to a custom **LangGraph architecture** with an explicit plan-execute-reflect loop. Now the agent plans what tools to call, executes them, looks at what it got, and decides if it needs more data. The depth is configurable (1-3), which makes the research thoroughness predictable instead of hoping the LLM feels like exploring more.

This also pairs well with Gemini 2.5 Flash: a great model, fast and cheap, but it tends to under-explore when left on its own. The graph structure forces it through the reflection cycles regardless, so we get the best of both: Gemini's speed and cost efficiency + guaranteed deep exploration.

---

## Configurable Research Depth

I wanted the chat experience to feel similar to how people use ChatGPT. Sometimes you just want a quick answer, sometimes you need the model to really dig in. So I made depth configurable:

- **Depth 1** (2 iterations): Quick answers, simple lookups
- **Depth 2** (4 iterations): Moderate analysis, some cross-referencing
- **Depth 3** (6 iterations): Deep multi-step research across SQL and RAG

The user picks the depth at startup, and it applies to the whole session. The **reflection-node** also prevents the graph running for unnecessarily long if the answer was found earlier, but a deeper reasoning was selected.

---

## Why Google AI (Gemini 2.5 Flash + gemini-embedding-001)

My choice for Google was driven by the embedding model. `gemini-embedding-001` is a really strong multilingual embedder, cheap, fast, and performs well across 100+ languages. This matters here because the retail dataset covers 7 countries (US, UK, Germany, France, Spain, Portugal, China), so users might ask questions in any of these languages. Many popular embedding models have hard time with multilingual performance, but Gemini embedding handles it well.

I didn't want to use local embedding models for this assignment, they can be messy to evaluate (GPU dependencies, platform issues). A hosted API was the cleaner choice, and that's generally what I prefer in my projects as well.
Also, since I picked Gemini for embeddings, staying in the Google ecosystem for the LLM made sense: one API key, one provider, simple setup.

(Side note: In a real production app, the embedding pipeline and LLM could be swapped independently. But for this assessment, the all-Google stack was the most practical)

Also worth noting: Gemini 2.5 Flash can be hosted on Google Vertex AI with full EU data residency, making it GDPR compliant, which matters for enterprise deployments.

---

## SQLite + SQLModel

I chose SQLite because it's the simplest way to iterate locally. No server, no Docker, no connection config, just a file. Since the dataset comes from local CSV files (Kaggle), SQLite makes the project easy to deploy for evaluation: clone the repo and run.

I used SQLModel as the ORM so the schema can be switched to PostgreSQL by changing a single connection string. In production, I'd use Postgres for LangGraph checkpointing and RAG storage (pgvector), and the main sales database could be anything, Postgres, MS SQL, whatever the client uses, SQLModel can handle it without needing to touch the applications code.

---

## Golden Bucket (Example material for AI)

I created a local JSON file with 20 example query workflows that get loaded into the system prompt. These teach the agent how to write proper SQL for this specific schema: return top 5-10 results instead of just 1, use JOINs to include context columns, compare across countries and categories, evaluate questions with multiple aspects, etc.

Right now all 20 examples are loaded every time, which works fine for a demo. In production, I'd store them in a Postgres database and retrieve them with similarity search, and only inject the relevant ones. That way the example library can grow to hundreds of patterns without bloating the context.

---

## Chunking Strategy

I used LangChain's `RecursiveCharacterTextSplitter` with 1000 character chunks and 150 character overlap. The product PDFs are short (1-2 pages each), so 1000 chars keeps each chunk focused on a single topic (materials, care instructions, sizing, etc.) without splitting mid-sentence.

The 150 overlap makes sure we don't lose context at chunk boundaries. This is a good sweetspot for not losing details while still keeping the chunks focused enough to be relevant during retrieval.

---

## RAG: 2-Step with Source Tracking

The RAG tool does two steps: (1) retrieve top-5 chunks from sqlite-vec, (2) synthesize an answer with Gemini using structured output. The structured output returns both the answer and which PDF sources the LLM actually used: not just which chunks were retrieved, but which ones it based its answer on. This is more accurate for faithfulness evaluation.

It's important to mention, that the rag tool already has an LLM built in, which can synthesize the answers back to the main orchestrator agent, so we are not bloating it's context with a the unnecessary chunks, that was not used for the answer.

Sources accumulate across multiple RAG calls within a single turn (the agent might call RAG several times during the reflect loop), and the final answer includes a "Sources" section listing all referenced PDFs.

---

## Structured Output for Routing and Reflection

The router and reflect nodes use Pydantic models with `with_structured_output()` and `RouteDecision` for classifying intent (chat vs. data question) and `ReflectDecision` for deciding if the agent has enough data. No free-text parsing, no ambiguity.

The reflect node also outputs an updated TODO list that the plan node picks up on the next iteration, so feedback turns into concrete next steps.

---

## SQL Tool: Read-Only with 200-Row Cap

The SQL tool only allows SELECT queries (enforced at runtime) and caps results at 200 rows. This keeps the sales database safe from accidental writes and prevents unbounded results from accidentally blowing up the token budget.

When a query fails, the error comes back as a normal tool output, the agent sees it (we tell the model than only the first 200 items is shown out of the X results), and can fix the query on the next iteration. No try-catch wrapping, no fallbacks.

---

## UI: Textual TUI

I built a Textual terminal interface where you can follow exactly what the agent is doing, every tool call, every SQL query, every reflection step is visible live as it happens. It also has clickable session history and a depth selector. It satisfies the "trivial UI" requirement while still giving good visibility into the agent's reasoning.

---

## Trade-offs

| What I optimized for | What I traded off |
|---|---|
| **Local simplicity** (SQLite, single API key) | No cloud-native features |
| **Single model** (Gemini 2.5 Flash for everything) | Locked into one provider |
| **Deterministic depth** (configurable iterations) | More API calls at higher depths |
| **Real Kaggle data** (realistic schema) | Dataset preprocessing was needed (filtering, sampling) |
| **Terminal UI** (Textual) | No browser-based access |

---

## Measuring Success in Production

**RAG quality**: Faithfulness, relevance, and context precision, measured with automated tests and LangSmith evaluations as real users interact with the system
**Tool selection accuracy**: Check if the agent picking the right tool (SQL, RAG, or both) for each question type
**Response latency and cost per query**: LangSmith tracks token usage and cost per run, so we can evaluate the typical price of different question types and depth levels
**Golden bucket coverage**: If users are asking questions that are far off from the Golden Bucket, we can extend the Golden Bucket with new, properly crafted examples.

---

## What I'd Improve With More Time

- **Production database**: Postgres for LangGraph state, RAG vectors (pgvector), and golden bucket storage with similarity search
- **Cloud dataset deployment:** Application, Vector, Document data should be loaded from cloud provider (Blob Storage, Azure MS SQL)
- **Evaluation pipeline**: Automated faithfulness and relevance scoring on a test set
- **Containerization**: Docker Compose for one-command deployment
- **More golden bucket examples**: Especially for RAG and hybrid queries
- I would consider adding user profiling with LangMem to track the typical user preferences (table formats are preferred instead bullet points etc)
