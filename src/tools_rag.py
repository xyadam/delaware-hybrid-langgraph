###########################################################################
##                            IMPORTS
###########################################################################

import json
from pathlib import Path

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_community.vectorstores import SQLiteVec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import BaseModel, Field


###########################################################################
##                           CONSTANTS
###########################################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAG_DB_FILE = str(PROJECT_ROOT / "db" / "rag.db")
RAG_TABLE = "product_knowledge"


###########################################################################
##                      STRUCTURED OUTPUT MODEL
###########################################################################


class RAGResponse(BaseModel):
    answer: str = Field(description="The answer to the question, based only on the provided context.")
    used_sources: list[str] = Field(description="List of source filenames (e.g. '7021.pdf') that you actually used to form your answer. Only include sources you directly referenced.")


###########################################################################
##                           RAG TOOL
###########################################################################


@tool
def query_rag(question: str) -> str:
    """Search product technical sheet PDFs for product knowledge (materials, care instructions, sizing, sustainability, style notes).
    Use this tool for questions about what products are made of, how to care for them, size guides, eco certifications, and outfit pairing suggestions.
    Do NOT use this for sales numbers, revenue, or customer data -- use query_sql instead.
    IMPORTANT: Always use product names/descriptions in your question, NOT product IDs. Semantic search matches on text similarity, so 'silk retro coat' will find results but 'product 7021' will not."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = SQLiteVec(table=RAG_TABLE, connection=None, embedding=embeddings, db_file=RAG_DB_FILE)

    docs = vector_store.similarity_search(question, k=5)
    if not docs:
        return "No relevant product technical sheet context found for this question."

    context_blocks = []
    for doc in docs:
        meta = doc.metadata or {}
        source = Path(str(meta.get("source_path", "unknown"))).name
        context_blocks.append(f"Source: {source}\n{doc.page_content}")
    context_text = "\n\n---\n\n".join(context_blocks)

    llm = init_chat_model("google_genai:gemini-2.5-flash", temperature=0.7)
    structured_llm = llm.with_structured_output(RAGResponse)

    rag_response = structured_llm.invoke(
        "You are a product knowledge assistant. Answer only from the provided context. "
        "If context is insufficient, say that clearly. "
        "In used_sources, list ONLY the source filenames you actually based your answer on. "
        "NEVER include folder paths, only basename values like '235664.pdf'.\n\n"
        f"Question: {question}\n\nContext:\n{context_text}"
    )

    # Encode sources into the tool output so collect_results can extract them
    result = {
        "answer": rag_response.answer,
        "used_sources": rag_response.used_sources,
    }
    return json.dumps(result)
