"""One-shot PDF ingestion: load all product tech sheet PDFs into sqlite-vec.

Run:  uv run python src/ingest.py
Idempotent: deletes and recreates rag.db on every run.
"""

###########################################################################
##                            IMPORTS
###########################################################################

from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import SQLiteVec
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rich.console import Console

load_dotenv()


###########################################################################
##                           CONSTANTS
###########################################################################

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_DIR = PROJECT_ROOT / "data" / "pdf"
RAG_DB_FILE = str(PROJECT_ROOT / "db" / "rag.db")
RAG_TABLE = "product_knowledge"

console = Console()


###########################################################################
##                           FUNCTIONS
###########################################################################


def ingest_pdfs() -> None:
    Path(RAG_DB_FILE).parent.mkdir(parents=True, exist_ok=True)
    pdf_files = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    if not pdf_files:
        console.print("[yellow]No PDF files found in data/pdf/ -- nothing to ingest.[/]")
        return

    console.print(f"Found [cyan]{len(pdf_files)}[/] PDFs in {PDF_DIR}")

    ######################### Step 1: Clean old DB #########################
    db_path = Path(RAG_DB_FILE)
    if db_path.exists():
        db_path.unlink()
        console.print("[dim]Deleted old rag.db[/]")

    ######################### Step 2: Create vector store ##################
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    vector_store = SQLiteVec(table=RAG_TABLE, connection=None, embedding=embeddings, db_file=RAG_DB_FILE)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    total_chunks = 0

    ######################### Step 3: Load and chunk PDFs ##################
    for pdf_path in pdf_files:
        pages = PyPDFLoader(str(pdf_path)).load()
        chunks = splitter.split_documents(pages)
        product_id = int(pdf_path.stem) if pdf_path.stem.isdigit() else None

        for i, chunk in enumerate(chunks):
            chunk.metadata["source_path"] = str(pdf_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
            chunk.metadata["product_id"] = product_id
            chunk.metadata["chunk_index"] = i

        if chunks:
            vector_store.add_documents(chunks)
        total_chunks += len(chunks)
        console.print(f"  [dim]{pdf_path.name}[/] -> {len(chunks)} chunks")

    console.print(f"\n[green]Done:[/] {len(pdf_files)} PDFs, {total_chunks} chunks -> {Path(RAG_DB_FILE).name}")


###########################################################################
##                              MAIN
###########################################################################

if __name__ == "__main__":
    ingest_pdfs()
