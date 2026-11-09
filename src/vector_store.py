import shutil
from pathlib import Path
import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL, get_gemini_api_key

COLLECTION_NAME = "institutional_docs"


def get_embeddings(model_name: str = EMBEDDING_MODEL) -> GoogleGenerativeAIEmbeddings:
    """Initialize the Google Gemini embeddings model."""
    api_key = get_gemini_api_key()
    return GoogleGenerativeAIEmbeddings(model=model_name, google_api_key=api_key)


def clear_vector_store(persist_directory: str | Path = CHROMA_PERSIST_DIR) -> None:
    """Clear collection or remove the persisted ChromaDB directory."""
    dir_path = Path(persist_directory)
    if not dir_path.exists():
        return

    try:
        client = chromadb.PersistentClient(path=str(dir_path))
        existing_collections = [c.name for c in client.list_collections()]
        if COLLECTION_NAME in existing_collections:
            client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    shutil.rmtree(dir_path, ignore_errors=True)


def create_vector_store(
    documents: list[Document],
    persist_directory: str | Path = CHROMA_PERSIST_DIR,
    embedding_function: Embeddings | None = None,
    reset: bool = True,
) -> Chroma:
    """Index document chunks into a local ChromaDB vector store."""
    if not documents:
        raise ValueError("Cannot create vector store from an empty document list.")

    persist_path = Path(persist_directory)
    embeddings = embedding_function or get_embeddings()

    if reset and persist_path.exists():
        clear_vector_store(persist_path)

    persist_path.mkdir(parents=True, exist_ok=True)

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=COLLECTION_NAME,
    )


def load_vector_store(
    persist_directory: str | Path = CHROMA_PERSIST_DIR,
    embedding_function: Embeddings | None = None,
) -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    persist_path = Path(persist_directory)
    sqlite_file = persist_path / "chroma.sqlite3"
    if not sqlite_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at '{persist_path}'. Run ingestion and indexing first."
        )

    embeddings = embedding_function or get_embeddings()
    return Chroma(
        persist_directory=str(persist_path),
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )
