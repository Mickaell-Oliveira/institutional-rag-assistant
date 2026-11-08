import re
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import CHUNK_OVERLAP, CHUNK_SIZE, SAMPLE_PUBLIC_DIR


def clean_text(text: str) -> str:
    """Clean formatting noise while keeping paragraph breaks."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdfs_from_directory(directory: Path | str = SAMPLE_PUBLIC_DIR) -> list[Document]:
    """Load all PDF files from a directory and normalize source and page metadata."""
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    pdf_files = sorted(dir_path.glob("*.pdf"))
    documents: list[Document] = []

    for pdf_path in pdf_files:
        loader = PyMuPDFLoader(str(pdf_path))
        pages = loader.load()

        for doc in pages:
            cleaned = clean_text(doc.page_content)
            if not cleaned:
                continue

            doc.page_content = cleaned
            doc.metadata["source"] = pdf_path.name
            doc.metadata["page"] = int(doc.metadata.get("page", 0)) + 1
            documents.append(doc)

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest_documents(
    directory: Path | str = SAMPLE_PUBLIC_DIR,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Load PDFs from a directory and return processed text chunks."""
    pages = load_pdfs_from_directory(directory)
    return split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
