import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR: Path = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR: Path = BASE_DIR / "data"
SAMPLE_PUBLIC_DIR: Path = DATA_DIR / "sample_public"
RAW_PRIVATE_DIR: Path = DATA_DIR / "raw_private"

_default_chroma_dir = str(DATA_DIR / "chroma_db")
CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", _default_chroma_dir).strip() or _default_chroma_dir

# Model and retrieval settings
LLM_MODEL: str = os.getenv("GEMINI_LLM_MODEL", "gemini-3.5-flash").strip() or "gemini-3.5-flash"
LLM_TEMPERATURE: float = float(os.getenv("GEMINI_LLM_TEMPERATURE", "0.0"))
EMBEDDING_MODEL: str = (
    os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001").strip()
    or "models/gemini-embedding-001"
)

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
RETRIEVER_K: int = int(os.getenv("RETRIEVER_K", "4"))

# API key
PLACEHOLDER_KEYS: set[str] = {"", "your_gemini_api_key_here", "sua_chave_aqui"}
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()

# Sync with GOOGLE_API_KEY for langchain-google-genai
if GEMINI_API_KEY and GEMINI_API_KEY not in PLACEHOLDER_KEYS:
    os.environ.setdefault("GOOGLE_API_KEY", GEMINI_API_KEY)


def get_gemini_api_key() -> str:
    """Return the Gemini API key or raise an error if it's missing."""
    key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    if key in PLACEHOLDER_KEYS:
        raise ValueError(
            "GEMINI_API_KEY is not configured. Please set your API key in the .env file."
        )
    os.environ["GOOGLE_API_KEY"] = key
    return key


def validate_config() -> None:
    """Check required credentials and parameters."""
    get_gemini_api_key()
    if CHUNK_SIZE <= 0:
        raise ValueError("CHUNK_SIZE must be greater than 0.")
    if CHUNK_OVERLAP < 0 or CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError("CHUNK_OVERLAP must be non-negative and smaller than CHUNK_SIZE.")
    if RETRIEVER_K <= 0:
        raise ValueError("RETRIEVER_K must be greater than 0.")
