# Institutional Governance RAG Assistant

A Retrieval-Augmented Generation (RAG) system designed for querying institutional governance documents (bylaws, minutes, manuals) with strict source traceability, hallucination prevention, and *Privacy by Design* (LGPD compliance).

## Architecture & Tech Stack

- **Language:** Python 3.11+
- **Orchestration:** LangChain (LCEL)
- **LLM & Embeddings:** Google Gemini API (`gemini-2.5-flash` & `models/text-embedding-004`)
- **Vector Store:** ChromaDB (local persistence)
- **Document Parsing:** PyMuPDF (`fitz`)
- **User Interface:** Streamlit

## Project Structure

```text
institutional-rag-assistant/
├── data/
│   ├── raw_private/          # Git-ignored: Confidential/real organizational documents
│   ├── sample_public/        # Public synthetic documents for testing and auditing
│   └── chroma_db/            # Git-ignored: Local persisted vector store
├── src/
│   ├── __init__.py
│   ├── config.py             # Environment variables and hyperparameter settings
│   ├── ingestion.py          # PDF extraction, cleaning, and semantic chunking
│   ├── vector_store.py       # Embedding generation and ChromaDB indexing/retrieval
│   └── rag_engine.py         # Governance system prompt, LCEL chain, and citations
├── app.py                    # Streamlit chat interface with source verification
├── .env.example              # Template for environment variables
├── .gitignore                # Strict exclusion of private data, .env, and local DB
├── requirements.txt          # Project dependencies
└── README.md                 # Project documentation
```

## Privacy by Design (LGPD)

Sensitive organizational records placed in `data/raw_private/` and local vector indices in `data/chroma_db/` are strictly excluded from version control via `.gitignore`. Public demonstrations and automated tests rely exclusively on synthetic documents stored in `data/sample_public/`.
