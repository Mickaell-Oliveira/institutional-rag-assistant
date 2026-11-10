from pathlib import Path
import streamlit as st

from src.config import (
    CHROMA_PERSIST_DIR,
    RAW_PRIVATE_DIR,
    SAMPLE_PUBLIC_DIR,
    get_gemini_api_key,
)
from src.ingestion import ingest_documents
from src.rag_engine import ask_question
from src.vector_store import create_vector_store, load_vector_store

st.set_page_config(
    page_title="Institutional RAG Assistant",
    page_icon="🏛️",
    layout="wide",
)


def get_vector_store_status() -> tuple[bool, int]:
    """Check if the local ChromaDB database exists and return total chunk count."""
    persist_path = Path(CHROMA_PERSIST_DIR)
    sqlite_file = persist_path / "chroma.sqlite3"
    if not sqlite_file.exists():
        return False, 0
    try:
        store = load_vector_store(persist_path)
        count = store._collection.count()
        return True, count
    except Exception:
        return False, 0


def render_sidebar():
    """Render sidebar controls for ingestion, vector store management and session."""
    st.sidebar.title("🏛️ Governança & Dados")
    st.sidebar.caption("Controle de base vetorial e conformidade LGPD")

    source_options = {
        "Documentos Sintéticos Públicos (sample_public)": SAMPLE_PUBLIC_DIR,
        "Documentos Confidenciais (raw_private)": RAW_PRIVATE_DIR,
    }

    selected_label = st.sidebar.selectbox(
        "Origem dos Documentos:",
        options=list(source_options.keys()),
        index=0,
    )
    selected_dir = source_options[selected_label]

    is_indexed, chunk_count = get_vector_store_status()

    st.sidebar.markdown("---")
    st.sidebar.subheader("Status do Banco Vetorial")
    if is_indexed:
        st.sidebar.success(f"Base ativa: {chunk_count} trechos indexados")
    else:
        st.sidebar.warning("Base não indexada localmente")

    if st.sidebar.button("Indexar / Atualizar Base", use_container_width=True):
        try:
            get_gemini_api_key()
        except ValueError as err:
            st.sidebar.error(str(err))
            return

        with st.sidebar.status("Processando documentos...", expanded=True) as status:
            try:
                st.write("1. Extraindo PDFs e gerando chunks semânticos...")
                chunks = ingest_documents(selected_dir)
                if not chunks:
                    status.update(label="Nenhum PDF encontrado!", state="error")
                    st.sidebar.warning(f"Nenhum arquivo PDF encontrado em '{selected_dir.name}'.")
                    return

                st.write(f"2. Gerando embeddings para {len(chunks)} blocos no ChromaDB...")
                create_vector_store(chunks, reset=True)
                status.update(label="Indexação concluída com sucesso!", state="complete")
                st.sidebar.success(f"{len(chunks)} trechos indexados em disco!")
                st.rerun()
            except Exception as e:
                status.update(label="Falha na indexação", state="error")
                st.sidebar.error(f"Erro ao processar: {e}")

    if st.sidebar.button("Limpar Histórico de Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Privacy by Design:**\n"
        "- Documentos em `data/raw_private/` e o banco vetorial local são 100% isolados pelo `.gitignore`.\n"
        "- Respostas estritamente ancoradas nos trechos oficiais recuperados."
    )


def main():
    render_sidebar()

    st.title("🏛️ Institutional Governance RAG Assistant")
    st.markdown(
        "Consulte normas, estatutos e manuais corporativos com **rastreabilidade estrita** "
        "e garantia de prevenção contra alucinações."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            sources = msg.get("sources")
            if sources:
                with st.expander("Fontes consultadas"):
                    for idx, src in enumerate(sources, start=1):
                        st.markdown(
                            f"**[{idx}] Documento:** `{src['source']}` — **Página:** {src['page']}"
                        )
                        st.caption(src["content"])

    # Chat input
    prompt = st.chat_input("Digite sua pergunta sobre a governança institucional...")
    if prompt:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Check API key
        try:
            get_gemini_api_key()
        except ValueError as key_err:
            st.error(str(key_err))
            return

        # Check if vector store is initialized
        is_indexed, _ = get_vector_store_status()
        if not is_indexed:
            msg_warning = (
                "A base de dados vetorial ainda não foi indexada. "
                "Por favor, clique em **'Indexar / Atualizar Base'** na barra lateral."
            )
            st.warning(msg_warning)
            st.session_state.messages.append(
                {"role": "assistant", "content": msg_warning, "sources": []}
            )
            return

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Consultando documentos oficiais..."):
                try:
                    result = ask_question(prompt)
                    answer = result["answer"]
                    source_docs = result.get("source_documents", [])

                    st.markdown(answer)

                    serialized_sources = []
                    if source_docs:
                        with st.expander("Fontes consultadas"):
                            for idx, doc in enumerate(source_docs, start=1):
                                src_name = doc.metadata.get("source", "desconhecido")
                                page_num = doc.metadata.get("page", "?")
                                content = doc.page_content

                                serialized_sources.append(
                                    {
                                        "source": src_name,
                                        "page": page_num,
                                        "content": content,
                                    }
                                )
                                st.markdown(
                                    f"**[{idx}] Documento:** `{src_name}` — **Página:** {page_num}"
                                )
                                st.caption(content)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": serialized_sources,
                        }
                    )
                except Exception as err:
                    st.error(f"Erro ao processar a pergunta: {err}")


if __name__ == "__main__":
    main()
