from typing import Any
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, RunnableParallel, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import LLM_MODEL, LLM_TEMPERATURE, get_gemini_api_key
from src.vector_store import get_retriever

GOVERNANCE_SYSTEM_PROMPT = (
    "Você é um assistente corporativo de governança. "
    "Responda à pergunta do usuário baseando-se EXCLUSIVAMENTE no contexto fornecido abaixo. "
    "Se a resposta não estiver no contexto, diga 'Não encontrei essa informação nos documentos oficiais'. "
    "Sempre cite o nome do documento e a página ao final da resposta.\n\n"
    "Contexto:\n{context}"
)


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents with explicit source and page headers."""
    if not docs:
        return ""

    formatted_chunks: list[str] = []
    for doc in docs:
        source = doc.metadata.get("source", "documento_desconhecido")
        page = doc.metadata.get("page", "?")
        formatted_chunks.append(f"[Fonte: {source} | Página: {page}]\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_chunks)


def get_llm(
    model_name: str = LLM_MODEL,
    temperature: float = LLM_TEMPERATURE,
) -> ChatGoogleGenerativeAI:
    """Initialize the Google Gemini chat model with strict temperature."""
    api_key = get_gemini_api_key()
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )


def build_rag_chain(
    retriever: BaseRetriever | None = None,
    llm: BaseLanguageModel | None = None,
) -> Runnable:
    """Build the LCEL RAG chain returning both answer and source_documents."""
    active_retriever = retriever or get_retriever()
    active_llm = llm or get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GOVERNANCE_SYSTEM_PROMPT),
            ("human", "{question}"),
        ]
    )

    generation_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(x["source_documents"])
        )
        | prompt
        | active_llm
        | StrOutputParser()
    )

    return RunnableParallel(
        source_documents=active_retriever,
        question=RunnablePassthrough(),
    ).assign(answer=generation_chain)


def ask_question(
    question: str,
    retriever: BaseRetriever | None = None,
    llm: BaseLanguageModel | None = None,
) -> dict[str, Any]:
    """Query the RAG engine and return the answer along with retrieved sources."""
    cleaned_question = question.strip()
    if not cleaned_question:
        raise ValueError("Question cannot be empty.")

    chain = build_rag_chain(retriever=retriever, llm=llm)
    result = chain.invoke(cleaned_question)
    return {
        "answer": result["answer"],
        "source_documents": result["source_documents"],
    }
