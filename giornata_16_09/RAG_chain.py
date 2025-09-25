#from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
 
import faiss
from langchain.schema import Document
#from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.text_splitter import RecursiveCharacterTextSplitter
 
# LangChain Core (prompt/chain)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
from openai import AzureOpenAI
 
# Chat model init (provider-agnostic, qui puntiamo a LM Studio via OpenAI-compatible)
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
#from config import Config
 
# =========================
# Configurazione
# =========================
 
load_dotenv()


class Config:
    # Persistenza FAISS
    persist_dir: str = "faiss_index_example"
    # Text splitting
    chunk_size: int = 300
    chunk_overlap: int = 30
    # Retriever (MMR)
    search_type: str = "mmr"        # "mmr" o "similarity"
    k: int = 4                      # risultati finali
    fetch_k: int = 20               # candidati iniziali (per MMR)
    mmr_lambda: float = 0.3         # 0 = diversificazione massima, 1 = pertinenza massima
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
 
CONFIG = Config()
# =========================
# Componenti di base
# =========================

 
def generate_embeddings(config: Config) -> AzureOpenAIEmbeddings:

    return AzureOpenAIEmbeddings(azure_endpoint=os.getenv("EMBEDDING_MODEL_ENDPOINT"),
                                 azure_deployment=os.getenv("AZURE_EMBEDDING_MODEL"),
                                 openai_api_version=config.api_version
                                 )
 

def get_llm_from_azure(config: Config):
    """
    Inizializza un ChatModel puntando a Azure OpenAI.
    Richiede:
      - AZURE_OPENAI_ENDPOINT
      - AZURE_OPENAI_API_KEY
      - AZURE_OPENAI_DEPLOYMENT (nome del deployment del modello in Azure)
    """
    endpoint = os.getenv("AZURE_OPENAI_BASE_URL")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = config.api_version
 
    if not endpoint or not api_key or not api_version:
        raise RuntimeError(
            f"AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY e {config.api_version} devono essere impostate per Azure OpenAI."
        )
   
    return AzureChatOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=api_key,
        name=os.getenv("AZURE_OPENAI_NAME"),
        )
 
def simulate_corpus() -> List[Document]:
    """
    Crea un piccolo corpus di documenti in inglese con metadati e 'source' per citazioni.
    """
    docs = [
        Document(
            page_content=(
                "LangChain is a framework that helps developers build applications "
                "powered by Large Language Models (LLMs). It provides chains, agents, "
                "prompt templates, memory, and integrations with vector stores."
            ),
            metadata={"id": "doc1", "source": "intro-langchain.md"}
        ),
        Document(
            page_content=(
                "FAISS is a library for efficient similarity search and clustering of dense vectors. "
                "It supports exact and approximate nearest neighbor search and scales to millions of vectors."
            ),
            metadata={"id": "doc2", "source": "faiss-overview.md"}
        ),
        Document(
            page_content=(
                "Sentence-transformers like all-MiniLM-L6-v2 produce sentence embeddings suitable "
                "for semantic search, clustering, and information retrieval. The embedding size is 384."
            ),
            metadata={"id": "doc3", "source": "embeddings-minilm.md"}
        ),
        Document(
            page_content=(
                "A typical RAG pipeline includes indexing (load, split, embed, store) and "
                "retrieval+generation. Retrieval selects the most relevant chunks, and the LLM produces "
                "an answer grounded in those chunks."
            ),
            metadata={"id": "doc4", "source": "rag-pipeline.md"}
        ),
        Document(
            page_content=(
                "Maximal Marginal Relevance (MMR) balances relevance and diversity during retrieval. "
                "It helps avoid redundant chunks and improves coverage of different aspects."
            ),
            metadata={"id": "doc5", "source": "retrieval-mmr.md"}
        ),
    ]
    return docs

 
 
def split_documents(docs: List[Document], config: Config) -> List[Document]:
    """
    Applica uno splitting robusto ai documenti per ottimizzare il retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=[
            "\n\n", "\n", ". ", "? ", "! ", "; ", ": ",
            ", ", " ", ""  # fallback aggressivo
        ],
    )
    return splitter.split_documents(docs)
 
 
def build_faiss_vectorstore(chunks: List[Document], embeddings: AzureOpenAIEmbeddings, config: Config) -> FAISS:
    """
    Costruisce da zero un FAISS index (IndexFlatL2) e lo salva su disco.
    """
    # Determina la dimensione dell'embedding
    vs = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )
 
    Path(config.persist_dir).mkdir(parents=True, exist_ok=True)
    vs.save_local(config.persist_dir)
    return vs
 
 
def load_or_build_vectorstore(config: Config, embeddings: AzureOpenAIEmbeddings, docs: List[Document]) -> FAISS:
    """
    Tenta il load di un indice FAISS persistente; se non esiste, lo costruisce e lo salva.
    """
    persist_path = Path(config.persist_dir)
    index_file = persist_path / "index.faiss"
    meta_file = persist_path / "index.pkl"
 
    if index_file.exists() and meta_file.exists():
        # Dal 2024/2025 molte build richiedono il flag 'allow_dangerous_deserialization' per caricare pkl locali
        return FAISS.load_local(
            config.persist_dir,
            embeddings,
            allow_dangerous_deserialization=True
        )
 
    chunks = split_documents(docs, config)
    return build_faiss_vectorstore(chunks, embeddings, config) 
 
 
def make_retriever(vector_store: FAISS, config: Config):
    """
    Configura il retriever. Con 'mmr' otteniamo risultati meno ridondanti e più coprenti.
    """
    if config.search_type == "mmr":
        return vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": config.k, "fetch_k": config.fetch_k, "lambda_mult": config.mmr_lambda},
        )
    else:
        return vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": config.k},
        )
 
 
def format_docs_for_prompt(docs: List[Document]) -> str:
    """
    Prepara il contesto per il prompt, includendo citazioni [source].
    """
    lines = []
    for i, d in enumerate(docs, start=1):
        src = d.metadata.get("source", f"doc{i}")
        lines.append(f"[source:{src}] {d.page_content}")
    return "\n\n".join(lines)
 
 
def build_rag_chain(llm, retriever):
    """
    Costruisce la catena RAG (retrieval -> prompt -> LLM) con citazioni e regole anti-hallucination.
    """
    system_prompt = (
        "Sei un assistente esperto. Rispondi in italiano. "
        "Usa esclusivamente il CONTENUTO fornito nel contesto. "
        "Se l'informazione non è presente, dichiara che non è disponibile. "
        "Includi citazioni tra parentesi quadre nel formato [source:...]. "
        "Sii conciso, accurato e tecnicamente corretto."
    )
 
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human",
         "Domanda:\n{question}\n\n"
         "Contesto (estratti selezionati):\n{context}\n\n"
         "Istruzioni:\n"
         "1) Rispondi solo con informazioni contenute nel contesto.\n"
         "2) Cita sempre le fonti pertinenti nel formato [source:FILE].\n"
         "3) Se la risposta non è nel contesto, scrivi: 'Non è presente nel contesto fornito.'")
    ])
 
    # LCEL: dict -> prompt -> llm -> parser
    chain = (
        {
            "context": retriever | format_docs_for_prompt,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
 
 
def rag_answer(question: str, chain) -> str:
    """
    Esegue la catena RAG per una singola domanda.
    """
    return chain.invoke(question)

def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()
 
 
# =========================
# Esecuzione dimostrativa
# =========================
 
 
def main():

    os.environ["TIKTOKEN_CACHE_DIR"] = r"C:\Users\XT286AX\OneDrive - EY\Desktop\data-gym-cache"
    config = Config
 
    # 1) Componenti
    embeddings = generate_embeddings(config)
    llm = get_llm_from_azure(config)
 

    docs = [
        Document(
            page_content=read_file("kpop.txt"),
            metadata={"id": "doc1", "source": "wikipedia-kpop.txt"}
        ),
        Document(
            page_content=read_file("trasformers.txt"),
            metadata={"id": "doc2", "source": "wikipedia-trasformers.txt"}
        )
    ]

    vector_store = load_or_build_vectorstore(config, embeddings, docs)
    # 3) Retriever ottimizzato
    retriever = make_retriever(vector_store, config)
 
    # 4) Catena RAG
    chain = build_rag_chain(llm, retriever)
 
    # 5) Esempi di domande
    questions = [
        "parlami del pop coreano",
        "dimmi come è fatto un modello trasformer?",
        "what is a Trasformers?",
        "Che dimensioni ha il colosseo di milano?",
        "sono bello?"

    ]
 
    for q in questions:
        print("=" * 80)
        print("Q:", q)
        print("-" * 80)
        ans = rag_answer(q, chain)
        print(ans)
        print()
 
if __name__ == "__main__":
    main()