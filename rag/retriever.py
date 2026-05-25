from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

DEFAULT_K = 8
RETRY_K_INCREMENT = 4
MAX_K = 16


def get_retriever(k: int = DEFAULT_K):
    k = min(max(k, 1), MAX_K)
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory="embeddings/chroma_db",
        embedding_function=embedding,
    )

    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": max(k * 4, 24)},
    )
