from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 

def get_retriever():
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma(
        persist_directory="embeddings/chroma_db",
        embedding_function=embedding
    )

    return vectordb.as_retriever(search_kwargs={"k": 3})