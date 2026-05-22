from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings 
# Note: sentence-transformers is often used via langchain-huggingface now
import os

def ingest_uploaded_files(file_paths):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_docs = []

    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        all_docs.extend(docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(all_docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="embeddings/chroma_db"
    )

    vectordb.persist()
    return "✅ Documents uploaded & indexed!"