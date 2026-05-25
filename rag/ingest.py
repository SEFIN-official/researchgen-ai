import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_TABLE_PATTERN = re.compile(r"\btable\s+\d+\b", re.I)
_NUMERIC_LINE = re.compile(r"(\d+\.?\d*[^\d\n]*){2,}")


def _is_table_heavy(text: str) -> bool:
    if _TABLE_PATTERN.search(text):
        return True
    lines = text.split("\n")
    numeric_lines = sum(1 for line in lines if len(re.findall(r"\d+\.?\d*", line)) >= 3)
    return numeric_lines >= 2


def _split_documents(all_docs):
    normal_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "],
    )
    table_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = []
    for doc in all_docs:
        text = doc.page_content or ""
        splitter = table_splitter if _is_table_heavy(text) else normal_splitter
        for chunk in splitter.split_documents([doc]):
            chunk.metadata = dict(doc.metadata)
            chunk.metadata["table_heavy"] = _is_table_heavy(chunk.page_content)
            chunks.append(chunk)
    return chunks


def ingest_uploaded_files(file_paths):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    all_docs = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())

    chunks = _split_documents(all_docs)

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="embeddings/chroma_db",
    )

    vectordb.persist()
    return "✅ Documents uploaded & indexed (table-aware chunking applied)!"
