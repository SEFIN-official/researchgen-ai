from rag.retriever import get_retriever
from rag.query_decompose import decompose_query


def _doc_key(doc) -> str:
    content = (doc.page_content or "")[:500]
    meta = str(sorted((doc.metadata or {}).items()))
    return f"{content}::{meta}"


def multi_query_retrieve(query: str, k: int, llm) -> list:
    """Retrieve using decomposed sub-queries and merge unique chunks."""
    sub_queries = decompose_query(query, llm)
    n = max(len(sub_queries), 1)
    per_query_k = max(2, k // n)

    seen = set()
    merged = []

    for sq in sub_queries:
        retriever = get_retriever(k=per_query_k)
        for doc in retriever.invoke(sq):
            key = _doc_key(doc)
            if key not in seen:
                seen.add(key)
                merged.append(doc)

    if len(merged) < k:
        retriever = get_retriever(k=k)
        for doc in retriever.invoke(query):
            key = _doc_key(doc)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
            if len(merged) >= k:
                break

    return merged[:k] if len(merged) > k else merged
