from utils.schemas import QueryDecomposition


def decompose_query(query: str, llm) -> list[str]:
    """Return deduplicated search queries for multi-angle retrieval."""
    structured_llm = llm.with_structured_output(QueryDecomposition)
    prompt = f"""
Decompose this research question into focused search queries for a vector database.

Main question:
{query}

Provide:
- main_query: best single query capturing the full question
- theory_query: query for theoretical/complexity/comparison content (empty if not needed)
- architecture_query: query for model architecture/components (empty if not needed)
- experiments_query: query for results, tables, metrics, BLEU, ablations (empty if not needed)

Use empty strings for angles not relevant to the question.
"""
    result: QueryDecomposition = structured_llm.invoke(prompt)

    queries = [result.main_query or query]
    for q in (
        result.theory_query,
        result.architecture_query,
        result.experiments_query,
    ):
        q = (q or "").strip()
        if q and q not in queries:
            queries.append(q)

    return queries
