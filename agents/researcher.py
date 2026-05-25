from utils.requirements import format_requirements_for_prompt


def _format_sources(documents) -> str:
    if not documents:
        return "(No source excerpts available.)"
    parts = []
    for i, doc in enumerate(documents, start=1):
        meta = doc.metadata or {}
        label = f"[Source {i}"
        if meta.get("table_heavy"):
            label += ", table-heavy"
        if meta.get("page") is not None:
            label += f", p.{meta['page']}"
        label += "]"
        parts.append(f"{label}\n{doc.page_content}")
    return "\n\n".join(parts)


def researcher_agent(state, llm):
    query = state["query"]
    context = _format_sources(state.get("documents") or [])
    missing_points = state.get("missing_points") or []
    requirements = state.get("requirements") or {}
    req_block = format_requirements_for_prompt(requirements)

    feedback_block = ""
    if missing_points:
        gaps = "\n".join(f"- {point}" for point in missing_points)
        feedback_block = f"""
Previous attempt was judged INCOMPLETE or PARTIAL. Fix these gaps using ONLY the sources below:
{gaps}
"""

    prompt = f"""
You are a Researcher Agent preparing evidence-backed notes for a research answer.

Question:
{query}

Requirements for this question:
{req_block}

Rules:
- Use ONLY the source excerpts below
- Include exact figures, metrics, and Table/Figure IDs when they appear in sources (e.g. "Table 2", "28.4 BLEU")
- Use structured bullet points grouped by: Theory | Architecture | Experiments (when relevant)
- Cite [Source N] for every major claim
- Do NOT be overly brief — completeness matters more than brevity for advanced questions
{feedback_block}
Source excerpts:
{context}
"""

    response = llm.invoke(prompt)
    state["research"] = response.content
    return state
