from utils.requirements import format_requirements_for_prompt


def writer_agent(state, llm):
    query = state["query"]
    research = state["research"]
    complexity = state.get("complexity", "ADVANCED")
    requirements = state.get("requirements") or {}
    missing_points = state.get("missing_points") or []
    is_revision = state.get("final_retry_count", 0) > 0

    if complexity == "SIMPLE":
        prompt = f"""
You are a Writer Agent.

Answer concisely (3–5 lines). Be clear and precise.

Question:
{query}

Content:
{research}
"""
    else:
        req_block = format_requirements_for_prompt(requirements)
        revision_block = ""
        if is_revision and missing_points:
            gaps = "\n".join(f"- {p}" for p in missing_points)
            revision_block = f"""
REVISION REQUIRED — the previous final answer was judged incomplete. Fix:
{gaps}
"""

        prompt = f"""
You are a Writer Agent producing a research-grade answer.

Question:
{query}

{req_block}
{revision_block}

Rules for ADVANCED answers:
- Use structured sections (e.g. Theoretical basis, Architecture, Experimental evidence, Conclusion)
- MUST include quantitative results and Table/Figure references from the research notes — do NOT strip numbers
- MUST include explicit theoretical comparisons when present in the notes (e.g. path length, complexity)
- MUST name architectural components when present in the notes
- Form a nuanced synthesis in the conclusion — do not list bullets only
- Only state claims supported by the research notes; do not invent results

Research notes:
{research}
"""

    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state
