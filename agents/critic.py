from utils.schemas import CriticVerdict, FinalCriticVerdict
from utils.requirements import format_requirements_for_prompt


def _format_sources(documents) -> str:
    if not documents:
        return "(No source excerpts retrieved.)"
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


def _build_checklist_instructions(requirements: dict) -> str:
    req_text = format_requirements_for_prompt(requirements or {})
    return f"""
Question-driven requirements (enforce strictly):
{req_text}

Checklist rules:
- addresses_all_parts: false if any explicit sub-question is skipped
- has_experimental_evidence: false if experimental evidence was required but notes lack numbers/metrics/table refs
- has_architectural_detail: false if architecture was required but notes only mention concepts vaguely
- has_theoretical_comparison: false if theory was required but notes lack explicit comparisons (e.g. complexity, path length)
- claims_supported_by_sources: false if major claims are not traceable to the source excerpts

Verdict:
- COMPLETE: all checklist items true
- PARTIAL: some requirements met but important evidence/types missing (list missing_points)
- INCOMPLETE: largely fails requirements or unsupported (list missing_points)
"""


def critic_agent(state, llm):
    sources = _format_sources(state.get("documents") or [])
    requirements = state.get("requirements") or {}
    checklist_help = _build_checklist_instructions(requirements)
    structured_llm = llm.with_structured_output(CriticVerdict)

    prompt = f"""
You are a strict research-quality evaluator for NOTES (not the final essay).

Question:
{state["query"]}

{checklist_help}

Source excerpts (ground truth — judge support ONLY against these):
{sources}

Research notes to evaluate:
{state["research"]}

Be strict: if the question asks for experimental evidence, notes MUST include specific numbers
and/or Table/Figure references when they appear in the sources.

Fill checklist honestly. Return specific missing_points for PARTIAL or INCOMPLETE.
"""
    result: CriticVerdict = structured_llm.invoke(prompt)
    state["critique"] = result.verdict
    state["missing_points"] = result.missing_points or []
    state["critic_checklist"] = result.checklist.model_dump()
    return state


def final_critic_agent(state, llm):
    """Second-pass critic on the user-facing final answer."""
    sources = _format_sources(state.get("documents") or [])
    requirements = state.get("requirements") or {}
    checklist_help = _build_checklist_instructions(requirements)
    structured_llm = llm.with_structured_output(FinalCriticVerdict)

    prompt = f"""
You are a strict research-quality evaluator for the FINAL ANSWER shown to the user.

Question:
{state["query"]}

{checklist_help}

Source excerpts (ground truth):
{sources}

Final answer:
{state["final_answer"]}

Evaluate the final answer (not just the notes). Be strict about:
- Missing experimental numbers/tables when the question required evidence
- Missing architectural specifics when required
- Missing theoretical comparisons (e.g. O(1) vs O(n)) when required
- Unsupported claims not found in sources

Fill checklist honestly. Return specific missing_points for PARTIAL or INCOMPLETE.
"""
    result: FinalCriticVerdict = structured_llm.invoke(prompt)
    state["final_critique"] = result.verdict
    state["missing_points"] = result.missing_points or []
    state["final_critic_checklist"] = result.checklist.model_dump()
    return state
