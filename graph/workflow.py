from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any

from agents.researcher import researcher_agent
from agents.critic import critic_agent, final_critic_agent
from agents.writer import writer_agent
from rag.retriever import DEFAULT_K, RETRY_K_INCREMENT
from rag.multi_retrieve import multi_query_retrieve
from utils.llm import get_llm, get_judge_llm
from utils.schemas import ComplexityClassification
from utils.requirements import detect_requirements

MAX_RETRIES = 2
MAX_FINAL_RETRIES = 1


class GraphState(TypedDict):
    query: str
    documents: list
    research: str
    critique: str
    final_answer: str
    complexity: str
    missing_points: List[str]
    retry_count: int
    retrieval_k: int
    requirements: Dict[str, Any]
    critic_checklist: Dict[str, Any]
    final_critique: str
    final_critic_checklist: Dict[str, Any]
    final_retry_count: int


def classify(state, llm):
    structured_llm = llm.with_structured_output(ComplexityClassification)
    prompt = f"""
Classify this question for a research assistant.

SIMPLE: basic definition or short factual answer; no document comparison needed.
ADVANCED: explanation, comparison, synthesis, or reasoning that benefits from uploaded PDFs.

Question:
{state["query"]}
"""
    result: ComplexityClassification = structured_llm.invoke(prompt)
    state["complexity"] = result.complexity
    return state


def set_requirements(state):
    state["requirements"] = detect_requirements(state["query"])
    return state


def retrieve(state, llm):
    k = state.get("retrieval_k") or DEFAULT_K
    docs = multi_query_retrieve(state["query"], k, llm)
    state["documents"] = docs
    state["retrieval_k"] = k
    return state


def simple_writer(state, llm):
    query = state["query"]
    prompt = f"""
Answer the question concisely.

Rules:
- Keep it short (3–5 lines)
- No unnecessary details
- Clear explanation

Question:
{query}
"""
    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state


def route_after_classify(state):
    if state["complexity"] == "SIMPLE":
        return "simple_writer"
    return "set_requirements"


def prepare_retry(state):
    state["retry_count"] = state.get("retry_count", 0) + 1
    state["retrieval_k"] = state.get("retrieval_k", DEFAULT_K) + RETRY_K_INCREMENT
    return state


def prepare_final_retry(state):
    state["final_retry_count"] = state.get("final_retry_count", 0) + 1
    return state


def check_research(state):
    if state["critique"] in ("INCOMPLETE", "PARTIAL") and state.get("retry_count", 0) < MAX_RETRIES:
        return "prepare_retry"
    return "writer"


def check_final(state):
    if state.get("final_critique") in ("INCOMPLETE", "PARTIAL") and state.get(
        "final_retry_count", 0
    ) < MAX_FINAL_RETRIES:
        return "prepare_final_retry"
    return END  # noqa: graph terminal


def build_graph():
    llm = get_llm()
    judge_llm = get_judge_llm()

    builder = StateGraph(GraphState)

    builder.add_node("classify", lambda s: classify(s, judge_llm))
    builder.add_node("set_requirements", set_requirements)
    builder.add_node("retrieve", lambda s: retrieve(s, judge_llm))
    builder.add_node("researcher", lambda s: researcher_agent(s, llm))
    builder.add_node("critic", lambda s: critic_agent(s, judge_llm))
    builder.add_node("prepare_retry", prepare_retry)
    builder.add_node("writer", lambda s: writer_agent(s, llm))
    builder.add_node("final_critic", lambda s: final_critic_agent(s, judge_llm))
    builder.add_node("prepare_final_retry", prepare_final_retry)
    builder.add_node("simple_writer", lambda s: simple_writer(s, llm))

    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_after_classify)

    builder.add_edge("set_requirements", "retrieve")
    builder.add_edge("retrieve", "researcher")
    builder.add_edge("researcher", "critic")
    builder.add_conditional_edges("critic", check_research)
    builder.add_edge("prepare_retry", "retrieve")
    builder.add_edge("writer", "final_critic")
    builder.add_conditional_edges("final_critic", check_final)
    builder.add_edge("prepare_final_retry", "writer")
    builder.add_edge("simple_writer", END)

    return builder.compile()
