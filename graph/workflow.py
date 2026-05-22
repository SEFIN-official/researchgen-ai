from langgraph.graph import StateGraph, END
from typing import TypedDict

from agents.researcher import researcher_agent
from agents.critic import critic_agent
from agents.writer import writer_agent
from rag.retriever import get_retriever
from utils.llm import get_llm


class GraphState(TypedDict):
    query: str
    documents: list
    research: str
    critique: str
    final_answer: str
    complexity: str  # NEW


# 🔍 1. CLASSIFIER NODE
def classify(state, llm):
    prompt = f"""
Classify this question.

Return ONLY:
- SIMPLE (basic definition, short answer)
- ADVANCED (requires explanation, comparison, deep reasoning)

Question:
{state["query"]}
"""

    response = llm.invoke(prompt).content.strip().upper()

    if "ADVANCED" in response:
        state["complexity"] = "ADVANCED"
    else:
        state["complexity"] = "SIMPLE"

    return state


# 📚 2. RETRIEVAL
def retrieve(state):
    retriever = get_retriever()
    docs = retriever.invoke(state["query"])
    state["documents"] = docs
    return state


# 🚀 3. DIRECT WRITER (for SIMPLE queries)
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


# 🔁 4. CONDITIONAL FLOW AFTER CLASSIFICATION
def route_after_classify(state):
    if state["complexity"] == "SIMPLE":
        return "simple_writer"
    return "retrieve"


# 🔁 5. CRITIC CHECK
def check(state):
    if "INCOMPLETE" in state["critique"]:
        return "researcher"
    return "writer"


# 🧠 BUILD GRAPH
def build_graph():
    llm = get_llm()

    builder = StateGraph(GraphState)

    # Nodes
    builder.add_node("classify", lambda s: classify(s, llm))
    builder.add_node("retrieve", retrieve)
    builder.add_node("researcher", lambda s: researcher_agent(s, llm))
    builder.add_node("critic", lambda s: critic_agent(s, llm))
    builder.add_node("writer", lambda s: writer_agent(s, llm))
    builder.add_node("simple_writer", lambda s: simple_writer(s, llm))

    # Entry point
    builder.set_entry_point("classify")

    # Flow
    builder.add_conditional_edges("classify", route_after_classify)

    # ADVANCED path
    builder.add_edge("retrieve", "researcher")
    builder.add_edge("researcher", "critic")
    builder.add_conditional_edges("critic", check)
    builder.add_edge("writer", END)

    # SIMPLE path
    builder.add_edge("simple_writer", END)

    return builder.compile()