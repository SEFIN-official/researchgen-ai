def writer_agent(state, llm):
    query = state["query"]
    research = state["research"]

    prompt = f"""
You are a Writer Agent.

Generate a clear and concise answer.

Rules:
- If question is simple → give short answer (3-5 lines)
- If question is complex → give detailed structured answer
- Avoid unnecessary technical depth unless asked
- Be precise and easy to understand

Question:
{query}

Content:
{research}
"""

    response = llm.invoke(prompt)
    state["final_answer"] = response.content
    return state