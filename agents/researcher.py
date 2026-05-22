def researcher_agent(state, llm):
    query = state["query"]
    context_docs = state["documents"]

    context = "\n\n".join([doc.page_content for doc in context_docs])

    prompt = f"""
You are a Researcher Agent.

Extract key points relevant to the question.

Question:
{query}

Context:
{context}

Return only important points.
"""

    response = llm.invoke(prompt)

    state["research"] = response.content
    return state