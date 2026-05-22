def critic_agent(state, llm):
    prompt = f"""
You are a strict evaluator.

Answer ONLY one word:
- COMPLETE
- INCOMPLETE

Do not explain.

Question: {state["query"]}
Answer: {state["research"]}
"""

    response = llm.invoke(prompt).content.strip().upper()

    if "INCOMPLETE" in response:
        state["critique"] = "INCOMPLETE"
    else:
        state["critique"] = "COMPLETE"

    return state