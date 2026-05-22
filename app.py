import streamlit as st
import os
from datetime import datetime
from graph.workflow import build_graph
from rag.ingest import ingest_uploaded_files


# Initialize graph
graph = build_graph()


def save_uploaded_files(uploaded_files):
    os.makedirs("uploads", exist_ok=True)
    paths = []
    for f in uploaded_files:
        path = os.path.join("uploads", f.name)
        with open(path, "wb") as out:
            out.write(f.getbuffer())
        paths.append(path)
    return paths


def main():
    st.set_page_config(page_title="ResearchGen AI", layout="centered")

    st.markdown("# 🧠 ResearchGen AI")
    st.markdown("### Upload your documents → Ask questions → Get structured answers")

    # Upload section (top)
    uploaded_files = st.file_uploader(
        "📄 Upload PDF Documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Upload & Index"):
        if uploaded_files:
            paths = save_uploaded_files(uploaded_files)
            status = ingest_uploaded_files(paths)
            st.success(status)
        else:
            st.warning("No files selected.")

    st.divider()

    # Chat section (bottom) — ChatGPT-style UI
    if "history" not in st.session_state:
        st.session_state.history = []

    # Render chat messages using Streamlit chat components
    for m in st.session_state.history:
        role = m.get("role", "user")
        content = m.get("content", "")
        ts = m.get("time")
        with st.chat_message(role):
            if ts:
                st.markdown(f"*{ts}*")
            st.markdown(content)

    # Use chat_input for ChatGPT-like input
    user_input = st.chat_input("Ask any AI/ML question...")

    if user_input:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # append user message
        st.session_state.history.append({"role": "user", "content": user_input, "time": now})

        # show assistant placeholder
        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("⏳ Thinking...")

        try:
            result = graph.invoke({
                "query": user_input,
                "documents": [],
                "research": "",
                "critique": "",
                "final_answer": "",
                "complexity": ""
            })
            answer = result.get("final_answer", "No response generated.")
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                answer = "⚠️ API quota exceeded. Please wait 30 seconds and try again."
            else:
                answer = f"⚠️ Error: {str(e)}"

        # replace placeholder and save assistant message
        placeholder.markdown(answer)
        st.session_state.history.append({"role": "assistant", "content": answer, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


if __name__ == "__main__":
    main()