# ResearchGen AI

A multi-agent research assistant that lets you **upload PDF papers**, **ask complex questions**, and receive **structured, evidence-grounded answers** powered by RAG, LangGraph, and Google Gemini.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-green)

---

## Features

- **PDF upload & indexing** — ChromaDB vector store with table-aware chunking
- **Smart routing** — SIMPLE questions get fast answers; ADVANCED questions use the full agent pipeline
- **Multi-agent workflow** — Researcher → Critic → Writer with retry loops
- **Multi-query retrieval** — Decomposes questions into theory / architecture / experiments searches
- **Grounded critics** — Requirement-aware evaluation with checklist (experimental, architectural, theoretical evidence)
- **Second-pass QA** — Final answer is reviewed before delivery; one revision if needed
- **Chat UI** — Streamlit interface with conversation history

---

## Architecture
