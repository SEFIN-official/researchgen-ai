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

```
Upload PDFs → Ingest (chunk + embed) → ChromaDB

User question
    → Classify (SIMPLE / ADVANCED)
    → [ADVANCED]
        → Detect requirements
        → Multi-query retrieve
        → Researcher (evidence notes)
        → Critic (notes QA, retry up to 2×)
        → Writer (structured answer)
        → Final critic (answer QA, revise up to 1×)
    → [SIMPLE] → Direct short answer
```

### Tech stack

| Component | Technology |
|-----------|------------|
| UI | Streamlit |
| Orchestration | LangGraph |
| LLM | Google Gemini 2.5 Flash |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` |
| Vector DB | ChromaDB |
| PDF parsing | PyPDF |

---

## Project structure

```
Research-agent/
├── app.py                 # Streamlit entry point
├── agents/
│   ├── researcher.py      # Extracts evidence-backed notes
│   ├── critic.py          # Research + final answer evaluation
│   └── writer.py          # Produces user-facing answer
├── graph/
│   └── workflow.py        # LangGraph pipeline
├── rag/
│   ├── ingest.py          # PDF indexing (table-aware chunks)
│   ├── retriever.py       # MMR retrieval
│   ├── query_decompose.py # Sub-query generation
│   └── multi_retrieve.py  # Multi-angle retrieval
├── utils/
│   ├── llm.py             # Gemini client
│   ├── schemas.py         # Structured outputs (Pydantic)
│   └── requirements.py    # Question requirement detection
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Prerequisites

- Python 3.10 or newer
- [Google Gemini API key](https://aistudio.google.com/apikey)

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/SEFIN-official/researchgen-ai.git
cd researchgen-ai
```

### 2. Create and activate a virtual environment

**Windows (PowerShell):**

```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy the example env file and add your API key:

```bash
copy .env.example .env        # Windows
# cp .env.example .env        # macOS / Linux
```

Edit `.env`:

```env
GOOGLE_API_KEY=your_api_key_here
```

---

## Run the app

```bash
streamlit run app.py
```

Open the URL shown in the terminal (usually `http://localhost:8501`).

---

## Usage

1. **Upload** one or more PDF research papers.
2. Click **Upload & Index** and wait for the success message.
3. **Ask a question** in the chat box.

### Example flow

| Step | Action |
|------|--------|
| 1 | Upload a paper (e.g. Transformer / ML paper) |
| 2 | Index documents |
| 3 | Ask a complex question |

**Example question:**

> Compare the model's theoretical advantages, architectural components, and experimental results (including tables and metrics) using evidence from the uploaded paper.

4. Review the **structured multi-agent answer** with theory, architecture, experiments, and synthesis.

### Question types

| Type | Example | Pipeline |
|------|---------|----------|
| SIMPLE | "What is gradient descent?" | Fast direct answer |
| ADVANCED | "Compare Adam vs SGD using my paper" | Full RAG + multi-agent |

---

## Configuration (optional)

| Setting | Location | Default |
|---------|----------|---------|
| Retrieval chunk count | `rag/retriever.py` | `DEFAULT_K = 8` |
| Max research retries | `graph/workflow.py` | `MAX_RETRIES = 2` |
| Max final answer revisions | `graph/workflow.py` | `MAX_FINAL_RETRIES = 1` |
| LLM model | `utils/llm.py` | `gemini-2.5-flash` |

---

## Re-indexing after code updates

If you change ingestion logic, **re-upload and re-index** your PDFs so ChromaDB uses the latest chunking strategy.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| API key error | Set `GOOGLE_API_KEY` in `.env` and restart Streamlit |
| Generic answers | Confirm PDFs were indexed; ask ADVANCED-style questions |
| Quota exceeded | Wait and retry; check Gemini API limits |
| Missing table data | Re-index PDFs; tables depend on PDF text extraction quality |

---

## Limitations

- Answers are only as good as retrieved PDF chunks.
- PDF tables may not parse perfectly as structured data.
- ADVANCED questions use more API calls (higher latency and cost).

---

## License

MIT (or add your preferred license).

---

## Acknowledgments

Built with [LangChain](https://www.langchain.com/), [LangGraph](https://langchain-ai.github.io/langgraph/), [Streamlit](https://streamlit.io/), [ChromaDB](https://www.trychroma.com/), and [Google Gemini](https://ai.google.dev/).
