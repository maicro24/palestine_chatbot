<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Flag_of_Palestine.svg/320px-Flag_of_Palestine.svg.png" width="160"/>

# 🕌 AI for Palestine Smart Library
### مكتبة فلسطين الذكية — Agentic RAG System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red?logo=streamlit)](https://streamlit.io)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-purple)](https://langchain-ai.github.io/langgraph/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Competition Entry** — AI for Palestine 2024  
> A bilingual (Arabic 🇵🇸 / English 🇬🇧) Agentic RAG application over 15 curated documents on Palestinian history and humanitarian crises.

</div>

---

## 📌 Table of Contents
- [Overview](#-overview)
- [Architecture](#-architecture--5-node-langgraph-workflow)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Document Corpus](#-document-corpus-15-pdfs)
- [Competition Notes](#-competition-notes)

---

## 🎯 Overview

The **AI for Palestine Smart Library** is a production-grade, bilingual Agentic RAG system that enables researchers, journalists, and the public to query 15 curated documents about Palestine — with **strict citation requirements** and **anti-hallucination guarantees**.

Every answer must be grounded in the documents and end with:
```
(Source: [Document Title], Page: [Page Number])
```

---

## 🏗️ Architecture — 5-Node LangGraph Workflow

```
User Query
    │
    ▼
┌─────────────────────┐
│  Node 1             │  Detect language (ar/en)
│  Analyze Query      │  Classify intent: factual / analytical / comparative
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Node 2             │  Set dynamic k (4 / 6 / 8 chunks)
│  Plan Retrieval     │  Expand query with context keywords
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Node 3             │  ChromaDB similarity_search(k)
│  Retrieve           │  Returns top-k document chunks
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Node 4             │  LLM-as-judge: yes/no per chunk
│  Grade Documents    │  Keeps only relevant chunks
└────────┬────────────┘
         │
    ┌────┴────┐
   Yes       No
    │         │
    ▼         ▼
┌────────┐  ┌──────────┐
│ Node 5 │  │ Fallback │
│Generate│  │"not found│
│        │  │in docs"  │
└───┬────┘  └──────────┘
    │
    ▼
Cited Bilingual Answer
(Source: [Title], Page: [N])
```

---

## ✨ Features

| Feature | Description |
|---|---|
| 💬 **Smart Chat** | Full Agentic RAG chat with source expander & workflow trace |
| 🔍 **Discourse Analysis** | Bias, propaganda & framing detection |
| ⚖️ **Compare Documents** | Side-by-side bilingual topic comparison |
| 📄 **Document Summary** | AI-generated summaries per document (language-aware) |
| 🗺️ **Interactive Map** | Folium map of key Palestinian locations |
| 📅 **Historical Timeline** | 15 key events from 1917 to 2024 |
| ☁️ **Word Cloud** | Generated from live ChromaDB corpus |
| 📊 **Statistics** | Chunks, docs, languages, response times |
| 📤 **Upload PDF (Secret Test)** | Live indexing → verification → Q&A — no restart needed |
| ℹ️ **About** | Team info, Mermaid architecture diagram |

### 🔐 Anti-Hallucination Rules (enforced in LLM system prompt)
1. Answer **ONLY** from provided context
2. If not found → reply **EXACTLY**: `"not found in documents"`
3. Every answer **MUST** end with: `(Source: [Title], Page: [N])`
4. Auto-detect language → Arabic question = Arabic answer

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | AI Grid · `Qwen3-30B-A3B-Thinking` via OpenAI-compatible API |
| **Orchestration** | LangGraph `StateGraph` (5 nodes + conditional edges) |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| **Vector DB** | ChromaDB (persistent, local — no cloud needed) |
| **PDF Parsing** | pdfplumber (Arabic) + PyPDFLoader (fallback) |
| **UI** | Streamlit (10 tabs, dark glassmorphism theme) |
| **Map** | Folium + streamlit-folium |

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/maicro24/palestine_chatbot.git
cd palestine_chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure secrets
Create `.streamlit/secrets.toml`:
```toml
AI_API_KEY  = "your-ai-grid-api-key"
AI_MODEL    = "Qwen3-30B-A3B-Thinking"
AI_BASE_URL = "http://app.ai-grid.io:4000/v1"
```

### 4. Add your PDFs
Place all 15 PDF files inside the `data/` folder.

### 5. Build the vector database (run ONCE)
```bash
python ingest.py
```

> **Tip:** To force a full rebuild: `python ingest.py --force-rebuild`

### 6. Launch the app
```bash
python -m streamlit run app.py
```

Open → **http://localhost:8501**

---

## 📁 Project Structure

```
palestine_chatbot/
│
├── app.py                  # Main Streamlit application (10 tabs)
├── ingest.py               # PDF ingestion pipeline → ChromaDB
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/                   # 📚 Place your 15 PDFs here
│   ├── فلسطين.pdf
│   ├── كتاب-النخبة-1-.pdf
│   ├── Khalidi-Rashid-Palestinian-Identity.pdf
│   └── ... (12 more)
│
├── chroma_db/              # 🔒 Auto-created by ingest.py (vector store)
│   └── ...
│
├── .streamlit/
│   └── secrets.toml        # 🔑 API keys (NOT committed to Git)
│
└── .pdf_hashes.json        # 🔄 Change-detection cache (skip unchanged PDFs)
```

---

## 📚 Document Corpus (15 PDFs)

| # | Title | Language |
|---|---|---|
| 1 | 20241106-Gaza-Update-Report-OPT | 🇬🇧 English |
| 2 | 2024_04_20_UNRWA-final-technical_report | 🇬🇧 English |
| 3 | Humanitarian-Situation-Update-176 (UN OCHA) | 🇬🇧 English |
| 4 | Israel-Palestine-History-Timeline-2024-25 | 🇬🇧 English |
| 5 | Khalidi-Rashid-Palestinian-Identity | 🇬🇧 English |
| 6 | Palestinian-History-Calendar | 🇬🇧 English |
| 7 | The Hundred Years' War on Palestine | 🇬🇧 English |
| 8 | ga_res_1941948 (UN General Assembly) | 🇬🇧 English |
| 9 | تقرير غزة الإنساني 2024 | 🇵🇸 Arabic |
| 10 | ذاكرة المكان | 🇵🇸 Arabic |
| 11 | شخصيات فلسطينية | 🇵🇸 Arabic |
| 12 | فلسطين العربية | 🇵🇸 Arabic |
| 13 | فلسطين | 🇵🇸 Arabic |
| 14 | كتاب النخبة - الجزء الأول | 🇵🇸 Arabic |
| 15 | كتاب النخبة - الجزء الثاني | 🇵🇸 Arabic |

---

## 🏆 Competition Notes

### Mandatory Requirements Checklist ✅

- [x] **15 PDF documents** processed (Arabic + English)
- [x] **Smart chunking** — `RecursiveCharacterTextSplitter` (size=1000, overlap=200)
- [x] **Metadata on EVERY chunk**: `document_title` + `page_number`
- [x] **Free embeddings** — `sentence-transformers/all-MiniLM-L6-v2`
- [x] **Persistent vector DB** — ChromaDB (no rebuild on re-run, hash-based cache)
- [x] **Anti-hallucination** — strict system prompt + grading node
- [x] **Citations** — every answer ends with `(Source: [Title], Page: [N])`
- [x] **Bilingual** — Arabic & English auto-detected and answered in kind
- [x] **Agentic RAG** — LangGraph `StateGraph` with 5 nodes + conditional routing
- [x] **Live PDF upload** — no restart needed (15-point secret test)
- [x] **10 Streamlit tabs** — all functional

### Scoring Breakdown (estimated)
| Criterion | Points |
|---|---|
| PDF Ingestion + Metadata | 20 |
| RAG Pipeline Quality | 25 |
| Anti-Hallucination + Citations | 20 |
| Bilingual Support | 10 |
| Live PDF Upload (Secret Test) | **15** |
| UI / UX (10 tabs) | 10 |

---

## 👥 Team

Built with ❤️ for Palestine — leveraging AI for truth, memory, and justice.

> *"To exist is to resist."*

---

<div align="center">
<sub>AI for Palestine Competition 2024 · MIT License</sub>
</div>
