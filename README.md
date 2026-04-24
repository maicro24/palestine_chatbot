# 🕌 AI for Palestine Smart Library
### Agentic RAG System | مكتبة فلسطين الذكية

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic_RAG-purple?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-green?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge)
![AI Grid](https://img.shields.io/badge/AI_Grid-LLM-orange?style=for-the-badge)

**A bilingual (Arabic/English) production-grade RAG application built for the AI for Palestine competition.**

</div>

---

## 🎯 Overview

The **AI for Palestine Smart Library** is a complete Agentic RAG (Retrieval-Augmented Generation) system that enables intelligent, cited, bilingual Q&A over 15 curated Palestinian history and humanitarian documents.

### Key Features
- ✅ **5-Node LangGraph Agentic Workflow** — Query Analysis → Plan → Retrieve → Grade → Generate
- ✅ **Anti-Hallucination** — Strict "not found in documents" fallback when evidence is missing
- ✅ **Mandatory Citations** — Every answer ends with `(Source: [Title], Page: [N])`
- ✅ **Bilingual** — Auto-detects Arabic/English and responds in the same language
- ✅ **15 PDF Corpus** — Arabic & English documents fully indexed with metadata
- ✅ **Live PDF Upload** — Dynamically index new documents without restart (Secret Test)
- ✅ **10-Tab Streamlit Dashboard** — Chat, Analysis, Maps, Timeline, WordCloud, Stats & more

---

## 🏗️ Architecture — 5-Node LangGraph

```
User Query
    │
    ▼
┌─────────────────────┐
│  Node 1             │  Detect language (ar/en)
│  Analyze Query      │  Classify: factual / analytical / comparative
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Node 2             │  Set k (4/6/8 based on type)
│  Plan Retrieval     │  Expand query with context keywords
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Node 3             │  ChromaDB similarity_search(query, k)
│  Retrieve           │  Returns top-k chunks with metadata
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Node 4             │  LLM-as-judge: grades each chunk yes/no
│  Grade Documents    │  Keeps only relevant chunks
└────────┬────────────┘
         │
    ┌────┴────┐
    │         │
   Yes        No
    │         │
    ▼         ▼
┌────────┐  ┌──────────────┐
│ Node 5 │  │   Fallback   │
│Generate│  │not found in  │
│+Cite   │  │documents     │
└────────┘  └──────────────┘
```

---

## 📚 Document Corpus (15 PDFs)

| # | Document | Language |
|---|---|---|
| 1 | 20241106-Gaza-Update-Report-OPT | 🇬🇧 English |
| 2 | 2024_04_20_UNRWA-final-technical_report | 🇬🇧 English |
| 3 | Humanitarian-Situation-Update-176 | 🇬🇧 English |
| 4 | Israel-Palestine-History-Timeline-2024-25 | 🇬🇧 English |
| 5 | Khalidi-Rashid-Palestinian-Identity | 🇬🇧 English |
| 6 | Palestinian-History-Calendar | 🇬🇧 English |
| 7 | The Hundred Years' War on Palestine | 🇬🇧 English |
| 8 | ga_res_1941948 | 🇬🇧 English |
| 9 | تقرير غزة الإنساني 2024 | 🇵🇸 Arabic |
| 10 | ذاكرة المكان | 🇵🇸 Arabic |
| 11 | شخصيات فلسطينية | 🇵🇸 Arabic |
| 12 | فلسطين العربية | 🇵🇸 Arabic |
| 13 | فلسطين | 🇵🇸 Arabic |
| 14 | كتاب النخبة 1 | 🇵🇸 Arabic |
| 15 | كتاب النخبة 2 | 🇵🇸 Arabic |

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **LLM** | AI Grid · `Qwen3-30B-A3B-Thinking` (OpenAI-compatible API) |
| **Orchestration** | LangGraph `StateGraph` |
| **Embeddings** | `sentence-transformers/all-MiniLM-L6-v2` (free, local) |
| **Vector Store** | ChromaDB (persistent, local `chroma_db/`) |
| **UI** | Streamlit (10 tabs) |
| **PDF Parsing** | pdfplumber (primary) + PyPDFLoader (fallback) |
| **Language Detection** | Unicode Arabic range regex |

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

### 3. Configure API keys
Create `.streamlit/secrets.toml`:
```toml
AI_API_KEY  = "your-ai-grid-api-key"
AI_MODEL    = "Qwen3-30B-A3B-Thinking"
AI_BASE_URL = "http://app.ai-grid.io:4000/v1"
```

### 4. Add your PDFs
```
palestine_chatbot/
└── data/
    ├── document1.pdf
    ├── document2.pdf
    └── ...  (15 PDFs total)
```

### 5. Build the vector database (run ONCE)
```bash
python ingest.py
```
> ⚡ Smart caching: only re-processes changed/new PDFs on subsequent runs.
> Use `--force-rebuild` to wipe and rebuild from scratch.

### 6. Launch the app
```bash
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser.

---

## 📁 Project Structure

```
palestine_chatbot/
│
├── app.py                  # Main Streamlit application (10 tabs)
├── ingest.py               # PDF ingestion pipeline
├── requirements.txt        # Python dependencies
├── README.md               # This file
│
├── data/                   # Place your 15 PDF files here
│   ├── *.pdf
│   └── ...
│
├── chroma_db/              # Auto-generated ChromaDB vector store
│   └── ...
│
├── .pdf_hashes.json        # Hash cache for incremental re-ingestion
│
└── .streamlit/
    └── secrets.toml        # API keys (DO NOT commit this file)
```

---

## 📊 Chunking Strategy

```python
RecursiveCharacterTextSplitter(
    chunk_size    = 1000,   # characters per chunk
    chunk_overlap = 200,    # overlap for context continuity
    separators    = ["\n\n", "\n", ".", "،", " ", ""]
)
```

Every chunk carries **mandatory metadata**:
```python
{
    "document_title": "Khalidi-Rashid-Palestinian-Identity",  # MANDATORY
    "page_number"   : 42,                                      # MANDATORY
    "source"        : "/absolute/path/to/file.pdf",
    "language"      : "en",   # or "ar"
    "chunk_size"    : 847,
}
```

---

## 🖥️ The 10 Tabs

| Tab | Feature |
|---|---|
| 💬 **Smart Chat** | Full Agentic RAG with workflow trace + cited answers |
| 🔍 **Discourse Analysis** | Bias & propaganda detection |
| ⚖️ **Compare Documents** | Side-by-side topic comparison |
| 📄 **Document Summary** | Auto-summary per document (bilingual) |
| 🗺️ **Interactive Map** | Folium map of key Palestinian locations |
| 📅 **Historical Timeline** | 1917 → 2024 key events |
| ☁️ **Word Cloud** | Generated from your actual corpus |
| 📊 **Statistics** | Chunks, languages, response times |
| 📤 **Upload PDF** ⭐ | Live indexing + immediate Q&A (Secret Test — 15pts) |
| ℹ️ **About** | Architecture + Mermaid diagram |

---

## 🔒 Secret Test (Upload PDF — 15 Points)

The Upload tab implements a 3-phase live ingestion test:

**Phase 1 — Upload & Index**
- PDF is saved to a temp file
- Text extracted via pdfplumber (Arabic) or PyPDFLoader (fallback)
- Chunked with mandatory `document_title` + `page_number` metadata
- Added to live ChromaDB — **no app restart required**

**Phase 2 — Automatic Verification**
- Immediately queries ChromaDB for the new document
- Shows retrieved chunk as proof of successful indexing

**Phase 3 — Live Q&A**
- Full 5-node LangGraph pipeline runs against the new document
- Bilingual cited answers with page references

---

## ⚙️ Ingestion Pipeline Commands

```bash
# Normal run (skips unchanged PDFs)
python ingest.py

# Force full rebuild
python ingest.py --force-rebuild
```

Output example:
```
00:09:07  INFO  → Khalidi-Rashid-Palestinian-Identity.pdf   pages=312  chunks=1847
00:09:17  INFO  → فلسطين.pdf                                pages=489  chunks=2103
...
✅  Ingestion complete. ChromaDB saved to: chroma_db/
```

---

## 🛡️ Anti-Hallucination Design

```
If retrieved chunks do NOT contain the answer:
    Node 4 (Grade) → grade_passed = False
    Route → Fallback node
    Response = "not found in documents"  (exact string, no LLM call)

If retrieved chunks DO contain the answer:
    Node 5 (Generate) → produces answer
    MUST end with: (Source: [Title], Page: [N])
```

---

## 👥 Team

Built with ❤️ for **Palestine** — leveraging AI for truth, memory, and justice.

> *"To exist is to resist."*

---

## 📄 License

MIT License — Free to use, share, and build upon.

---

<div align="center">
<b>AI for Palestine Competition 2024</b><br>
🕌 Smart Library · Agentic RAG · Bilingual · Open Source
</div>
