"""
=============================================================================
  AI for Palestine Smart Library — PDF Ingestion Pipeline
  Author  : Expert AI Engineer
  Purpose : Read 15 PDFs (Arabic + English), chunk them with full metadata,
            embed via sentence-transformers, and persist to ChromaDB.
            The DB is ONLY rebuilt when new/modified PDFs are detected,
            saving time on every subsequent run.
=============================================================================
"""

import os
import re
import hashlib
import json
import logging
import unicodedata
from pathlib import Path
from typing import List, Dict, Any, Optional

# ── PDF extraction ──────────────────────────────────────────────────────────
try:
    import pdfplumber          # primary — handles Arabic text better
    _USE_PDFPLUMBER = True
except ImportError:
    import PyPDF2              # fallback
    _USE_PDFPLUMBER = False

# ── LangChain ───────────────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ── Embeddings ──────────────────────────────────────────────────────────────
from langchain_huggingface import HuggingFaceEmbeddings

# ── Vector store ────────────────────────────────────────────────────────────
from langchain_chroma import Chroma
import chromadb

# ── Logging setup ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION  (edit freely)
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR          = Path("data")          # folder containing your 15 PDFs
CHROMA_DIR        = Path("chroma_db")     # persisted vector DB
COLLECTION_NAME   = "smart_library"
HASH_CACHE_FILE   = Path(".pdf_hashes.json")   # tracks already-ingested files

EMBED_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DEVICE      = "cpu"                 # change to "cuda" if GPU available

CHUNK_SIZE        = 1000                  # characters per chunk
CHUNK_OVERLAP     = 200                   # overlap between adjacent chunks

# Arabic + Latin separators for the recursive splitter
SEPARATORS = [
    "\n\n",      # paragraph
    "\n",        # newline
    "。", ".", # sentence enders
    "؟", "?",
    "!", "！",
    "؛", ";",
    "،", ",",
    " ", "",     # word / character fallback
]

# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: Compute a stable MD5 for a file (used to detect changes)
# ─────────────────────────────────────────────────────────────────────────────
def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: Load / save the hash cache
# ─────────────────────────────────────────────────────────────────────────────
def _load_hash_cache() -> Dict[str, str]:
    if HASH_CACHE_FILE.exists():
        with open(HASH_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_hash_cache(cache: Dict[str, str]) -> None:
    with open(HASH_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER: Clean extracted text (handles RTL artefacts, ligatures, etc.)
# ─────────────────────────────────────────────────────────────────────────────
_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")

def _clean_text(text: str) -> str:
    # Normalise Unicode (NFC is safest for Arabic)
    text = unicodedata.normalize("NFC", text)
    # Remove null bytes and non-printable control chars (keep \n, \t)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    # Collapse excessive whitespace / blank lines
    text = re.sub(r" {3,}", "  ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _detect_language(text: str) -> str:
    """Heuristic: if >20 % of chars are Arabic → mark as Arabic."""
    if not text:
        return "en"
    arabic_chars = len(_ARABIC_RANGE.findall(text))
    ratio = arabic_chars / max(len(text), 1)
    return "ar" if ratio > 0.20 else "en"


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Extract pages from a single PDF
# ─────────────────────────────────────────────────────────────────────────────
def extract_pages_pdfplumber(pdf_path: Path) -> List[Dict[str, Any]]:
    """Return a list of dicts: {page_number, text} — 1-indexed."""
    pages = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                text = _clean_text(raw)
                if text:
                    pages.append({"page_number": i, "text": text})
    except Exception as exc:
        log.warning("pdfplumber failed on %s: %s — skipping", pdf_path.name, exc)
    return pages


def extract_pages_pypdf2(pdf_path: Path) -> List[Dict[str, Any]]:
    """Fallback using PyPDF2."""
    pages = []
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                raw = page.extract_text() or ""
                text = _clean_text(raw)
                if text:
                    pages.append({"page_number": i, "text": text})
    except Exception as exc:
        log.warning("PyPDF2 failed on %s: %s — skipping", pdf_path.name, exc)
    return pages


def extract_pages(pdf_path: Path) -> List[Dict[str, Any]]:
    if _USE_PDFPLUMBER:
        pages = extract_pages_pdfplumber(pdf_path)
        if not pages:          # fallback even when pdfplumber is installed
            try:
                import PyPDF2
                pages = extract_pages_pypdf2(pdf_path)
            except ImportError:
                pass
    else:
        pages = extract_pages_pypdf2(pdf_path)
    return pages


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Chunk pages and attach mandatory metadata
# ─────────────────────────────────────────────────────────────────────────────
def build_documents(pdf_path: Path) -> List[Document]:
    """
    Converts one PDF into a list of LangChain Document objects.
    Every Document carries mandatory metadata:
        - document_title : clean stem of the filename
        - page_number    : 1-indexed page where the chunk begins
        - source         : absolute path to the PDF
        - language       : 'ar' or 'en'
        - chunk_index    : sequential chunk index within the document
    """
    doc_title = pdf_path.stem          # filename without extension
    pages     = extract_pages(pdf_path)

    if not pages:
        log.warning("No extractable text found in: %s", pdf_path.name)
        return []

    splitter = RecursiveCharacterTextSplitter(
        separators        = SEPARATORS,
        chunk_size        = CHUNK_SIZE,
        chunk_overlap     = CHUNK_OVERLAP,
        length_function   = len,
        is_separator_regex= False,
        keep_separator    = True,          # keep delimiters for readability
    )

    documents: List[Document] = []
    chunk_index = 0

    for page_data in pages:
        page_num = page_data["page_number"]
        page_text = page_data["text"]
        lang = _detect_language(page_text)

        # Split the page text into chunks
        chunks = splitter.split_text(page_text)

        for chunk_text in chunks:
            if not chunk_text.strip():
                continue

            metadata = {
                # ── MANDATORY competition fields ──────────────────────────
                "document_title": doc_title,
                "page_number"   : page_num,
                # ── Extra enrichment ──────────────────────────────────────
                "source"        : str(pdf_path.resolve()),
                "file_name"     : pdf_path.name,
                "language"      : lang,
                "chunk_index"   : chunk_index,
                "chunk_size"    : len(chunk_text),
            }

            documents.append(Document(page_content=chunk_text, metadata=metadata))
            chunk_index += 1

    log.info(
        "  %-55s  pages=%3d  chunks=%4d",
        pdf_path.name[:55], len(pages), len(documents)
    )
    return documents


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Build / update ChromaDB
# ─────────────────────────────────────────────────────────────────────────────
def get_embedding_function() -> HuggingFaceEmbeddings:
    log.info("Loading embedding model: %s  (device=%s)", EMBED_MODEL, EMBED_DEVICE)
    return HuggingFaceEmbeddings(
        model_name      = EMBED_MODEL,
        model_kwargs    = {"device": EMBED_DEVICE},
        encode_kwargs   = {"normalize_embeddings": True, "batch_size": 64},
    )


def build_or_update_vectorstore(
    new_docs   : List[Document],
    embeddings : HuggingFaceEmbeddings,
    *,
    reset      : bool = False,
) -> Chroma:
    """
    Persist `new_docs` into ChromaDB.
    If `reset=True` the entire collection is wiped first.
    """
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
            log.info("Existing collection deleted (reset=True).")
        except Exception:
            pass

    vectorstore = Chroma(
        client              = client,
        collection_name     = COLLECTION_NAME,
        embedding_function  = embeddings,
    )

    if new_docs:
        log.info("Adding %d chunks to ChromaDB …", len(new_docs))
        # Add in batches of 500 to avoid memory spikes
        batch_size = 500
        for i in range(0, len(new_docs), batch_size):
            batch = new_docs[i : i + batch_size]
            vectorstore.add_documents(batch)
            log.info("  … committed batch %d/%d", i // batch_size + 1,
                     -(-len(new_docs) // batch_size))

    total = vectorstore._collection.count()
    log.info("ChromaDB collection '%s' now contains %d vectors.", COLLECTION_NAME, total)
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ENTRYPOINT
# ─────────────────────────────────────────────────────────────────────────────
def main(force_rebuild: bool = False) -> Chroma:
    """
    Orchestrates the full ingestion pipeline.

    Args:
        force_rebuild: If True, wipe ChromaDB and re-ingest everything.

    Returns:
        A ready-to-query Chroma vectorstore instance.
    """
    log.info("=" * 70)
    log.info("  AI for Palestine Smart Library — Ingestion Pipeline")
    log.info("=" * 70)

    # ── Discover PDFs ────────────────────────────────────────────────────────
    pdf_files: List[Path] = sorted(DATA_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in '{DATA_DIR.resolve()}'")
    log.info("Found %d PDF file(s) in '%s'.", len(pdf_files), DATA_DIR)

    # ── Decide which PDFs need processing (hash-based change detection) ──────
    hash_cache = _load_hash_cache()
    to_process: List[Path] = []

    for pdf in pdf_files:
        current_hash = _file_hash(pdf)
        cached_hash  = hash_cache.get(pdf.name)
        if force_rebuild or cached_hash != current_hash:
            to_process.append(pdf)
        else:
            log.info("  SKIP (unchanged): %s", pdf.name)

    if not to_process and not force_rebuild:
        log.info("All PDFs are already indexed. Loading existing vectorstore …")
        embeddings  = get_embedding_function()
        client      = chromadb.PersistentClient(path=str(CHROMA_DIR))
        vectorstore = Chroma(
            client             = client,
            collection_name    = COLLECTION_NAME,
            embedding_function = embeddings,
        )
        log.info("Vectorstore loaded. Total vectors: %d",
                 vectorstore._collection.count())
        return vectorstore

    # ── Extract + Chunk ──────────────────────────────────────────────────────
    log.info("\nProcessing %d PDF(s) …\n", len(to_process))
    all_docs: List[Document] = []

    for pdf_path in to_process:
        log.info("→  %s", pdf_path.name)
        docs = build_documents(pdf_path)
        all_docs.extend(docs)

    log.info("\nTotal new chunks to embed: %d", len(all_docs))

    # ── Embed + Persist ──────────────────────────────────────────────────────
    embeddings  = get_embedding_function()
    vectorstore = build_or_update_vectorstore(
        new_docs   = all_docs,
        embeddings = embeddings,
        reset      = force_rebuild,
    )

    # ── Update hash cache so these files are skipped next run ────────────────
    for pdf in to_process:
        hash_cache[pdf.name] = _file_hash(pdf)
    _save_hash_cache(hash_cache)
    log.info("Hash cache updated → %s", HASH_CACHE_FILE)

    # ── Quick sanity check ───────────────────────────────────────────────────
    log.info("\n── Sanity check: sample retrieval ──────────────────────────────")
    sample_results = vectorstore.similarity_search(
        "Palestine history", k=3
    )
    for idx, doc in enumerate(sample_results, 1):
        m = doc.metadata
        log.info(
            "  [%d] title=%-40s  page=%s  lang=%s",
            idx, m.get("document_title", "?")[:40],
            m.get("page_number", "?"),
            m.get("language", "?"),
        )

    log.info("\n✅  Ingestion complete. ChromaDB saved to: %s", CHROMA_DIR.resolve())
    return vectorstore


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest PDFs into ChromaDB for the Smart Library RAG pipeline."
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Wipe the existing ChromaDB collection and re-ingest all PDFs.",
    )
    args = parser.parse_args()

    vectorstore = main(force_rebuild=args.force_rebuild)

    # ── Optional: print collection statistics ────────────────────────────────
    print("\n" + "=" * 60)
    print("  COLLECTION STATISTICS")
    print("=" * 60)
    collection = vectorstore._collection
    total      = collection.count()
    print(f"  Total vectors : {total:,}")

    # Count per document
    results = collection.get(include=["metadatas"])
    title_counts: Dict[str, int] = {}
    lang_counts : Dict[str, int] = {}
    for meta in results["metadatas"]:
        t = meta.get("document_title", "unknown")
        l = meta.get("language", "?")
        title_counts[t] = title_counts.get(t, 0) + 1
        lang_counts[l]  = lang_counts.get(l, 0) + 1

    print(f"\n  By language  : {lang_counts}")
    print(f"\n  By document  :")
    for title, cnt in sorted(title_counts.items(), key=lambda x: -x[1]):
        print(f"    {cnt:>5}  {title}")
    print("=" * 60)
