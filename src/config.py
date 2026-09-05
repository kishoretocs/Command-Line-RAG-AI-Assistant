import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CORPUS_DIR = PROJECT_ROOT / "corpus"
DATA_DIR = PROJECT_ROOT / "data"
EVAL_DIR = PROJECT_ROOT / "eval"

# Persistent storage paths
DATA_DIR.mkdir(exist_ok=True, parents=True)
CHROMA_DIR = DATA_DIR / "chroma_db"
SECTION_STORE_PATH = DATA_DIR / "section_store.json"
BM25_INDEX_PATH = DATA_DIR / "bm25_index.pkl"
SQLITE_DB_PATH = DATA_DIR / "rag_trace.db"

# Manifest and Overview
MANIFEST_PATH = CORPUS_DIR / "corpus_manifest.json"

# Models
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
RERANKER_MODEL_NAME = os.getenv("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_GENERATION_MODEL = os.getenv("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b")
GROQ_FAST_MODEL = os.getenv("GROQ_FAST_MODEL", "openai/gpt-oss-20b")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free")

# Pipeline Constraints & Thresholds
BASELINE_DATE = "2026-08-27"
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
COSINE_MIN_SCORE = float(os.getenv("COSINE_MIN_SCORE", "0.40"))
BM25_MIN_RELATIVE_SCORE = float(os.getenv("BM25_MIN_RELATIVE_SCORE", "0.30"))
RRF_K = 60
RETRIEVAL_TOP_K = 15
RERANK_TOP_K = 3

# Chunking Specifications (single-level: one section = one retrieval chunk)
MAX_SECTION_TOKENS = int(os.getenv("MAX_SECTION_TOKENS", "2000"))   # safety cap; larger sections split into parts
MIN_SECTION_TOKENS = int(os.getenv("MIN_SECTION_TOKENS", "50"))     # smaller sections merge into the previous one

# Security Canary
SYSTEM_CANARY = "CERULEAN-CANARY-9x7b2"
