import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

from src.config import (
    CORPUS_DIR,
    SECTION_STORE_PATH,
    BM25_INDEX_PATH,
    CHROMA_DIR,
    EMBEDDING_MODEL_NAME
)
from src.ingestion.manifest_loader import ManifestLoader
from src.ingestion.pdf_parser import PDFParser
from src.ingestion.chunker import SectionChunker
from src.observability.tracer import Tracer

class IngestionPipeline:
    def __init__(self):
        self.manifest_loader = ManifestLoader()
        self.pdf_parser = PDFParser()
        self.chunker = SectionChunker()

    def run(self) -> Dict[str, Any]:
        """
        Executes full ingestion:
        1. Loads manifest and computes supersession status
        2. Parses all 13 PDFs
        3. Generates single-level section chunks
        4. Indexes section chunks into ChromaDB and BM25
        """
        print("=== Starting Cerulean Systems Ingestion Pipeline ===")
        
        manifest_map = self.manifest_loader.load()
        print(f"Loaded {len(manifest_map)} documents from manifest.")

        all_section_chunks: List[Dict[str, Any]] = []

        for doc_id, doc_meta in tqdm(manifest_map.items(), desc="Parsing & Chunking PDFs"):
            file_name = doc_meta["file"]
            pdf_path = CORPUS_DIR / file_name

            if not pdf_path.exists():
                print(f"WARNING: File {pdf_path} does not exist. Skipping.")
                continue

            try:
                pages = self.pdf_parser.parse_pdf(pdf_path)
                section_chunks = self.chunker.chunk_document(doc_meta, pages)
                all_section_chunks.extend(section_chunks)
            except Exception as e:
                Tracer.log_error("INGESTION", e, context_payload={"file": file_name, "doc_id": doc_id})
                print(f"ERROR processing {file_name}: {e}")

        # 0. Save section store (used by eval numeric-grounding checks and debugging)
        SECTION_STORE_PATH.parent.mkdir(exist_ok=True, parents=True)
        with open(SECTION_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump({c["chunk_id"]: c for c in all_section_chunks}, f, indent=2, ensure_ascii=False)
        print(f"[OK] Saved {len(all_section_chunks)} section chunks to {SECTION_STORE_PATH}")

        # 1. Index into ChromaDB
        self._index_chroma(all_section_chunks)

        # 2. Index into BM25
        self._index_bm25(all_section_chunks)

        summary = {
            "total_documents": len(manifest_map),
            "total_section_chunks": len(all_section_chunks),
            "section_store_path": str(SECTION_STORE_PATH),
            "chroma_dir": str(CHROMA_DIR),
            "bm25_path": str(BM25_INDEX_PATH)
        }
        print("\n=== Ingestion Complete ===")
        print(json.dumps(summary, indent=2))
        return summary

    def _index_chroma(self, chunks: List[Dict[str, Any]]):
        import chromadb
        from sentence_transformers import SentenceTransformer

        print(f"\nEmbedding {len(chunks)} section chunks with {EMBEDDING_MODEL_NAME}...")
        embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        texts = [c["context_injected_text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True)

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        # Reset or create collection
        try:
            client.delete_collection("cerulean_sections")
        except Exception:
            pass

        collection = client.create_collection(
            name="cerulean_sections",
            metadata={"hnsw:space": "cosine"}
        )

        ids = [c["chunk_id"] for c in chunks]
        metadatas = [
            {
                "chunk_id": c["chunk_id"],
                "document_id": c["document_id"],
                "section_number": c["metadata"]["section_number"],
                "section_title": c["metadata"]["section_title"],
                "page": c["metadata"]["page"],
                "version": c["metadata"]["version"],
                "effective_date": c["metadata"]["effective_date"],
                "is_superseded": c["metadata"]["is_superseded"],
                "token_count": c["token_count"]
            }
            for c in chunks
        ]

        # Add to Chroma in batches of 100
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            collection.add(
                ids=ids[i:i+batch_size],
                embeddings=embeddings[i:i+batch_size].tolist(),
                documents=texts[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size]
            )
        print(f"[OK] Indexed {len(chunks)} section vectors into ChromaDB at {CHROMA_DIR}")

    def _index_bm25(self, chunks: List[Dict[str, Any]]):
        from rank_bm25 import BM25Okapi

        print("Building BM25 sparse keyword index...")
        tokenized_corpus = [c["context_injected_text"].lower().split() for c in chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        bm25_data = {
            "bm25": bm25,
            "chunk_ids": [c["chunk_id"] for c in chunks],
            "chunks": chunks
        }

        with open(BM25_INDEX_PATH, "wb") as f:
            pickle.dump(bm25_data, f)
        print(f"[OK] Saved BM25 index to {BM25_INDEX_PATH}")


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()
