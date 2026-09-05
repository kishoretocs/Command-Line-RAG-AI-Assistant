import pickle
from pathlib import Path
from typing import List, Dict, Any
from src.config import BM25_INDEX_PATH, BM25_MIN_RELATIVE_SCORE, RETRIEVAL_TOP_K

class BM25Search:
    _instance = None
    _bm25_data = None

    def __init__(self, index_path: Path = BM25_INDEX_PATH):
        self.index_path = index_path
        if BM25Search._bm25_data is None:
            if not self.index_path.exists():
                raise FileNotFoundError(f"BM25 index not found at {self.index_path}. Run ingestion first.")
            with open(self.index_path, "rb") as f:
                BM25Search._bm25_data = pickle.load(f)

        self.bm25 = BM25Search._bm25_data["bm25"]
        self.chunk_ids = BM25Search._bm25_data["chunk_ids"]
        self.chunks = BM25Search._bm25_data["chunks"]

    def search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict[str, Any]]:
        """
        Runs sparse keyword search using BM25.
        Returns candidate child chunks with BM25 score.
        """
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        if not scored_indices or scores[scored_indices[0]] <= 0:
            return []

        best_score = float(scores[scored_indices[0]])
        minimum_score = best_score * BM25_MIN_RELATIVE_SCORE
        scored_indices = [
            idx for idx in scored_indices
            if float(scores[idx]) >= minimum_score
        ]

        hits: List[Dict[str, Any]] = []
        for rank, idx in enumerate(scored_indices):
            chunk = self.chunks[idx]
            hits.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "section_number": chunk.get("section_number", ""),
                "section_title": chunk.get("section_title", ""),
                "score": float(scores[idx]),
                "text": chunk["context_injected_text"],
                "metadata": chunk["metadata"]
            })

        return hits
