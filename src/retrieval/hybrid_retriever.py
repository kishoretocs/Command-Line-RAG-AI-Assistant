from typing import List, Dict, Any, Tuple
from src.config import RETRIEVAL_TOP_K, RERANK_TOP_K, RRF_K
from src.retrieval.vector_store import VectorStore
from src.retrieval.bm25_search import BM25Search
from src.retrieval.reranker import CrossEncoderReranker

class HybridRetriever:
    def __init__(self):
        self.vector_store = VectorStore()
        self.bm25_search = BM25Search()
        self.reranker = CrossEncoderReranker()

    def rrf_fuse(
        self,
        dense_hits: List[Dict[str, Any]],
        sparse_hits: List[Dict[str, Any]],
        k: int = RRF_K,
        top_n: int = RETRIEVAL_TOP_K
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Combines dense and sparse results using Reciprocal Rank Fusion:
        RRF_score(d) = sum(1 / (k + rank))
        Returns:
          1. Fused candidate list
          2. Full log of evaluated candidates
        """
        scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}
        dense_ranks: Dict[str, int] = {}
        sparse_ranks: Dict[str, int] = {}

        for rank, hit in enumerate(dense_hits):
            cid = hit["chunk_id"]
            dense_ranks[cid] = rank + 1
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            chunk_map[cid] = hit

        for rank, hit in enumerate(sparse_hits):
            cid = hit["chunk_id"]
            sparse_ranks[cid] = rank + 1
            scores[cid] = scores.get(cid, 0.0) + (1.0 / (k + rank + 1))
            if cid not in chunk_map:
                chunk_map[cid] = hit

        sorted_cids = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)

        candidates: List[Dict[str, Any]] = []
        chunks_log: List[Dict[str, Any]] = []

        for cid in sorted_cids:
            chunk = chunk_map[cid]
            rrf_score = scores[cid]
            d_rank = dense_ranks.get(cid)
            s_rank = sparse_ranks.get(cid)

            log_entry = {
                "chunk_id": cid,
                "document_id": chunk.get("document_id", ""),
                "section_number": chunk.get("metadata", {}).get("section_number", ""),
                "dense_rank": d_rank,
                "sparse_rank": s_rank,
                "rrf_score": rrf_score,
                "cross_encoder_score": None,
                "is_selected": False
            }
            chunks_log.append(log_entry)

            if len(candidates) < top_n:
                candidates.append(chunk)

        return candidates, chunks_log

    def retrieve_and_rerank(
        self,
        query: str,
        retrieval_k: int = RETRIEVAL_TOP_K,
        rerank_k: int = RERANK_TOP_K
    ) -> Tuple[List[Dict[str, Any]], float, List[Dict[str, Any]]]:
        """
        Executes end-to-end single-level section retrieval for a query:
        1. Dense semantic search on section chunks
        2. Sparse keyword search on section chunks
        3. RRF Fusion
        4. Cross-Encoder reranking on section chunks directly

        Returns:
          1. top_sections: List of top reranked section chunks
          2. top_score: Highest Cross-Encoder score
          3. chunks_log: Granular log of evaluated chunks for SQLite tracing
        """
        dense_hits = self.vector_store.search(query, top_k=retrieval_k)
        sparse_hits = self.bm25_search.search(query, top_k=retrieval_k)

        candidates, chunks_log = self.rrf_fuse(dense_hits, sparse_hits, top_n=retrieval_k)

        if not candidates:
            return [], -10.0, chunks_log

        # Rerank candidates with Cross-Encoder directly
        scored_sections = self.reranker.rerank(query, candidates, top_n=rerank_k)

        top_score = scored_sections[0][1] if scored_sections else -10.0
        top_sections = [section for section, _ in scored_sections]
        selected_ids = {section["chunk_id"] for section in top_sections}

        # Update scores and selection flags in chunks_log
        score_map = {sec["chunk_id"]: score for sec, score in scored_sections}

        for entry in chunks_log:
            cid = entry["chunk_id"]
            if cid in score_map:
                entry["cross_encoder_score"] = score_map[cid]
            if cid in selected_ids:
                entry["is_selected"] = True

        return top_sections, top_score, chunks_log
