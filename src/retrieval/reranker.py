import math
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from src.config import RERANKER_MODEL_NAME

class CrossEncoderReranker:
    _instance = None
    _model = None

    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        if CrossEncoderReranker._model is None:
            CrossEncoderReranker._model = CrossEncoder(model_name)
        self.model = CrossEncoderReranker._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_n: int = 3
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Scores (query, chunk_text) pairs using the Cross-Encoder.
        Applies sigmoid normalization to convert raw logits to probabilities [0.0, 1.0].
        Returns sorted list of (chunk_dict, prob_score) tuples.
        """
        if not candidates:
            return []

        pairs = [[query, c["text"]] for c in candidates]
        raw_scores = self.model.predict(pairs)

        scored_candidates: List[Tuple[Dict[str, Any], float]] = []
        for idx, score in enumerate(raw_scores):
            # Sigmoid probability normalization
            prob = 1.0 / (1.0 + math.exp(-float(score)))
            scored_candidates.append((candidates[idx], prob))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        if top_n:
            return scored_candidates[:top_n]
        return scored_candidates
