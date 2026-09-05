import json
import uuid
import traceback
from typing import List, Dict, Optional, Any
from src.observability.db import get_db_connection

class Tracer:
    @staticmethod
    def generate_query_id() -> str:
        return str(uuid.uuid4())

    @staticmethod
    def log_query_trace(
        query_id: str,
        user_query: str,
        system_prompt_id: Optional[str],
        hyde_prompt_id: Optional[str],
        intent: str,
        intent_reasoning: Optional[str],
        pass1_top_score: Optional[float],
        hyde_triggered: bool,
        hyde_passage: Optional[str],
        pass2_top_score: Optional[float],
        confidence_passed: bool,
        retrieved_section_ids: List[str],
        full_assembled_prompt: Optional[str],
        llm_response: Optional[str],
        citations: List[Dict[str, Any]],
        latency_total_ms: int,
        latency_retrieval_ms: int,
        latency_generation_ms: int,
        status: str = "SUCCESS"
    ):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO query_traces (
                    query_id, user_query, system_prompt_id, hyde_prompt_id,
                    intent, intent_reasoning, pass1_top_score,
                    hyde_triggered, hyde_passage, pass2_top_score,
                    confidence_passed, retrieved_section_ids, full_assembled_prompt,
                    llm_response, citations_json, latency_total_ms,
                    latency_retrieval_ms, latency_generation_ms, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                query_id, user_query, system_prompt_id, hyde_prompt_id,
                intent, intent_reasoning, pass1_top_score,
                1 if hyde_triggered else 0, hyde_passage, pass2_top_score,
                1 if confidence_passed else 0, json.dumps(retrieved_section_ids),
                full_assembled_prompt, llm_response, json.dumps(citations),
                latency_total_ms, latency_retrieval_ms, latency_generation_ms, status
            ))
            conn.commit()

    @staticmethod
    def log_retrieval_chunks(query_id: str, chunks_log: List[Dict[str, Any]]):
        if not chunks_log:
            return
        with get_db_connection() as conn:
            cursor = conn.cursor()
            rows = [
                (
                    query_id,
                    c.get("pass_type", "PASS_1"),
                    c["chunk_id"],
                    c.get("document_id", ""),
                    c.get("section_number"),
                    c.get("dense_rank"),
                    c.get("sparse_rank"),
                    c.get("rrf_score"),
                    c.get("cross_encoder_score"),
                    1 if c.get("is_selected") or c.get("is_selected_for_expansion") else 0
                )
                for c in chunks_log
            ]
            cursor.executemany("""
                INSERT INTO retrieval_chunks_log (
                    query_id, pass_type, chunk_id, document_id, section_number,
                    dense_rank, sparse_rank, rrf_score, cross_encoder_score, is_selected
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()

    @staticmethod
    def log_error(
        error_stage: str,
        error_obj: Exception,
        query_id: Optional[str] = None,
        context_payload: Optional[Dict[str, Any]] = None
    ):
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO error_logs (
                    query_id, error_stage, error_type, error_message, stack_trace, context_payload
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (
                query_id,
                error_stage,
                type(error_obj).__name__,
                str(error_obj),
                traceback.format_exc(),
                json.dumps(context_payload) if context_payload else None
            ))
            conn.commit()
