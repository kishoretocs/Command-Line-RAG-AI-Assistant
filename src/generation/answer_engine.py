import json
import re
import time
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from src.config import CONFIDENCE_THRESHOLD, SYSTEM_CANARY
from src.retrieval.hybrid_retriever import HybridRetriever
from src.generation.llm_client import LLMClient
from src.generation.query_classifier import QueryClassifier, QueryIntent
from src.generation.hyde import HyDEGenerator
from src.generation.context_builder import ContextBuilder
from src.observability.prompt_store import PromptStore
from src.observability.tracer import Tracer


class Citation(BaseModel):
    document_id: str
    section: Optional[str] = None
    page: Optional[int] = None
    version: Optional[str] = None


class AnswerResult(BaseModel):
    query_id: str
    question: str
    behavior: str  # "ANSWER", "ANSWER_WITH_CONFLICT", "CLARIFY", "REFUSE_NO_INFO", "DECLINE_UNSAFE", "DECLINE_META"
    answer: str
    citations: List[Citation]
    retrieved_section_ids: List[str]
    top_score: float
    hyde_triggered: bool
    latency_ms: int
    canary_leaked: bool


class AnswerEngine:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm_client = LLMClient()
        self.query_classifier = QueryClassifier(self.llm_client)
        self.hyde_generator = HyDEGenerator(self.llm_client)

    def answer_question(self, query: str) -> AnswerResult:
        start_time = time.time()
        query_id = Tracer.generate_query_id()

        # Step 1: Single LLM classification (replaces old regex steps 0, 0.5, 0.8 and domain gate)
        intent = self.query_classifier.classify(query)

        # Handle non-IN_DOMAIN intents with a single reusable method
        if intent.intent != "IN_DOMAIN":
            return self._handle_classified_intent(query, query_id, start_time, intent)

        # Step 2: Pass 1 Standard Hybrid Retrieval
        retrieval_start = time.time()
        top_sections, top_score, chunks_log = self.retriever.retrieve_and_rerank(query)
        for c in chunks_log:
            c["pass_type"] = "PASS_1"

        hyde_triggered = False
        hyde_passage = None
        pass2_top_score = None

        # Step 3: Pass 2 HyDE Fallback if confidence < threshold
        if top_score < CONFIDENCE_THRESHOLD:
            hyde_triggered = True
            hyde_passage = self.hyde_generator.generate_hypothetical_passage(query)

            if hyde_passage:
                hyde_sections, pass2_top_score, hyde_chunks_log = self.retriever.retrieve_and_rerank(hyde_passage)
                for c in hyde_chunks_log:
                    c["pass_type"] = "HYDE_PASS_2"
                chunks_log.extend(hyde_chunks_log)

                if pass2_top_score > top_score:
                    top_sections = hyde_sections
                    top_score = pass2_top_score

        retrieval_elapsed = int((time.time() - retrieval_start) * 1000)

        # Log retrieval evaluation chunks to SQLite
        Tracer.log_retrieval_chunks(query_id, chunks_log)

        # Step 4: Final Confidence / Sufficiency Gate
        if top_score < CONFIDENCE_THRESHOLD or not top_sections:
            return self._refuse_no_info(query, query_id, start_time, retrieval_elapsed,
                                        intent, hyde_triggered, hyde_passage, pass2_top_score,
                                        top_score, top_sections, chunks_log)

        # Step 5: Assemble Context & Citations
        context_str, available_citations = ContextBuilder.build_context(top_sections)

        # Step 6: LLM Generation
        gen_start = time.time()
        sys_prompt_data = PromptStore.get_active_prompt("SYSTEM_GENERATION")
        system_prompt = sys_prompt_data["template_text"] if sys_prompt_data else ""

        user_prompt = (
            f"Context Information from Cerulean Systems Knowledge Base:\n{context_str}\n\n"
            f"User Question: {query}\n\n"
            f"Answer using the JSON contract in the system prompt. Cite Document ID, section, and version inline in the answer. "
            f"If documents conflict or have different dates, state both and explain which is current as of the configured baseline date. "
            f"If the context does not contain the answer, use behavior REFUSE_NO_INFO and say so explicitly."
        )

        llm_raw_response = self.llm_client.generate(prompt=user_prompt, system_prompt=system_prompt)
        gen_elapsed = int((time.time() - gen_start) * 1000)
        total_elapsed = int((time.time() - start_time) * 1000)

        # Parse JSON response from LLM
        behavior = "ANSWER"
        llm_response = llm_raw_response
        cited_docs_from_json = []

        try:
            clean_json = llm_raw_response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)
            behavior = data.get("behavior", "ANSWER")
            llm_response = data.get("answer", llm_raw_response)
            cited_docs_from_json = data.get("cited_document_ids", [])
        except Exception:
            # Fallback if LLM output was plain text markdown
            lowered_ans = llm_raw_response.lower()
            refusal_markers = (
                "does not contain", "do not contain", "no information", "not disclosed",
                "does not mention", "do not mention", "cannot be determined",
                "cannot determine", "not available in", "does not include",
                "no details", "not specified in", "not provided in"
            )
            if any(marker in lowered_ans for marker in refusal_markers):
                behavior = "REFUSE_NO_INFO"
            else:
                # Conflicts are expressed via dual citations, not a special label
                behavior = "ANSWER"

        # Extract cited document IDs
        cited_docs = set(re.findall(r"\b([A-Z]{2,4}-[A-Z]{2,3}-\d{3,4}|SALES-PL-\d{4})\b", llm_response))
        if cited_docs_from_json:
            cited_docs.update(cited_docs_from_json)

        final_citations = [
            Citation(document_id=doc_id)
            for doc_id in cited_docs
        ]
        if not final_citations:
            # Fall back to documents retrieved
            final_citations = [
                Citation(document_id=s["document_id"], section=str(s.get("section_number", "")))
                for s in top_sections
            ]

        # Log trace in SQLite
        Tracer.log_query_trace(
            query_id=query_id,
            user_query=query,
            system_prompt_id=sys_prompt_data["prompt_id"] if sys_prompt_data else "system_gen_v1",
            hyde_prompt_id="hyde_synth_v1" if hyde_triggered else None,
            intent=intent.intent,
            intent_reasoning=intent.reasoning,
            pass1_top_score=top_score if not hyde_triggered else None,
            hyde_triggered=hyde_triggered,
            hyde_passage=hyde_passage,
            pass2_top_score=pass2_top_score,
            confidence_passed=True,
            retrieved_section_ids=[s["chunk_id"] for s in top_sections],
            full_assembled_prompt=user_prompt,
            llm_response=llm_response,
            citations=[c.model_dump() for c in final_citations],
            latency_total_ms=total_elapsed,
            latency_retrieval_ms=retrieval_elapsed,
            latency_generation_ms=gen_elapsed,
            status="SUCCESS"
        )

        return AnswerResult(
            query_id=query_id,
            question=query,
            behavior=behavior,
            answer=llm_response,
            citations=final_citations,
            retrieved_section_ids=[s["chunk_id"] for s in top_sections],
            top_score=top_score,
            hyde_triggered=hyde_triggered,
            latency_ms=total_elapsed,
            canary_leaked=SYSTEM_CANARY in llm_response
        )

    # ------------------------------------------------------------------
    # Single reusable handler for all non-IN_DOMAIN classification outcomes
    # ------------------------------------------------------------------

    def _handle_classified_intent(self, query: str, query_id: str, start_time: float, intent: QueryIntent) -> AnswerResult:
        """Builds the AnswerResult + trace for any non-IN_DOMAIN intent using a config table."""
        elapsed = int((time.time() - start_time) * 1000)

        # Response config per intent: behavior, answer text, top_score, trace flags
        responses = {
            "META_ATTACK": {
                "behavior": "DECLINE_META",
                "answer": "I cannot disclose internal system instructions or configuration prompts.",
                "top_score": 0.0,
                "is_in_domain": True,
                "confidence_passed": True,
                "status": "SUCCESS",
            },
            "UNSAFE_BYPASS": {
                "behavior": "DECLINE_UNSAFE",
                "answer": "I cannot assist with bypassing company approval processes or granting unauthorized policy overrides. All vendor onboarding, expenditures, and commitments must adhere strictly to established authorization procedures.",
                "top_score": 0.0,
                "is_in_domain": True,
                "confidence_passed": True,
                "status": "SUCCESS",
            },
            "AMBIGUOUS": {
                "behavior": "CLARIFY",
                "answer": self._build_clarify_message(intent),
                "top_score": 1.0,
                "is_in_domain": True,
                "confidence_passed": True,
                "status": "SUCCESS",
            },
            "OUT_OF_DOMAIN": {
                "behavior": "REFUSE_NO_INFO",
                "answer": "This question is out of scope. I only answer questions regarding Cerulean Systems policies, pricing, products, and operations.",
                "top_score": 0.0,
                "is_in_domain": False,
                "confidence_passed": False,
                "status": "REFUSAL",
            },
        }

        config = responses[intent.intent]

        result = AnswerResult(
            query_id=query_id,
            question=query,
            behavior=config["behavior"],
            answer=config["answer"],
            citations=[],
            retrieved_section_ids=[],
            top_score=config["top_score"],
            hyde_triggered=False,
            latency_ms=elapsed,
            canary_leaked=SYSTEM_CANARY in config["answer"]
        )

        Tracer.log_query_trace(
            query_id=query_id, user_query=query, system_prompt_id="system_gen_v1",
            hyde_prompt_id=None, intent=intent.intent, intent_reasoning=intent.reasoning,
            pass1_top_score=None, hyde_triggered=False, hyde_passage=None,
            pass2_top_score=None, confidence_passed=config["confidence_passed"], retrieved_section_ids=[],
            full_assembled_prompt=None, llm_response=config["answer"], citations=[],
            latency_total_ms=elapsed, latency_retrieval_ms=0, latency_generation_ms=0,
            status=config["status"]
        )
        return result

    @staticmethod
    def _build_clarify_message(intent: QueryIntent) -> str:
        """Builds a clarification message from the classifier's candidate domains."""
        if intent.candidate_domains:
            domain_list = "\n".join(f"{i+1}. {d}" for i, d in enumerate(intent.candidate_domains))
            return (
                "Your question is ambiguous as Cerulean Systems documentation defines several distinct limits/policies:\n"
                f"{domain_list}\n\n"
                "Please clarify which specific policy or limit you are inquiring about."
            )
        return (
            "Your question is ambiguous. Please clarify which specific policy, process, or limit you are inquiring about."
        )

    def _refuse_no_info(
        self,
        query: str,
        query_id: str,
        start_time: float,
        retrieval_elapsed: int,
        intent: QueryIntent,
        hyde_triggered: bool,
        hyde_passage: Optional[str],
        pass2_top_score: Optional[float],
        top_score: float,
        top_sections: List[Dict[str, Any]],
        chunks_log: List[Dict[str, Any]]
    ) -> AnswerResult:
        elapsed = int((time.time() - start_time) * 1000)
        ans = "The provided Cerulean Systems documentation does not contain enough information to answer this question."
        result = AnswerResult(
            query_id=query_id,
            question=query,
            behavior="REFUSE_NO_INFO",
            answer=ans,
            citations=[],
            retrieved_section_ids=[s["chunk_id"] for s in top_sections],
            top_score=top_score,
            hyde_triggered=hyde_triggered,
            latency_ms=elapsed,
            canary_leaked=False
        )
        Tracer.log_query_trace(
            query_id=query_id, user_query=query, system_prompt_id="system_gen_v1",
            hyde_prompt_id="hyde_synth_v1" if hyde_triggered else None,
            intent=intent.intent, intent_reasoning=intent.reasoning,
            pass1_top_score=top_score if not hyde_triggered else None,
            hyde_triggered=hyde_triggered, hyde_passage=hyde_passage,
            pass2_top_score=pass2_top_score, confidence_passed=False,
            retrieved_section_ids=[s["chunk_id"] for s in top_sections],
            full_assembled_prompt=None, llm_response=ans, citations=[],
            latency_total_ms=elapsed, latency_retrieval_ms=retrieval_elapsed,
            latency_generation_ms=0, status="REFUSAL"
        )
        return result