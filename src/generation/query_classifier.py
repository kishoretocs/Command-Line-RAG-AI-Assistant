import json
import re
from dataclasses import dataclass, field
from typing import List, Optional

from src.generation.llm_client import LLMClient
from src.observability.prompt_store import PromptStore


@dataclass
class QueryIntent:
    intent: str  # "IN_DOMAIN" | "META_ATTACK" | "UNSAFE_BYPASS" | "AMBIGUOUS" | "OUT_OF_DOMAIN"
    reasoning: str = ""
    candidate_domains: List[str] = field(default_factory=list)


class QueryClassifier:
    VALID_INTENTS = {
        "IN_DOMAIN",
        "META_ATTACK",
        "UNSAFE_BYPASS",
        "AMBIGUOUS",
        "OUT_OF_DOMAIN",
    }

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def classify(self, query: str) -> QueryIntent:
        """
        Classifies a user query into one of the intent categories using a single LLM call.
        Falls back conservatively if the LLM call or JSON parsing fails.
        """
        try:
            prompt_data = PromptStore.get_active_prompt("QUERY_CLASSIFIER")
            template = prompt_data["template_text"] if prompt_data else self._default_template()
            prompt = template.replace("{query}", query)
            # Use the generation model (not fast) for safety-critical classification —
            # the fast model returns empty/incorrect responses on adversarial queries (e.g., vendor approval injection).
            response = self.llm_client.generate(prompt)

            # Parse JSON (strip potential markdown code block wrappers)
            clean_json = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_json)

            intent = str(data.get("intent", "IN_DOMAIN")).strip().upper()
            reasoning = str(data.get("reasoning", ""))
            domains = data.get("candidate_domains", []) or []

            # Validate intent against allowed set
            if intent not in self.VALID_INTENTS:
                return self._fallback_intent(query, "Classifier returned an invalid intent")

            return QueryIntent(
                intent=intent,
                reasoning=reasoning,
                candidate_domains=domains if isinstance(domains, list) else []
            )

        except Exception as exc:
            return self._fallback_intent(query, f"Classifier unavailable: {type(exc).__name__}")

    @staticmethod
    def _fallback_intent(query: str, reason: str) -> QueryIntent:
        """Fail closed when the classifier cannot reliably determine intent."""
        normalized = query.lower().strip()

        if re.search(
            r"(?:system prompt|system instructions|hidden instructions|hidden prompt|"
            r"secret prompt|reveal your prompt|repeat your instructions|ignore previous instructions)",
            normalized,
        ):
            return QueryIntent(intent="META_ATTACK", reasoning=f"{reason}; matched meta-request safeguard")

        if re.search(
            r"(?:bypass|circumvent|override|evade|skip).*(?:approval|policy|control|security|limit)|"
            r"(?:approve|authorize).*(?:without|no).*(?:approval|check)",
            normalized,
        ):
            return QueryIntent(intent="UNSAFE_BYPASS", reasoning=f"{reason}; matched safety safeguard")

        return QueryIntent(intent="OUT_OF_DOMAIN", reasoning=f"{reason}; fail-closed fallback")

    @staticmethod
    def _default_template() -> str:
        return """Classify the user query into one of: IN_DOMAIN, META_ATTACK, UNSAFE_BYPASS, AMBIGUOUS, OUT_OF_DOMAIN.
Respond in JSON: {"intent": "...", "reasoning": "...", "candidate_domains": [...]}

User Query: {query}"""