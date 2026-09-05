from typing import Optional
from src.generation.llm_client import LLMClient
from src.observability.prompt_store import PromptStore

class HyDEGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_hypothetical_passage(self, query: str) -> Optional[str]:
        """
        Generates a concise hypothetical policy or procedure excerpt that answers the query
        using formal corporate terminology.
        """
        try:
            prompt_data = PromptStore.get_active_prompt("HYDE_SYNTHESIS")
            template = (
                prompt_data["template_text"]
                if prompt_data
                else "Draft a hypothetical corporate policy excerpt answering: {query}"
            )

            prompt = template.replace("{query}", query)
            hypothetical_text = self.llm_client.generate_fast(prompt)
            return hypothetical_text.strip()
        except Exception as e:
            print(f"HyDE generation error: {e}")
            return None
