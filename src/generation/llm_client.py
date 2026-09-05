import os
from typing import Optional
from src.config import GROQ_API_KEY, GROQ_GENERATION_MODEL, GROQ_FAST_MODEL

class LLMClient:
    _client = None

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")

        if self.api_key and LLMClient._client is None:
            try:
                from groq import Groq
                # Explicit timeout + limited retries so a stalled/rate-limited API
                # can never hang the pipeline indefinitely.
                LLMClient._client = Groq(api_key=self.api_key, timeout=60.0, max_retries=1)
            except Exception as e:
                print(f"Warning: Failed to initialize Groq client: {e}")

        self.client = LLMClient._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> str:
        """
        Calls Groq API to generate response. If Groq fails (e.g. Rate Limit 429),
        falls back to Openrouter API.
        """
        # Re-check key in case it was updated in .env
        if not self.client and os.getenv("GROQ_API_KEY"):
            self.api_key = os.getenv("GROQ_API_KEY")
            from groq import Groq
            LLMClient._client = Groq(api_key=self.api_key, timeout=60.0, max_retries=1)
            self.client = LLMClient._client

        if self.client:
            target_model = model or GROQ_GENERATION_MODEL
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            try:
                response = self.client.chat.completions.create(
                    model=target_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content.strip()
            except Exception as e:
                print(f"Groq API call failed ({e}), attempting fallback to OpenRouter...")

        # OpenRouter Fallback if Groq API key is missing or failed/rate-limited
        openrouter_response = self._generate_openrouter(prompt, system_prompt, temperature, max_tokens)
        if openrouter_response:
            return openrouter_response

        # Offline / Pre-configured fallback synthesizer if API keys are not available
        return self._offline_context_summary(prompt)

    def _generate_openrouter(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1500
    ) -> Optional[str]:
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
        if not openrouter_key:
            return None

        model_name = os.getenv("OPENROUTER_MODEL", "minimax/minimax-m3:free")

        # Use standard HTTP request pointing to OpenRouter API
        try:
            import urllib.request
            import json

            url = "https://openrouter.ai/api/v1/chat/completions"
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            payload_dict = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens
            }

            # Enable reasoning if using reasoning-supported models like minimax/minimax-m3:free
            if "minimax" in model_name.lower():
                payload_dict["reasoning"] = {"enabled": True}

            payload = json.dumps(payload_dict).encode("utf-8")

            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {openrouter_key}"
                }
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as ex:
            print(f"OpenRouter HTTP fallback failed: {ex}")

        return None

    def generate_fast(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.0
    ) -> str:
        """
        Uses faster model for intent gating & HyDE generation.
        """
        return self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            model=GROQ_FAST_MODEL,
            temperature=temperature,
            max_tokens=400
        )

    def _offline_context_summary(self, prompt: str) -> str:
        """Dynamic notice when running without LLM API keys configured."""
        return (
            "[Notice: GROQ_API_KEY is not configured in .env. "
            "Please add your GROQ_API_KEY to .env for full live answer synthesis.]"
        )
