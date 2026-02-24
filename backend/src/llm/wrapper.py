
from typing import Optional
import time


def _detect_provider(api_key: str) -> str:
    """Detect AI provider from API key prefix."""
    if api_key.startswith("AIza"):
        return "gemini"
    elif api_key.startswith("sk-ant-"):
        return "anthropic"
    elif api_key.startswith("sk-"):
        return "openai"
    else:
        return "gemini"  # default fallback


class LLMWrapper:
    """
    Unified LLM wrapper that auto-detects provider from API key format.

    Key prefixes:
      AIza...       -> Google Gemini (google-genai SDK)
      sk-ant-...    -> Anthropic Claude
      sk-...        -> OpenAI
    """

    def __init__(self, api_key: str, model_name: Optional[str] = None, max_retries: int = 3):
        self.api_key = api_key
        self.provider = _detect_provider(api_key)
        self.max_retries = max_retries
        self._client = None

        if self.provider == "gemini":
            self.model_name = model_name or "gemini-2.0-flash"
            self._init_gemini()
        elif self.provider == "anthropic":
            self.model_name = model_name or "claude-3-5-haiku-20241022"
            self._init_anthropic()
        elif self.provider == "openai":
            self.model_name = model_name or "gpt-4o-mini"
            self._init_openai()

    def _init_gemini(self):
        from google import genai
        self._client = genai.Client(api_key=self.api_key)

    def _init_anthropic(self):
        import anthropic
        self._client = anthropic.Anthropic(api_key=self.api_key)

    def _init_openai(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=self.api_key)

    def generate(self, prompt: str, system_instruction: Optional[str] = None, max_tokens: int = 2048) -> str:
        """Generate content using the detected provider."""
        for attempt in range(self.max_retries):
            try:
                if self.provider == "gemini":
                    return self._generate_gemini(prompt, system_instruction, max_tokens)
                elif self.provider == "anthropic":
                    return self._generate_anthropic(prompt, system_instruction, max_tokens)
                elif self.provider == "openai":
                    return self._generate_openai(prompt, system_instruction, max_tokens)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"[{self.provider}] API failed after {self.max_retries} retries: {e}")
                    raise
                time.sleep(2 ** attempt)
        return ""

    def _generate_gemini(self, prompt: str, system_instruction: Optional[str], max_tokens: int) -> str:
        from google.genai import types
        contents = prompt
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            temperature=0.1,
        )
        if system_instruction:
            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=0.1,
                system_instruction=system_instruction,
            )
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )
        return response.text or ""

    def _generate_anthropic(self, prompt: str, system_instruction: Optional[str], max_tokens: int) -> str:
        kwargs = {
            "model": self.model_name,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_instruction:
            kwargs["system"] = system_instruction
        message = self._client.messages.create(**kwargs)
        return message.content[0].text if message.content else ""

    def _generate_openai(self, prompt: str, system_instruction: Optional[str], max_tokens: int) -> str:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return response.choices[0].message.content or ""


# Backwards-compatible alias used by llm_labeler.py
class GeminiWrapper(LLMWrapper):
    """Deprecated alias kept for backwards compatibility."""
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        super().__init__(api_key=api_key, model_name=model_name)
