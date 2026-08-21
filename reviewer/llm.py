import os
import time

import anthropic as _anthropic_sdk

try:
    import ollama as _ollama
except ImportError:
    _ollama = None  # type: ignore[assignment]

try:
    from google import genai as _genai
except ImportError:
    _genai = None  # type: ignore[assignment]


class LLMClient:
    """Multi-backend LLM client. Use backend='anthropic', 'google', or 'ollama'."""

    def __init__(self, backend: str, model: str, max_tokens: int = 4096, max_retries: int = 3):
        if backend not in ("anthropic", "google", "ollama"):
            raise ValueError(f"backend must be 'anthropic', 'google', or 'ollama', got {backend!r}")
        if backend == "ollama" and _ollama is None:
            raise ImportError("ollama package not installed. Run: uv sync --extra ollama")
        if backend == "google" and _genai is None:
            raise ImportError("google-genai package not installed. Run: uv sync")

        self.backend = backend
        self.model = model
        self._max_tokens = max_tokens
        self._max_retries = max_retries

        self._anthropic_client = _anthropic_sdk.Anthropic() if backend == "anthropic" else None
        self._google_client = _genai.Client(api_key=os.environ.get("GOOGLE_API_KEY")) if backend == "google" else None

    def chat(self, messages: list[dict], schema: dict | None = None) -> str:
        """Send messages and return response text.

        A leading role='system' item is extracted and passed as system kwarg for Anthropic/Google.
        schema: optional JSON schema for constrained decoding (ollama/google).
        """
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                if self.backend == "anthropic":
                    return self._chat_anthropic(messages)
                if self.backend == "google":
                    return self._chat_google(messages, schema)
                return self._chat_ollama(messages, schema)
            except (ConnectionError, TimeoutError, OSError) as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
            except _anthropic_sdk.APIConnectionError as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
            except _anthropic_sdk.RateLimitError as exc:
                last_exc = exc
                time.sleep(2 ** attempt)
        raise RuntimeError(
            f"LLM call failed after {self._max_retries} attempts: {last_exc}"
        ) from last_exc

    def _chat_anthropic(self, messages: list[dict]) -> str:
        system: str | None = None
        user_messages: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                user_messages.append(m)
        if not user_messages:
            raise ValueError("messages must contain at least one non-system message")
        kwargs: dict = dict(model=self.model, max_tokens=self._max_tokens, messages=user_messages)
        if system is not None:
            kwargs["system"] = system
        assert self._anthropic_client is not None
        msg = self._anthropic_client.messages.create(**kwargs)
        return msg.content[0].text if msg.content else ""

    def _chat_google(self, messages: list[dict], schema: dict | None) -> str:
        assert self._google_client is not None
        system: str | None = None
        contents: list = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                # Basic conversion for genai SDK
                role = "user" if m["role"] == "user" else "model"
                contents.append(_genai.types.Content(role=role, parts=[_genai.types.Part(text=m["content"])]))

        config: dict = {"max_output_tokens": self._max_tokens}
        if system:
            config["system_instruction"] = system
        if schema:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema

        resp = self._google_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=_genai.types.GenerateContentConfig(**config)
        )
        return resp.text

    def _chat_ollama(self, messages: list[dict], schema: dict | None) -> str:
        assert _ollama is not None
        kwargs: dict = dict(model=self.model, messages=messages)
        if schema is not None:
            kwargs["format"] = schema
        resp = _ollama.chat(**kwargs)
        return resp["message"]["content"]


class MockLLMClient:
    """Deterministic stub for tests. Returns a fixed response string."""

    def __init__(self, response: str = ""):
        self._response = response
        self.call_count = 0
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], schema: dict | None = None) -> str:
        self.call_count += 1
        self.calls.append(messages)
        return self._response


def get_llm_client() -> LLMClient:
    """Build LLMClient from environment. Priority: ANTHROPIC_API_KEY > GOOGLE_API_KEY > OLLAMA_MODEL.

    Override model with LLM_MODEL env var.
    """
    model_override = os.environ.get("LLM_MODEL")

    if os.environ.get("ANTHROPIC_API_KEY"):
        model = model_override or "claude-haiku-4-5"
        return LLMClient(backend="anthropic", model=model)

    if os.environ.get("GOOGLE_API_KEY"):
        model = model_override or "gemini-2.5-flash"
        return LLMClient(backend="google", model=model)

    ollama_model = os.environ.get("OLLAMA_MODEL")
    if ollama_model:
        return LLMClient(backend="ollama", model=ollama_model)

    raise ValueError("Set ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OLLAMA_MODEL to enable LLM features")
