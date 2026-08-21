import pytest
from unittest.mock import MagicMock, patch
from reviewer.llm import LLMClient, MockLLMClient, get_llm_client


# --- MockLLMClient ---

def test_mock_returns_response():
    c = MockLLMClient("hello")
    assert c.chat([{"role": "user", "content": "hi"}]) == "hello"

def test_mock_ignores_schema():
    c = MockLLMClient('{"k":"v"}')
    assert c.chat([{"role": "user", "content": "hi"}], schema={"type": "object"}) == '{"k":"v"}'

def test_mock_tracks_call_count():
    c = MockLLMClient("x")
    c.chat([{"role": "user", "content": "a"}])
    c.chat([{"role": "user", "content": "b"}])
    assert c.call_count == 2

def test_mock_records_calls():
    c = MockLLMClient("x")
    msgs = [{"role": "user", "content": "hello"}]
    c.chat(msgs)
    assert c.calls[0] == msgs


# --- get_llm_client ---

def test_get_llm_client_anthropic(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with patch("reviewer.llm._anthropic_sdk.Anthropic"):
        c = get_llm_client()
    assert c.backend == "anthropic"

def test_get_llm_client_uses_lm_model_override(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "claude-sonnet-4-6")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with patch("reviewer.llm._anthropic_sdk.Anthropic"):
        c = get_llm_client()
    assert c.model == "claude-sonnet-4-6"

def test_get_llm_client_ollama(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
    with patch("reviewer.llm._ollama", MagicMock()):
        c = get_llm_client()
    assert c.backend == "ollama"
    assert c.model == "gemma4:e4b"

def test_get_llm_client_prefers_anthropic_over_ollama(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
    with patch("reviewer.llm._anthropic_sdk.Anthropic"):
        c = get_llm_client()
    assert c.backend == "anthropic"

def test_get_llm_client_google(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "api-test")
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with patch("reviewer.llm._genai.Client"):
        c = get_llm_client()
    assert c.backend == "google"
    assert c.model == "gemini-2.5-flash"

def test_get_llm_client_prefers_google_over_ollama(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "api-test")
    monkeypatch.setenv("OLLAMA_MODEL", "gemma4:e4b")
    with patch("reviewer.llm._genai.Client"):
        c = get_llm_client()
    assert c.backend == "google"

def test_get_llm_client_raises_without_config(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY, GOOGLE_API_KEY, or OLLAMA_MODEL"):
        get_llm_client()


# --- LLMClient invalid backend ---

def test_invalid_backend_raises():
    with pytest.raises(ValueError, match="backend must be"):
        LLMClient(backend="openai", model="gpt-4")


# --- LLMClient google backend ---

def test_google_chat_returns_text(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "api-test")
    mock_client = MagicMock()
    mock_client.models.generate_content.return_value.text = "google says hi"
    with patch("reviewer.llm._genai.Client", return_value=mock_client):
        c = LLMClient(backend="google", model="gemini-2.0-flash")
        assert c.chat([{"role": "user", "content": "hello"}]) == "google says hi"

def test_google_chat_extracts_system_message(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "api-test")
    mock_client = MagicMock()
    with patch("reviewer.llm._genai.Client", return_value=mock_client):
        c = LLMClient(backend="google", model="gemini-2.0-flash")
        c.chat([{"role": "system", "content": "be concise"}, {"role": "user", "content": "hi"}])
    kw = mock_client.models.generate_content.call_args[1]
    assert kw["config"].system_instruction == "be concise"
    assert len(kw["contents"]) == 1
    assert kw["contents"][0].role == "user"

def test_google_chat_passes_schema(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "api-test")
    mock_client = MagicMock()
    with patch("reviewer.llm._genai.Client", return_value=mock_client):
        c = LLMClient(backend="google", model="gemini-2.0-flash")
        schema = {"type": "object"}
        c.chat([{"role": "user", "content": "hi"}], schema=schema)
    kw = mock_client.models.generate_content.call_args[1]
    assert kw["config"].response_mime_type == "application/json"
    assert kw["config"].response_schema == schema


# --- LLMClient anthropic backend ---

def _mock_ac(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    ac = MagicMock()
    ac.messages.create.return_value = msg
    return ac

def test_anthropic_chat_returns_text(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("reviewer.llm._anthropic_sdk.Anthropic", return_value=_mock_ac("response")):
        c = LLMClient(backend="anthropic", model="claude-haiku-4-5-20251001")
        assert c.chat([{"role": "user", "content": "hello"}]) == "response"

def test_anthropic_chat_extracts_system_message(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock_ac = _mock_ac("ok")
    with patch("reviewer.llm._anthropic_sdk.Anthropic", return_value=mock_ac):
        c = LLMClient(backend="anthropic", model="claude-haiku-4-5-20251001")
        c.chat([{"role": "system", "content": "be concise"}, {"role": "user", "content": "hi"}])
    kw = mock_ac.messages.create.call_args[1]
    assert kw["system"] == "be concise"
    assert all(m["role"] != "system" for m in kw["messages"])

def test_anthropic_chat_no_system_omits_kwarg(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    mock_ac = _mock_ac("ok")
    with patch("reviewer.llm._anthropic_sdk.Anthropic", return_value=mock_ac):
        c = LLMClient(backend="anthropic", model="claude-haiku-4-5-20251001")
        c.chat([{"role": "user", "content": "hi"}])
    kw = mock_ac.messages.create.call_args[1]
    assert "system" not in kw

def test_anthropic_chat_empty_content_returns_empty_string(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    msg = MagicMock()
    msg.content = []
    mock_ac = MagicMock()
    mock_ac.messages.create.return_value = msg
    with patch("reviewer.llm._anthropic_sdk.Anthropic", return_value=mock_ac):
        c = LLMClient(backend="anthropic", model="claude-haiku-4-5-20251001")
        assert c.chat([{"role": "user", "content": "hi"}]) == ""

def test_anthropic_chat_raises_with_only_system_message(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    with patch("reviewer.llm._anthropic_sdk.Anthropic"):
        c = LLMClient(backend="anthropic", model="claude-haiku-4-5-20251001")
        with pytest.raises(ValueError, match="non-system message"):
            c.chat([{"role": "system", "content": "only system"}])


# --- LLMClient ollama backend ---

def test_ollama_chat_returns_text():
    with patch("reviewer.llm._ollama") as mock_ollama:
        mock_ollama.chat.return_value = {"message": {"content": "ollama says hi"}}
        c = LLMClient(backend="ollama", model="gemma4:e4b")
        assert c.chat([{"role": "user", "content": "hello"}]) == "ollama says hi"

def test_ollama_chat_passes_schema():
    with patch("reviewer.llm._ollama") as mock_ollama:
        mock_ollama.chat.return_value = {"message": {"content": "{}"}}
        c = LLMClient(backend="ollama", model="gemma4:e4b")
        schema = {"type": "object"}
        c.chat([{"role": "user", "content": "hi"}], schema=schema)
    assert mock_ollama.chat.call_args[1]["format"] == schema

def test_ollama_chat_omits_format_when_no_schema():
    with patch("reviewer.llm._ollama") as mock_ollama:
        mock_ollama.chat.return_value = {"message": {"content": "hi"}}
        c = LLMClient(backend="ollama", model="gemma4:e4b")
        c.chat([{"role": "user", "content": "hi"}])
    assert "format" not in mock_ollama.chat.call_args[1]
