from src.llm import AnthropicClient, Message, OllamaClient, ToolCall, ToolSpec


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"content": "", "tool_calls": []}}


def test_complete_pins_deterministic_and_context_options(monkeypatch):
    """Regression guard for a real live bug: Ollama's server-side default
    runtime context (observed: 4096 tokens) is much smaller than what
    mistral:latest actually supports (32768), so a multi-turn tool-use
    session silently lost its system prompt/tool definitions mid-session
    once accumulated tool results pushed the conversation past 4096 tokens
    — every session came back with no answer, and this was initially
    mistaken for a model-capability ceiling rather than a truncation bug.
    num_ctx (and temperature/seed/num_thread, for determinism) must always
    be sent explicitly rather than relying on Ollama's default.
    """
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return _FakeResponse(json)

    monkeypatch.setattr("src.llm.requests.post", fake_post)

    client = OllamaClient(model="mistral:latest")
    client.complete([Message(role="user", content="hi")], [])

    options = captured["json"]["options"]
    assert options["temperature"] == 0
    assert options["seed"] == 42
    assert options["num_thread"] == 8
    assert options["num_ctx"] == 8192


def test_num_ctx_is_configurable(monkeypatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return _FakeResponse(json)

    monkeypatch.setattr("src.llm.requests.post", fake_post)

    client = OllamaClient(model="mistral:latest", num_ctx=16384)
    client.complete([Message(role="user", content="hi")], [])

    assert captured["json"]["options"]["num_ctx"] == 16384


class _FakeAnthropicResponse:
    content = []


def test_anthropic_client_sends_no_temperature_and_caches_system_and_last_message(monkeypatch):
    """Regression guard for two real issues found while wiring up the frontier
    arm: (1) Claude Sonnet 5 rejects a non-default temperature/top_p/top_k with
    a 400 -- the old code unconditionally sent temperature=0, which would have
    broken every call; (2) without cache_control, the stateless API re-bills
    the full system prompt and growing conversation history on every turn of a
    multi-turn tool-use session.
    """
    client = AnthropicClient(model="claude-sonnet-5", api_key="fake-key-for-test")

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeAnthropicResponse()

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    client.complete(
        [
            Message(role="system", content="you are a helpful assistant"),
            Message(role="user", content="hello"),
        ],
        [],
    )

    assert "temperature" not in captured
    assert "top_p" not in captured
    assert "top_k" not in captured

    assert captured["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert captured["system"][0]["text"] == "you are a helpful assistant"

    last_message = captured["messages"][-1]
    assert last_message["content"][-1]["cache_control"] == {"type": "ephemeral"}
    assert last_message["content"][-1]["text"] == "hello"


def test_anthropic_client_omits_system_kwarg_when_no_system_message(monkeypatch):
    """Regression guard for a real live bug: RollingSummaryMemory calls
    llm_client.complete() with no system-role message at all (just a plain
    user-turn summarization prompt). The old code always passed
    system=<value-or-None>, and passing system=None explicitly returns a 400
    ('system: Input should be a valid array') -- unlike most REST clients,
    the Anthropic SDK does not treat an explicit None as 'omit this field'.
    """
    client = AnthropicClient(model="claude-sonnet-5", api_key="fake-key-for-test")

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeAnthropicResponse()

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    client.complete([Message(role="user", content="Summarize this session in 1-3 sentences.")], [])

    assert "system" not in captured


def test_anthropic_client_caches_last_block_of_tool_result_message(monkeypatch):
    """The cache breakpoint must land on whatever the last message actually is
    -- a tool_result block, not just a plain text turn."""
    client = AnthropicClient(model="claude-sonnet-5", api_key="fake-key-for-test")

    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return _FakeAnthropicResponse()

    monkeypatch.setattr(client._client.messages, "create", fake_create)

    client.complete(
        [
            Message(role="user", content="how many employees?"),
            Message(role="assistant", tool_calls=[ToolCall(id="c1", name="sql_query", arguments={"query": "SELECT 1"})]),
            Message(role="tool", content="12", tool_call_id="c1", tool_name="sql_query"),
        ],
        [],
    )

    last_message = captured["messages"][-1]
    assert last_message["content"][-1]["type"] == "tool_result"
    assert last_message["content"][-1]["cache_control"] == {"type": "ephemeral"}
