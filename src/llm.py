"""Provider-neutral LLM client interface used by the agent's tool loop.

Every memory strategy and every provider (local Ollama, frontier Anthropic,
or a scripted test double) speaks the same four types: `Message`, `ToolSpec`,
`ToolCall`, `LLMResponse`. `agent.py` only ever talks to an `LLMClient`; it
never knows which provider is behind it.

Ollama is a base dependency (the key-free default path `reproduce.sh`
promises). Anthropic is intentionally NOT a base dependency — it's imported
lazily inside `AnthropicClient` so a local-only run never needs it installed
or an API key present.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import requests


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict  # JSON schema for the tool's arguments


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None  # set on role="tool" messages
    tool_name: str | None = None  # set on role="tool" messages


@dataclass
class LLMResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse: ...


# --- Test double -------------------------------------------------------


class ScriptedLLM:
    """Deterministic test double: replays a fixed sequence of responses.

    No network, no randomness — used by the unit tests to exercise
    `agent.run_session`'s tool loop and each memory strategy's
    `build_context` without depending on a real model's behavior.
    """

    model = "scripted"

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[tuple[list[Message], list[ToolSpec]]] = []

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        self.calls.append((list(messages), list(tools)))
        if not self._responses:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self._responses.pop(0)


# --- Ollama (local, key-free) -------------------------------------------


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        timeout: int = 180,
        seed: int = 42,
        num_thread: int = 8,
        num_ctx: int = 8192,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.seed = seed
        self.num_thread = num_thread
        self.num_ctx = num_ctx

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [self._to_ollama_message(m) for m in messages],
            "tools": [self._to_ollama_tool(t) for t in tools],
            "stream": False,
            # temperature/seed pinned for reproducibility (see llm.py module docstring
            # history — a prior run without this produced wildly different accuracy on
            # identical reruns). num_thread is also pinned: BLAS/threading kernels can
            # introduce tiny floating-point nondeterminism across runs with different
            # thread counts, which occasionally flips an argmax at temperature=0.
            # num_ctx MUST be set explicitly: Ollama's server-side default runtime
            # context is much smaller than a model's max supported context (observed
            # live: n_ctx_slot=4096 for mistral:latest, which supports 32768) — the
            # system prompt (full DB schema + tool defs) plus a multi-turn tool loop
            # with up to 200-row SQL results silently exceeds 4096 within a handful of
            # turns, truncating the *front* of the conversation (system prompt, tool
            # definitions) without any error. This was mistaken for a model-capability
            # ceiling before the truncation was noticed in Ollama's own server log.
            "options": {
                "temperature": 0,
                "seed": self.seed,
                "num_thread": self.num_thread,
                "num_ctx": self.num_ctx,
            },
        }
        resp = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data["message"]
        tool_calls = [
            ToolCall(id=f"call_{i}", name=tc["function"]["name"], arguments=tc["function"].get("arguments") or {})
            for i, tc in enumerate(msg.get("tool_calls") or [])
        ]
        return LLMResponse(content=msg.get("content") or "", tool_calls=tool_calls)

    @staticmethod
    def _to_ollama_message(m: Message) -> dict:
        if m.role == "tool":
            return {"role": "tool", "content": m.content}
        return {"role": m.role, "content": m.content}

    @staticmethod
    def _to_ollama_tool(t: ToolSpec) -> dict:
        return {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }


# --- Anthropic (frontier, optional extra) --------------------------------


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None, max_tokens: int = 1024):
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(
                "AnthropicClient requires the 'anthropic' package. "
                "Install it with `uv sync --extra frontier` (it is intentionally "
                "not a base dependency, so the local-only Ollama path stays key-free)."
            ) from exc
        self.model = model
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, messages: list[Message], tools: list[ToolSpec]) -> LLMResponse:
        system_parts = [m.content for m in messages if m.role == "system"]
        anthropic_messages = []
        for m in messages:
            if m.role == "system":
                continue
            if m.role == "tool":
                anthropic_messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "tool_result", "tool_use_id": m.tool_call_id, "content": m.content}
                        ],
                    }
                )
            elif m.role == "assistant" and m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks.extend(
                    {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments} for tc in m.tool_calls
                )
                anthropic_messages.append({"role": "assistant", "content": blocks})
            else:
                anthropic_messages.append({"role": m.role, "content": m.content})

        anthropic_tools = [
            {"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools
        ]
        system_text = "\n\n".join(system_parts) if system_parts else None
        # cache_control on the system block: this system prompt (DB schema + doc
        # list + tool instructions) is byte-identical across every session/turn
        # in a run, so caching it cuts input cost by ~10x on every call after the
        # first (cache reads price at ~0.1x base input, per Anthropic's pricing).
        system_param = (
            [{"type": "text", "text": system_text, "cache_control": {"type": "ephemeral"}}]
            if system_text
            else None
        )
        # cache_control on the last message block too: the API is stateless, so
        # every turn re-sends the full conversation-so-far. Without this, a
        # multi-turn tool-use session pays full input price for turn 1's content
        # again on turn 2, again on turn 3, etc. Marking the latest block lets
        # each new turn resume from the previous turn's cached prefix instead.
        if anthropic_messages:
            last_content = anthropic_messages[-1]["content"]
            if isinstance(last_content, str):
                anthropic_messages[-1]["content"] = [
                    {"type": "text", "text": last_content, "cache_control": {"type": "ephemeral"}}
                ]
            elif isinstance(last_content, list) and last_content:
                last_content[-1] = {**last_content[-1], "cache_control": {"type": "ephemeral"}}
        create_kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
            "tools": anthropic_tools,
            # No temperature override: Claude Sonnet 5 (and the 4.6+/Opus 4.7+
            # family) rejects a non-default temperature/top_p/top_k with a 400 —
            # unlike Ollama, there's no equivalent "pin for determinism" knob here.
        }
        # Only include `system` when there's actual system content — passing
        # `system=None` explicitly (e.g. RollingSummaryMemory's one-off
        # summarization calls have no system message at all) returns a 400
        # ("system: Input should be a valid array"), unlike Ollama/most REST
        # APIs where an explicit None is silently treated as "omitted".
        if system_param is not None:
            create_kwargs["system"] = system_param
        resp = self._client.messages.create(**create_kwargs)
        content_text = "".join(b.text for b in resp.content if b.type == "text")
        tool_calls = [
            ToolCall(id=b.id, name=b.name, arguments=b.input) for b in resp.content if b.type == "tool_use"
        ]
        return LLMResponse(content=content_text, tool_calls=tool_calls)
