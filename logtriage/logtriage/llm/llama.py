"""Chat completions from llama-server, with native tool calling.

The real half of the LlmClient pair. Its counterpart replays a script, which is
what makes the whole agentic loop testable without a model; this one is the
thing the loop was written for.

Tool calling here is the server's, not ours. llama-server applies the model's
chat template, parses whatever tool-call syntax that template produces, and
returns OpenAI-shaped `tool_calls`. That is why `docker-compose.logtriage.yaml`
sets JINJA_ENABLED=true: without the template the server has nothing to parse
against, the model's tool call arrives as prose in `content`, and the loop
completes having called no tool -- a run that looks successful and consulted
nothing.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from logtriage.llm.base import Completion, LlmError, Message, ToolCall
from logtriage.transport import TransportError, post_json

# Deterministic by default. Two triage runs over one log should produce the same
# report: the detectors already are deterministic, and a narrative that changes
# between runs makes it impossible to tell a prompt change from sampling noise.
DEFAULT_TEMPERATURE = 0.0

# A cap, not a target. Reports are short; this exists so a model that starts
# repeating cannot generate until the context is full.
DEFAULT_MAX_TOKENS = 2048


class LlamaClient:
    """LlmClient backed by a llama.cpp server's OpenAI-compatible API."""

    def __init__(
        self,
        base_url: str,
        model: str = "default",
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float | None = None,
    ) -> None:
        self._url = base_url.rstrip("/") + "/v1/chat/completions"
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def complete(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [message.to_wire() for message in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            # The loop drives turns itself and needs the whole message before it
            # can dispatch a tool call, so there is nothing to stream to.
            "stream": False,
        }
        if tools:
            payload["tools"] = list(tools)
            # "auto" rather than "required": the last turn of the loop is the
            # one that writes the report, and a model forced to call a tool can
            # never take it.
            payload["tool_choice"] = "auto"

        kwargs: dict[str, Any] = {}
        if self._timeout is not None:
            kwargs["timeout"] = self._timeout

        try:
            response = post_json(self._url, payload, **kwargs)
        except TransportError as error:
            raise LlmError(str(error)) from error

        return _completion(response)


def _completion(response: dict[str, Any]) -> Completion:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        # A well-formed error body reaches here rather than the transport when
        # the server answers 200 with an error payload, which llama-server does
        # for some template failures.
        detail = response.get("error", response)
        raise LlmError(f"response carried no choices: {detail!r}")

    choice = choices[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if not isinstance(content, str):
        raise LlmError(f"content is {type(content).__name__}, expected a string")

    finish = choice.get("finish_reason")
    if finish == "length":
        # Truncation is worth naming. A report cut off mid-sentence still parses,
        # still renders, and its missing half is invisible to the citation check.
        raise LlmError(
            "the model hit the token limit mid-answer; the response is truncated"
        )

    return Completion(content=content, tool_calls=_tool_calls(message))


def _tool_calls(message: dict[str, Any]) -> tuple[ToolCall, ...]:
    raw = message.get("tool_calls")
    if not raw:
        return ()
    if not isinstance(raw, list):
        raise LlmError(f"tool_calls is {type(raw).__name__}, expected an array")

    calls: list[ToolCall] = []
    for entry in raw:
        function = entry.get("function") or {}
        name = function.get("name")
        if not name:
            raise LlmError(f"tool call carries no function name: {entry!r}")
        calls.append(
            ToolCall(
                # An id is required to pair the result back to the call. Servers
                # have been known to omit it for a single call; synthesising one
                # is better than dropping the call, and the position is unique
                # within the turn.
                id=str(entry.get("id") or f"call_{len(calls)}"),
                name=str(name),
                arguments=_arguments(function.get("arguments"), name),
            )
        )
    return tuple(calls)


def _arguments(raw: Any, name: str) -> dict[str, Any]:
    """Parse a tool call's arguments, which arrive as a JSON string.

    Both shapes are accepted because both occur: the OpenAI spec says string,
    and llama-server honours that, but a model whose template emits a bare
    object can produce one. Rejecting the object form would fail a run for a
    difference the caller cannot act on.

    A string that is not JSON is a different matter and is raised. It means the
    model produced malformed arguments, and dispatching the tool with an empty
    dict would run the wrong query and report its answer as evidence.
    """
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise LlmError(f"{name}: arguments are {type(raw).__name__}, expected a string")
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LlmError(f"{name}: arguments are not valid JSON: {raw!r}") from error
    if not isinstance(parsed, dict):
        raise LlmError(
            f"{name}: arguments parsed to {type(parsed).__name__}, expected an object"
        )
    return parsed
