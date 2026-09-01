"""The chat interface, in the shape llama-server's OpenAI-compatible API uses.

Modelled on that API rather than on something neutral, deliberately. Native tool
calling is a *server* feature -- llama-server parses the model's tool-call syntax
against the chat template and hands back structured calls -- so an abstraction
that invented its own message shape would have to translate into this one
anyway, and every translation is somewhere for a tool call to be silently
dropped.

Two implementations: a scripted one that replays fixtures, and a real one over
HTTP. The scripted one is what lets the entire agentic loop -- tool dispatch,
multi-turn, citation checking, report rendering -- be tested without a model.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Message:
    """One turn of the conversation."""

    role: str
    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        """The JSON form the server expects."""
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    # arguments is a JSON *string*, not an object. That is the
                    # OpenAI shape llama-server implements, and it is how the
                    # model emitted them; sending the parsed dict back is
                    # accepted by some servers and silently drops the call on
                    # others, which reads as a model that stopped using tools.
                    "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    },
                }
                for call in self.tool_calls
            ]
        if self.tool_call_id:
            out["tool_call_id"] = self.tool_call_id
        if self.name:
            out["name"] = self.name
        return out


@dataclass(frozen=True, slots=True)
class Completion:
    """One model response: prose, tool calls, or both."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class LlmClient(Protocol):
    """A chat model that can call tools."""

    def complete(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> Completion: ...


class LlmError(RuntimeError):
    """The model could not be reached, or answered with something unusable."""
