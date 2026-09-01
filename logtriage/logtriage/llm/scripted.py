"""A model that replays a fixture instead of thinking.

This is what makes the agentic loop testable: tool dispatch, multi-turn
conversation, citation checking and report rendering all run end to end with no
model, no GPU, and no network.

It raises when the script runs out rather than returning a default. That is the
whole design of the thing. A fake that answers "" to an unexpected request makes
a broken loop look like a working one -- the run completes, the report renders,
every test passes, and nothing was exercised. An exception naming the
conversation state is the only outcome that cannot be mistaken for success.

What it cannot tell you is whether a real model behaves this way. These fixtures
are what a correct model *would* emit; whether Gemma does is a question for a
served model, and no amount of scripting answers it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from logtriage.llm.base import Completion, LlmError, Message, ToolCall


@dataclass(frozen=True, slots=True)
class Step:
    """One scripted response, optionally guarded by what preceded it."""

    completion: Completion

    # Substring that must appear in the most recent message. A guard rather than
    # a matcher: steps replay in order, and this catches a fixture that has
    # drifted out of step with the pipeline instead of silently answering the
    # wrong question.
    expect: str | None = None

    def check(self, messages: Sequence[Message]) -> None:
        if self.expect is None:
            return
        recent = messages[-1].content if messages else ""
        if self.expect not in recent:
            raise LlmError(
                f"scripted step expected {self.expect!r} in the last message, "
                f"which was: {recent[:200]!r}"
            )


@dataclass
class ScriptedLlmClient:
    """Replays scripted completions in order."""

    steps: list[Step] = field(default_factory=list)
    calls: list[list[Message]] = field(default_factory=list)
    _position: int = 0

    @classmethod
    def of(cls, *completions: Completion) -> ScriptedLlmClient:
        return cls(steps=[Step(completion=c) for c in completions])

    def complete(
        self, messages: Sequence[Message], tools: Sequence[dict[str, Any]] | None = None
    ) -> Completion:
        self.calls.append(list(messages))
        if self._position >= len(self.steps):
            roles = " -> ".join(m.role for m in messages)
            raise LlmError(
                f"scripted client exhausted after {self._position} responses; the "
                f"pipeline asked for another. Conversation so far: {roles}"
            )
        step = self.steps[self._position]
        self._position += 1
        step.check(messages)
        return step.completion

    @property
    def exhausted(self) -> bool:
        return self._position >= len(self.steps)


def tool_call(name: str, **arguments: Any) -> ToolCall:
    """Build a tool call for a fixture, with an id derived from the name."""
    return ToolCall(id=f"call_{name}", name=name, arguments=dict(arguments))


def says(content: str) -> Completion:
    return Completion(content=content)


def calls(*tool_calls: ToolCall) -> Completion:
    return Completion(tool_calls=tuple(tool_calls))
