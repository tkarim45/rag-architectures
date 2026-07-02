"""The ReAct loop (Yao et al. 2023, arXiv:2210.03629): reason, act, observe, repeat.

One trajectory = up to ``max_steps`` decisions. Each decision is a structured LLM call (through
``core.StructuredCaller``, so JSON parse failures get one automatic repair round-trip) that either
names a tool + arguments or declares a final answer. Tool output comes back as an ``Observation``
appended to the scratchpad, and the *entire* scratchpad is re-presented next step — the scratchpad
is the agent's only memory.

Loop guards, in order of application per step:

1. **Structured decoding** — malformed JSON is retried by ``StructuredCaller``; if it still fails,
   the trajectory ends with ``stop_reason="malformed_action"`` rather than crashing (re-prompting a
   model that failed a repair round with an unchanged scratchpad rarely converges — cutting the
   budget loss is the better trade).
2. **Duplicate-action detection** — an identical (tool, args) pair is not re-executed; the
   observation becomes a nudge to change strategy. This breaks the most common ReAct stall: the
   model repeating a failing query verbatim.
3. **Observation truncation** — tool output is clipped to ``max_tool_output_chars`` before it
   enters the scratchpad, bounding per-step prompt growth.
4. **Step budget** — the loop hard-stops at ``max_steps`` with ``stop_reason="budget"``; the
   evidence gathered so far is still usable (the pipeline ranks it regardless of how the loop
   ended).

Every step runs in its own tracer span, so a trace of one question reads
``agentic.trajectory > agentic.step(×n)`` with the chosen tool and observation size on each span.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from core import StructuredCaller, StructuredOutputError, Tracer

from .config import Config
from .evidence import EvidenceLog
from .prompts import DUPLICATE_ACTION_NUDGE, SYSTEM_PROMPT, step_prompt
from .tools import ToolRegistry


# ------------------------------------------------------------------------------------------
# Decision (one structured LLM output)
# ------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Decision:
    """Validated shape of one agent decision: either a tool call or a final answer."""

    thought: str
    action: str | None = None
    action_input: Mapping[str, Any] | None = None
    final_answer: str | None = None


def validate_decision(value: Any) -> Decision:
    """StructuredCaller validator — raises ValueError with an actionable message on bad shape so
    the repair re-prompt tells the model exactly what to fix."""
    if not isinstance(value, dict):
        raise ValueError("decision must be a JSON object")
    thought = value.get("thought", "")
    if not isinstance(thought, str):
        raise ValueError("'thought' must be a string")
    if "final_answer" in value:
        final_answer = value["final_answer"]
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise ValueError("'final_answer' must be a non-empty string")
        return Decision(thought=thought.strip(), final_answer=final_answer.strip())
    action = value.get("action")
    if not isinstance(action, str) or not action.strip():
        raise ValueError("decision needs either 'final_answer' or a string 'action'")
    action_input = value.get("action_input", {})
    if not isinstance(action_input, dict):
        raise ValueError("'action_input' must be a JSON object of tool arguments")
    return Decision(thought=thought.strip(), action=action.strip(), action_input=action_input)


# ------------------------------------------------------------------------------------------
# Trajectory record
# ------------------------------------------------------------------------------------------

@dataclass
class AgentStep:
    """One completed loop iteration — the unit the diagnostics expose."""

    index: int
    thought: str
    action: str | None = None
    action_input: dict[str, Any] | None = None
    observation: str | None = None
    final_answer: str | None = None
    duplicate: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"step": self.index, "thought": self.thought, "action": self.action,
                "action_input": self.action_input, "observation": self.observation,
                "final_answer": self.final_answer, "duplicate": self.duplicate}


@dataclass
class Trajectory:
    """Full record of one agent run: every step plus why the loop stopped.

    ``stop_reason`` ∈ {"final_answer", "budget", "malformed_action"}.
    """

    steps: list[AgentStep] = field(default_factory=list)
    stop_reason: str = "budget"
    final_answer: str | None = None

    @property
    def n_tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.action is not None and not s.duplicate)

    def to_diagnostics(self) -> dict[str, Any]:
        return {"stop_reason": self.stop_reason, "n_steps": len(self.steps),
                "n_tool_calls": self.n_tool_calls, "final_answer": self.final_answer,
                "trajectory": [s.to_dict() for s in self.steps]}


# ------------------------------------------------------------------------------------------
# The agent
# ------------------------------------------------------------------------------------------

class ReActAgent:
    """Drives one question through the reason → act → observe loop.

    Built per-run by the pipeline (it shares the run's ``EvidenceLog`` with the tool closures);
    holds no cross-question state, so trajectories can never bleed into each other.
    """

    def __init__(self, *, caller: StructuredCaller, registry: ToolRegistry,
                 evidence: EvidenceLog, config: Config, tracer: Tracer) -> None:
        self._caller = caller
        self._registry = registry
        self._evidence = evidence
        self._config = config
        self._tracer = tracer
        self._tools_block = registry.describe_all()

    # ---- loop ------------------------------------------------------------------------------

    def run(self, question: str) -> Trajectory:
        trajectory = Trajectory()
        seen_actions: set[tuple[str, str]] = set()
        with self._tracer.span("agentic.trajectory", max_steps=self._config.max_steps) as tspan:
            for i in range(self._config.max_steps):
                with self._tracer.span("agentic.step", step=i) as span:
                    try:
                        decision = self._decide(question, trajectory.steps)
                    except StructuredOutputError as e:
                        trajectory.steps.append(AgentStep(
                            index=i, thought="",
                            observation=f"Error: unrecoverable malformed action: {e}"))
                        trajectory.stop_reason = "malformed_action"
                        span.set("outcome", "malformed_action")
                        break

                    if decision.final_answer is not None:
                        trajectory.steps.append(AgentStep(
                            index=i, thought=decision.thought,
                            final_answer=decision.final_answer))
                        trajectory.final_answer = decision.final_answer
                        trajectory.stop_reason = "final_answer"
                        span.set("outcome", "final_answer")
                        break

                    step = self._act(i, decision, seen_actions)
                    trajectory.steps.append(step)
                    span.set("outcome", "duplicate" if step.duplicate else "tool_call")
                    span.set("tool", step.action)
                    span.set("observation_chars", len(step.observation or ""))
                    self._tracer.count("agentic.tool_calls",
                                       0.0 if step.duplicate else 1.0)
            tspan.set("stop_reason", trajectory.stop_reason)
            tspan.set("steps", len(trajectory.steps))
        return trajectory

    # ---- one decision / one action -----------------------------------------------------------

    def _decide(self, question: str, steps: list[AgentStep]) -> Decision:
        prompt = step_prompt(question, self._tools_block, self._render_scratchpad(steps))
        return self._caller.call(prompt, validator=validate_decision, system=SYSTEM_PROMPT,
                                 max_tokens=self._config.max_decision_tokens)

    def _act(self, index: int, decision: Decision,
             seen_actions: set[tuple[str, str]]) -> AgentStep:
        action = decision.action or ""
        action_input = dict(decision.action_input or {})
        key = (action, json.dumps(action_input, sort_keys=True, default=str))
        if key in seen_actions:
            return AgentStep(index=index, thought=decision.thought, action=action,
                             action_input=action_input, observation=DUPLICATE_ACTION_NUDGE,
                             duplicate=True)
        seen_actions.add(key)
        self._evidence.begin_step(index)
        observation = self._truncate(self._registry.execute(action, action_input))
        return AgentStep(index=index, thought=decision.thought, action=action,
                         action_input=action_input, observation=observation)

    # ---- rendering -----------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        limit = self._config.max_tool_output_chars
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "\n… [output truncated]"

    @staticmethod
    def _render_scratchpad(steps: list[AgentStep]) -> str:
        """Serialize the trajectory back into the prompt in classic ReAct
        Thought / Action / Observation lines."""
        lines: list[str] = []
        for step in steps:
            if step.thought:
                lines.append(f"Thought: {step.thought}")
            if step.action is not None:
                lines.append(f"Action: {step.action}({json.dumps(step.action_input or {})})")
            if step.observation is not None:
                lines.append(f"Observation: {step.observation}")
        return "\n".join(lines)
