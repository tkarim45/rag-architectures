"""Every LLM touchpoint of the agentic package, in one place.

The step prompt is the entire agent-facing API: the system prompt sets the persona and the output
contract; ``STEP_TEMPLATE`` re-presents the question, the tool catalog, and the scratchpad
(thought / action / observation history) every step. ReAct works precisely because the model
re-reads its own trajectory each turn — the scratchpad IS the agent's memory; there is no other
state on the LLM side.

The literal phrase "Decide your next action" is the contract marker for the decision prompt:
offline tests key a stateful ``FakeLLM`` responder on it, so it must survive any rewording of the
surrounding prose.
"""
from __future__ import annotations

SYSTEM_PROMPT = (
    "You are a meticulous research agent answering questions over a document corpus you can only "
    "see through tools. You work in a ReAct loop: think about what you know so far, call ONE tool "
    "to gather evidence, read the observation, and repeat until you can answer from evidence you "
    "actually retrieved — never from prior knowledge. Multi-hop questions require chaining: search "
    "for the first entity, read what you find, then search for the entity it mentions. "
    "Give the final answer as soon as the evidence supports it; every extra step costs money. "
    "You must reply with a single JSON object and nothing else."
)

STEP_TEMPLATE = """Question: {question}

Available tools:
{tools}

Scratchpad (your previous thoughts, actions and observations):
{scratchpad}

Decide your next action. Reply with ONLY one JSON object, no prose, in one of these two shapes:
- call a tool:  {{"thought": "<why this tool and these arguments>", "action": "<tool name>", "action_input": {{<arguments matching the tool's schema>}}}}
- finish:       {{"thought": "<why the evidence is sufficient>", "final_answer": "<answer grounded in your observations>"}}"""

#: Shown in place of the scratchpad on the first step.
EMPTY_SCRATCHPAD = "(empty — this is your first step)"

#: Injected as the observation when the agent repeats an identical tool call. Re-running a
#: deterministic tool with identical arguments returns identical output, so the honest observation
#: is a nudge, not a re-execution — it breaks the most common ReAct stall (the query-repeat loop).
DUPLICATE_ACTION_NUDGE = (
    "You already ran this exact tool call and saw its output above; running it again returns the "
    "same result. Try a different query, a different tool, or give the final answer if the "
    "evidence is sufficient.")


def step_prompt(question: str, tools_block: str, scratchpad: str) -> str:
    """Render the per-step decision prompt."""
    return STEP_TEMPLATE.format(question=question, tools=tools_block,
                                scratchpad=scratchpad or EMPTY_SCRATCHPAD)
