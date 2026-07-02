"""All LLM prompt text for the multi-query architecture lives here.

Keeping prompts out of the control flow makes them diffable, reviewable, and testable: offline
tests route on the distinctive phrase "alternative phrasings" with ``FakeLLM.on(...)``, so the
wording below is part of the package's contract with its test suite.
"""
from __future__ import annotations

EXPANSION_PROMPT = """\
You expand a search query for a retrieval system. Generate {n} alternative phrasings of the \
question below. Each phrasing must ask for exactly the same information using different words — \
synonyms, restructured syntax, more or less formal register — so a lexical or embedding search \
gets more chances to match how the source documents happen to be written.

Rules:
- Preserve the original intent exactly: add no new constraints, drop none.
- Make each phrasing meaningfully different from the others, not a trivial reordering.
- Reply with ONLY a JSON array of {n} strings. No prose, no code fences.

Question: {question}
"""
