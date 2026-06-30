"""Real backends: local sentence-transformer embeddings (dense retrieval) + Claude on AWS Bedrock
(query transforms, generation, grading, graph extraction, RAPTOR summaries, the agent loop).

Credentials load from the project `.env` and the global `~/.env`. This project is real-only by
design — there is no deterministic mock — so every run makes real embedding + Bedrock calls.
"""
from __future__ import annotations

import os

os.environ.setdefault("USE_TF", "0")  # keep transformers off the Keras-3 TF path

try:
    from dotenv import load_dotenv
    load_dotenv()
    load_dotenv(os.path.expanduser("~/.env"))
except Exception:  # pragma: no cover
    pass

import numpy as np

_EMBED_MODEL = "all-MiniLM-L6-v2"
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL = os.getenv("RAGARCH_MODEL", "global.anthropic.claude-haiku-4-5-20251001-v1:0")

_embedder = None
_llm_client = None


def embed(texts: list[str]) -> np.ndarray:
    """L2-normalized sentence embeddings so cosine == dot product."""
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(_EMBED_MODEL)
    v = np.asarray(_embedder.encode(texts, normalize_embeddings=True), dtype=float)
    return v


def _client():
    global _llm_client
    if _llm_client is None:
        from anthropic import AnthropicBedrock
        _llm_client = AnthropicBedrock(aws_region=AWS_REGION)
    return _llm_client


def complete(prompt: str, max_tokens: int = 512) -> str:
    msg = _client().messages.create(
        model=BEDROCK_MODEL, max_tokens=max_tokens, temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


def judge_correct(question: str, answer: str, reference: str) -> bool:
    """LLM-as-judge: is `answer` correct given the reference? Lenient on phrasing, strict on facts."""
    out = complete(
        f"Question: {question}\nReference answer: {reference}\nCandidate answer: {answer}\n\n"
        "Is the candidate answer factually correct and complete relative to the reference? "
        "Ignore wording differences; judge only the facts. Reply with exactly YES or NO.",
        max_tokens=5,
    )
    return out.strip().upper().startswith("YES")
