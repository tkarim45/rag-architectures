"""CLI — run the RAG-architecture benchmark (real embeddings + Claude) and print the comparison
table. Use --limit / --methods to bound cost while iterating; --json for machine consumption."""
from __future__ import annotations

import argparse
import json
import logging
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="rag-bench",
        description="Benchmark RAG architectures on one corpus with labeled retrieval + answer "
                    "grading. Requires AWS Bedrock (or ANTHROPIC_API_KEY with "
                    "RAGARCH_LLM_BACKEND=anthropic).")
    parser.add_argument("--methods", help="comma-separated subset (default: all)")
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--k", type=int, default=5, help="retrieval cutoff (default 5)")
    parser.add_argument("--judge-samples", type=int, default=1,
                        help="odd number of judge votes per answer (default 1)")
    parser.add_argument("--json", action="store_true", help="emit full results as JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="per-question progress logs")
    parser.add_argument("--list", action="store_true", help="list method names and exit")
    args = parser.parse_args()

    from core import configure_logging

    configure_logging(logging.INFO if args.verbose else logging.WARNING)

    if args.list:
        from benchmark import CHUNK_VARIANTS

        base = ["naive", "sparse", "hybrid", "rerank", "multi_query", "rag_fusion", "hyde",
                "corrective", "adaptive", "agentic", "graphrag", "raptor"]
        print("\n".join(base + [f"chunk:{v}" for v in CHUNK_VARIANTS]))
        return

    from benchmark import run

    names = [m.strip() for m in args.methods.split(",")] if args.methods else None
    try:
        results = run(method_names=names, limit=args.limit, k=args.k,
                      judge_samples=args.judge_samples)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    meta = results["_meta"]
    width = 96
    print("=" * width)
    print(f"  RAG ARCHITECTURES BENCHMARK   ({meta['n_questions']} questions, "
          f"{meta['n_docs']} docs, cutoff k={meta['k']}, judge×{meta['judge_samples']})")
    print("=" * width)
    header = (f"{'method':<22}{'recall@k':>9}{'hit':>6}{'MRR':>7}{'NDCG':>7}"
              f"{'answer':>8}{'multi-hop':>11}{'ms/q':>9}")
    print(header)
    print("-" * width)
    for name in results["ranked"]:
        row = results["methods"][name]
        multi_hop = "—" if row["multi_hop_acc"] is None else f"{row['multi_hop_acc']:.0%}"
        print(f"{name:<22}{row['recall_at_k']:>9.2f}{row['hit_rate']:>6.0%}"
              f"{row['mrr']:>7.2f}{row['ndcg_at_k']:>7.2f}{row['answer_acc']:>8.0%}"
              f"{multi_hop:>11}{row['avg_latency_ms']:>9.0f}")
    print("-" * width)
    best = results["ranked"][0]
    line = f"best answer accuracy: {best} ({results['methods'][best]['answer_acc']:.0%})"
    if meta.get("tokens"):
        line += (f"   |   LLM calls: {meta['llm_calls']}, tokens in/out: "
                 f"{meta['tokens']['input']:,}/{meta['tokens']['output']:,}")
    print(line)
    print("=" * width)


if __name__ == "__main__":
    main()
