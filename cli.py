"""CLI — run the RAG-architecture benchmark (real embeddings + Claude on Bedrock) and print the
comparison table. Use --limit / --methods to bound cost while iterating."""
from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark many RAG architectures on one corpus.")
    ap.add_argument("--methods", help="comma-separated subset (default: all)")
    ap.add_argument("--limit", type=int, help="only the first N questions")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    from benchmark import run
    names = args.methods.split(",") if args.methods else None
    res = run(method_names=names, limit=args.limit, k=args.k)

    if args.json:
        print(json.dumps(res, indent=2))
        return

    m = res["_meta"]
    print("=" * 70)
    print(f"  RAG ARCHITECTURES BENCHMARK   ({m['n_questions']} questions, {m['n_docs']} docs, "
          f"recall@{m['k']})")
    print("=" * 70)
    print(f"{'method':<20}{'recall@k':>10}{'hit-rate':>10}{'answer acc':>13}")
    print("-" * 70)
    for name in res["ranked"]:
        r = res["methods"][name]
        print(f"{name:<20}{r['recall_at_k']:>10.2f}{r['hit_rate']:>10.0%}{r['answer_acc']:>13.0%}")
    print("-" * 70)
    best = res["ranked"][0]
    print(f"best answer accuracy: {best} ({res['methods'][best]['answer_acc']:.0%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
