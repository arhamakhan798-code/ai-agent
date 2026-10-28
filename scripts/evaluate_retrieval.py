"""
scripts/evaluate_retrieval.py
─────────────────────────────────────────────────────────────────────────────
Evaluates retrieval quality by running a set of realistic policy questions
against ChromaDB and reporting precision, recall (approximate), and average
retrieval score.

The evaluation dataset is a list of dicts:
    {
      "query": "What is the return window for electronics?",
      "expected_section": "RETURN POLICY",   # substring match in section_title
      "expected_keywords": ["15 days", "electronics"]  # all must appear in results
    }

Usage
──────
    python -m scripts.evaluate_retrieval
    python -m scripts.evaluate_retrieval --n_results 5 --threshold 0.3
    python -m scripts.evaluate_retrieval --dataset data/eval_queries.json

Output
───────
Prints a table with per-query results and aggregate precision/recall metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.WARNING)

# ── Built-in evaluation dataset (TechMart policy questions) ──────────────────

DEFAULT_EVAL_DATASET = [
    {
        "query": "What is the return window for electronics?",
        "expected_section": "RETURN POLICY",
        "expected_keywords": ["15 days", "electronics"],
    },
    {
        "query": "How long does a refund take to process?",
        "expected_section": "REFUND POLICY",
        "expected_keywords": ["5 business days"],
    },
    {
        "query": "Is free shipping available and what are the conditions?",
        "expected_section": "SHIPPING POLICY",
        "expected_keywords": ["$50", "free"],
    },
    {
        "query": "What does the manufacturer warranty cover?",
        "expected_section": "WARRANTY POLICY",
        "expected_keywords": ["1-year", "hardware defects"],
    },
    {
        "query": "Can I exchange a product after 30 days?",
        "expected_section": "EXCHANGE POLICY",
        "expected_keywords": ["30 days"],
    },
    {
        "query": "When is customer support available?",
        "expected_section": "CUSTOMER SUPPORT",
        "expected_keywords": ["Monday", "Friday", "9:00 AM"],
    },
    {
        "query": "What items cannot be returned?",
        "expected_section": "RETURN POLICY",
        "expected_keywords": ["Final Sale"],
    },
    {
        "query": "How much does express shipping cost?",
        "expected_section": "SHIPPING POLICY",
        "expected_keywords": ["$12.99", "1-2 business days"],
    },
]


def _hits(result_list: list[dict], expected_section: str, expected_keywords: list[str]) -> dict:
    """
    Evaluate a single query's result list.
    Returns:
        section_hit  — True if any result's section_title contains expected_section
        keyword_hits — {kw: True/False} for each expected keyword
        all_keywords — True if every keyword found in any result text
    """
    combined_text = " ".join(r["text"] for r in result_list).lower()
    combined_sections = " ".join(r.get("section_title", "") for r in result_list).upper()

    section_hit = expected_section.upper() in combined_sections

    kw_hits = {}
    for kw in expected_keywords:
        kw_hits[kw] = kw.lower() in combined_text

    return {
        "section_hit": section_hit,
        "keyword_hits": kw_hits,
        "all_keywords": all(kw_hits.values()),
        "avg_score": (
            sum(r["score"] for r in result_list) / len(result_list)
            if result_list else 0.0
        ),
    }


def run_evaluation(
    dataset: list[dict],
    n_results: int = 5,
    threshold: float = 0.35,
) -> None:
    from app.services.chroma_service import init_chroma, query_policy

    init_chroma()

    print(f"\n{'═' * 90}")
    print(f"  Retrieval Evaluation   n_results={n_results}  threshold={threshold}")
    print(f"{'═' * 90}")
    header = f"{'#':>3}  {'Section':>8}  {'Keys':>5}  {'Score':>6}  Query"
    print(header)
    print("-" * 90)

    section_hits = 0
    keyword_hits = 0
    total = len(dataset)

    for i, item in enumerate(dataset, start=1):
        results = query_policy(
            item["query"],
            n_results=n_results,
            score_threshold=threshold,
        )
        evaluation = _hits(
            results,
            item.get("expected_section", ""),
            item.get("expected_keywords", []),
        )

        s_mark = "✓" if evaluation["section_hit"] else "✗"
        k_mark = "✓" if evaluation["all_keywords"] else "✗"
        avg = f"{evaluation['avg_score']:.3f}"

        if evaluation["section_hit"]:
            section_hits += 1
        if evaluation["all_keywords"]:
            keyword_hits += 1

        query_short = item["query"][:55]
        print(f"{i:>3}  {s_mark:>8}  {k_mark:>5}  {avg:>6}  {query_short}")

        # Show missing keywords
        missing = [
            kw for kw, hit in evaluation["keyword_hits"].items() if not hit
        ]
        if missing:
            print(f"       Missing keywords: {missing}")

    print("-" * 90)
    print(f"  Section precision:  {section_hits}/{total} = {section_hits/total:.0%}")
    print(f"  Keyword precision:  {keyword_hits}/{total} = {keyword_hits/total:.0%}")
    print(f"{'═' * 90}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ChromaDB policy retrieval")
    parser.add_argument("--dataset", metavar="JSON",
                        help="Path to JSON eval dataset (default: built-in)")
    parser.add_argument("--n_results", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.35)
    args = parser.parse_args()

    if args.dataset:
        with open(args.dataset, encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        dataset = DEFAULT_EVAL_DATASET

    run_evaluation(dataset, n_results=args.n_results, threshold=args.threshold)


if __name__ == "__main__":
    main()
