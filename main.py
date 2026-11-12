#!/usr/bin/env python3
"""Advanced Enterprise RAG Platform CLI.

Usage:
    python main.py "How do I connect to the company VPN?"
    python main.py "expense limits" --type policy
    python main.py "incident escalation" --source it-security --after 2025-06-01

Loads the corpus from ``sample_data/docs`` (override with ``--docs``),
runs hybrid retrieval (TF-IDF + dense hash embeddings), reranks with
score fusion and MMR, prints the top passages with their sources and
scores, and finishes with an extractive answer assembled from them.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag import answer as answer_module
from rag import index, rerank, retriever


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="+", help="the question to ask")
    parser.add_argument("--docs", default="sample_data/docs",
                        help="directory of .md/.txt documents")
    parser.add_argument("--type", dest="type", default=None,
                        help="metadata filter: document type (e.g. policy, guide, runbook)")
    parser.add_argument("--source", default=None,
                        help="metadata filter: document source (e.g. it-security, hr)")
    parser.add_argument("--after", default=None,
                        help="metadata filter: only documents dated on/after YYYY-MM-DD")
    parser.add_argument("--before", default=None,
                        help="metadata filter: only documents dated on/before YYYY-MM-DD")
    parser.add_argument("--top-k", type=int, default=5,
                        help="number of passages to print")
    parser.add_argument("--candidates", type=int, default=10,
                        help="candidate pool size before reranking")
    args = parser.parse_args(argv)

    query = " ".join(args.query)
    filters = {}
    if args.type:
        filters["type"] = args.type
    if args.source:
        filters["source"] = args.source
    if args.after:
        filters["after"] = args.after
    if args.before:
        filters["before"] = args.before

    corpus = index.load_corpus(args.docs)
    candidates = retriever.hybrid_search(
        corpus, query, top_k=args.candidates, filters=filters or None
    )
    passages = rerank.rerank(candidates, corpus, top_k=args.top_k)

    print(f"Query: {query}")
    if filters:
        print(f"Filters: {filters}")
    print(f"Corpus: {len(corpus)} chunks")
    print()
    if not passages:
        print("No relevant passages found.")
        return 0

    print("Top passages:")
    for rank, passage in enumerate(passages, start=1):
        meta = passage["metadata"]
        print(
            f"[{rank}] {meta.get('title', passage['doc'])} "
            f"({meta.get('type', '?')}, {meta.get('source', '?')}, {meta.get('date', '?')}) "
            f"score={passage['score']:.3f} "
            f"(tfidf={passage['tfidf_score']:.3f}, dense={passage['dense_score']:.3f})"
        )
        print(f"    {passage['text'][:240].replace(chr(10), ' ')}...")
        print()

    result = answer_module.extract_answer(query, passages)
    print("Answer:")
    print(result["answer"])
    if result["sources"]:
        print("Sources:")
        for i, title in enumerate(result["sources"], start=1):
            print(f"  [{i}] {title}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
