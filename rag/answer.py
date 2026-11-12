"""Extractive answer assembly.

Without an LLM, the platform answers by selecting the sentences from the
top-ranked passages that share the most distinct query terms, preserving
their original order, and citing the source documents inline as ``[n]``.
"""

from __future__ import annotations

import re

from .embeddings import tokenize

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text):
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.replace("\n", " ")) if s.strip()]


def extract_answer(query, passages, max_sentences=3):
    """Assemble an extractive answer from reranked ``passages``.

    Returns ``{"answer": str, "sources": [title, ...]}`` where the answer
    sentences carry ``[n]`` citations indexing into ``sources``.
    """
    query_terms = set(tokenize(query))
    scored = []
    for rank, passage in enumerate(passages):
        title = passage["metadata"].get("title", passage["doc"])
        for sentence in split_sentences(passage["text"]):
            overlap = len(query_terms & set(tokenize(sentence)))
            if overlap:
                scored.append((overlap, rank, sentence, title))
    if not scored:
        return {"answer": "No relevant passages were found for this query.", "sources": []}

    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen = sorted(scored[:max_sentences], key=lambda item: item[1])

    sources = []
    answer_sentences = []
    for _, _, sentence, title in chosen:
        if title not in sources:
            sources.append(title)
        answer_sentences.append(f"{sentence} [{sources.index(title) + 1}]")
    return {"answer": " ".join(answer_sentences), "sources": sources}
