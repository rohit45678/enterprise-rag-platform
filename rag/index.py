"""Corpus loading and index construction.

Documents are Markdown/text files with an optional front-matter block::

    ---
    title: Remote Access and VPN Policy
    type: policy
    source: it-security
    date: 2025-01-15
    ---

    Body text...

Each document is chunked; every chunk carries its document's metadata so
retrievers can filter on ``type``, ``source`` and ``date``.
"""

from __future__ import annotations

import os

from .chunking import chunk_text
from .embeddings import HashEmbedding
from .tfidf import TfidfIndex


def parse_front_matter(text):
    """Split a document into (metadata dict, body)."""
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            metadata = {}
            for line in text[3:end].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
            return metadata, text[end + 4:].strip()
    return {}, text


class Corpus:
    """Chunked, indexed document collection."""

    def __init__(self, chunks, tfidf, dense_vectors):
        self.chunks = chunks                # [{chunk_id, doc, text, metadata}]
        self.tfidf = tfidf                  # TfidfIndex over chunk texts
        self.dense_vectors = dense_vectors  # parallel list of dense vectors

    def __len__(self):
        return len(self.chunks)


def load_corpus(docs_dir, max_chars=400, overlap=80, embedding_dim=256):
    """Load every ``.md``/``.txt`` file in ``docs_dir`` into a Corpus."""
    embedder = HashEmbedding(dim=embedding_dim)
    chunks = []
    for filename in sorted(os.listdir(docs_dir)):
        if not filename.endswith((".md", ".txt")):
            continue
        with open(os.path.join(docs_dir, filename), encoding="utf-8") as fh:
            metadata, body = parse_front_matter(fh.read())
        metadata.setdefault("title", filename)
        for position, chunk in enumerate(chunk_text(body, max_chars, overlap)):
            chunks.append(
                {
                    "chunk_id": len(chunks),
                    "doc": filename,
                    "position": position,
                    "text": chunk,
                    "metadata": metadata,
                }
            )
    tfidf = TfidfIndex().fit([c["text"] for c in chunks])
    dense_vectors = [embedder.embed(c["text"]) for c in chunks]
    return Corpus(chunks, tfidf, dense_vectors)
