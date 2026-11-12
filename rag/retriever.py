"""Hybrid retrieval with metadata filtering.

Each chunk is scored by a weighted fusion of TF-IDF keyword cosine and
dense (hash-embedding) cosine. Metadata filters (``type``, ``source``,
``after``/``before`` on the document date) are applied *before* scoring,
so filtered-out documents can never leak into results.
"""

from __future__ import annotations

from .embeddings import HashEmbedding, cosine

_embedder = HashEmbedding()


def matches_filters(metadata, filters):
    """True if a chunk's metadata satisfies all given filters."""
    if not filters:
        return True
    if "type" in filters and metadata.get("type") != filters["type"]:
        return False
    if "source" in filters and metadata.get("source") != filters["source"]:
        return False
    if "after" in filters and metadata.get("date", "") < filters["after"]:
        return False
    if "before" in filters and metadata.get("date", "9999") > filters["before"]:
        return False
    return True


def hybrid_search(corpus, query, top_k=10, filters=None, tfidf_weight=0.5):
    """Return up to ``top_k`` result dicts ranked by hybrid score.

    Each result: ``{chunk_id, doc, text, metadata, score, tfidf_score,
    dense_score}``. Only chunks with a positive hybrid score are returned.
    """
    query_tfidf = corpus.tfidf.transform(query)
    query_dense = _embedder.embed(query)

    results = []
    for chunk in corpus.chunks:
        if not matches_filters(chunk["metadata"], filters):
            continue
        chunk_id = chunk["chunk_id"]
        tfidf_score = corpus.tfidf.score(query_tfidf, chunk_id)
        dense_score = cosine(query_dense, corpus.dense_vectors[chunk_id])
        score = tfidf_weight * tfidf_score + (1 - tfidf_weight) * dense_score
        if score <= 0:
            continue
        results.append(
            {
                "chunk_id": chunk_id,
                "doc": chunk["doc"],
                "text": chunk["text"],
                "metadata": chunk["metadata"],
                "score": score,
                "tfidf_score": tfidf_score,
                "dense_score": dense_score,
            }
        )
    results.sort(key=lambda r: (-r["score"], r["chunk_id"]))
    return results[:top_k]
