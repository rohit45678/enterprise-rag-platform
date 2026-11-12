"""Reranking: score fusion followed by MMR diversification.

The fusion step min-max normalizes the TF-IDF and dense component scores
across the candidate set and recombines them, which corrects for the two
signals living on different scales. The MMR (maximal marginal relevance)
step then greedily selects passages that are relevant to the query but
not redundant with passages already selected, using dense-embedding
cosine as the redundancy measure.
"""

from __future__ import annotations

from .embeddings import cosine


def _normalize(values):
    low, high = min(values), max(values)
    if high == low:
        return [1.0 for _ in values]
    return [(v - low) / (high - low) for v in values]


def fuse_scores(results, tfidf_weight=0.5):
    """Return copies of ``results`` with a fused, normalized ``score``."""
    if not results:
        return []
    tfidf_norm = _normalize([r["tfidf_score"] for r in results])
    dense_norm = _normalize([r["dense_score"] for r in results])
    fused = []
    for result, t_norm, d_norm in zip(results, tfidf_norm, dense_norm):
        entry = dict(result)
        entry["score"] = tfidf_weight * t_norm + (1 - tfidf_weight) * d_norm
        fused.append(entry)
    fused.sort(key=lambda r: (-r["score"], r["chunk_id"]))
    return fused


def mmr_rerank(results, corpus, top_k=5, lambda_=0.7):
    """Greedily select ``top_k`` results balancing relevance and diversity."""
    if top_k <= 0:
        return []
    selected = []
    remaining = list(results)
    while remaining and len(selected) < top_k:
        best, best_value = None, None
        for candidate in remaining:
            candidate_vec = corpus.dense_vectors[candidate["chunk_id"]]
            redundancy = max(
                (
                    cosine(candidate_vec, corpus.dense_vectors[s["chunk_id"]])
                    for s in selected
                ),
                default=0.0,
            )
            value = lambda_ * candidate["score"] - (1 - lambda_) * redundancy
            if best_value is None or value > best_value:
                best, best_value = candidate, value
        selected.append(best)
        remaining.remove(best)
    return selected


def rerank(results, corpus, top_k=5, tfidf_weight=0.5, lambda_=0.7):
    """Full rerank pipeline: fuse scores, then diversify with MMR."""
    return mmr_rerank(fuse_scores(results, tfidf_weight), corpus, top_k, lambda_)
