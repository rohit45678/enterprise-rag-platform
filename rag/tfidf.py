"""TF-IDF keyword index (sparse, dependency-free).

Uses the smoothed idf formulation ``log((1 + N) / (1 + df)) + 1`` and
cosine similarity over sparse term-weight dictionaries.
"""

from __future__ import annotations

import math
from collections import Counter

from .embeddings import tokenize


class TfidfIndex:
    def __init__(self):
        self.idf = {}
        self.doc_vectors = []

    def fit(self, documents):
        """Index a list of document strings; order defines doc ids."""
        tokenized = [tokenize(doc) for doc in documents]
        n_docs = len(tokenized)
        doc_freq = Counter()
        for tokens in tokenized:
            doc_freq.update(set(tokens))
        self.idf = {
            term: math.log((1 + n_docs) / (1 + df)) + 1.0
            for term, df in doc_freq.items()
        }
        self.doc_vectors = [self._vectorize(tokens) for tokens in tokenized]
        return self

    def _vectorize(self, tokens):
        counts = Counter(tokens)
        if not counts:
            return {}
        max_count = max(counts.values())
        vector = {}
        for term, count in counts.items():
            tf = 0.5 + 0.5 * (count / max_count)  # augmented term frequency
            vector[term] = tf * self.idf.get(term, 0.0)
        norm = math.sqrt(sum(v * v for v in vector.values()))
        if norm:
            vector = {term: v / norm for term, v in vector.items()}
        return vector

    def transform(self, text):
        """Vectorize a query string into the fitted TF-IDF space."""
        return self._vectorize(tokenize(text))

    def score(self, query_vector, doc_id):
        """Cosine similarity between a query vector and an indexed doc."""
        doc_vector = self.doc_vectors[doc_id]
        if not query_vector or not doc_vector:
            return 0.0
        if len(query_vector) > len(doc_vector):
            query_vector, doc_vector = doc_vector, query_vector
        return sum(w * doc_vector.get(term, 0.0) for term, w in query_vector.items())
