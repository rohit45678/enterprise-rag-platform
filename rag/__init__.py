"""Advanced Enterprise RAG Platform (self-contained implementation).

Hybrid retrieval over enterprise documents: TF-IDF keyword scoring fused
with dense hash embeddings, metadata filtering, MMR-diversified reranking,
and extractive answer assembly. No external services or API keys required.
"""

__all__ = ["chunking", "embeddings", "tfidf", "index", "retriever", "rerank", "answer"]
__version__ = "1.0.0"
