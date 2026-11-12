# Advanced Enterprise RAG Platform
Enterprise knowledge assistant combining hybrid (dense + keyword) search, reranking, metadata filtering, and contextual retrieval to answer questions accurately over large document collections.
**Tech stack:** LangChain, Pinecone, BGE Embeddings, GPT-4o, Python
**Highlights:** hybrid retrieval pipeline; reranking for answer quality; metadata-filtered search; context-aware generation.

## Implementation

A self-contained Python 3 implementation of the platform's retrieval core
using **only the standard library** — no external services, vector
database, or API keys required. Managed embeddings (BGE) are replaced by a
deterministic hashing-trick dense embedding, the vector store by an
in-memory index, and managed generation by extractive answer assembly, so
the full pipeline runs offline and reproducibly. The module boundaries
mirror the production architecture, so each piece can be swapped for its
managed counterpart (Pinecone for the index, BGE for embeddings, GPT-4o
for generation) without changing callers.

### Features

- **Ingestion & chunking** — Markdown/text documents with front-matter
  metadata (`title`, `type`, `source`, `date`) are split into
  paragraph-aware chunks; oversized paragraphs are windowed with overlap.
- **Hybrid retrieval** — every chunk is scored by TF-IDF keyword cosine
  fused with dense hash-embedding cosine (50/50 by default).
- **Metadata filtering** — filter on document `type`, `source`, and date
  range, applied *before* scoring so excluded documents never leak into
  results.
- **Reranking** — min-max score fusion across the two retrieval signals,
  followed by MMR (maximal marginal relevance) diversification so the
  top passages are relevant *and* non-redundant.
- **Extractive answers** — the answer is assembled from the highest-value
  sentences of the top passages, with inline `[n]` citations to the
  source documents.

### Architecture

```
main.py                  CLI: python main.py "question" [--type T] [--source S]
                         [--after YYYY-MM-DD] [--before YYYY-MM-DD] [--top-k N]
rag/
  chunking.py            paragraph-aware chunking with sliding-window overlap
  embeddings.py          tokenizer + deterministic MD5 hashing-trick embeddings
  tfidf.py               sparse TF-IDF index and cosine scoring
  index.py               front-matter parsing + corpus construction
  retriever.py           hybrid search with pre-scoring metadata filters
  rerank.py              score fusion + MMR diversification
  answer.py              extractive answer assembly with citations
sample_data/docs/        6 sample enterprise documents (policies, guides, runbook)
tests/
  test_rag.py            test suite (pytest-compatible)
  run_tests.py           dependency-free test runner
```

### How to run

```bash
python main.py "How do I connect to the company VPN?"
python main.py "expense limits" --type policy
python main.py "incident escalation" --source it-security --after 2025-06-01

python tests/run_tests.py        # or: python -m pytest tests/
```

### Sample output

```
Query: How do I connect to the company VPN?
Corpus: 23 chunks

Top passages:
[1] Remote Access and VPN Policy (policy, it-security, 2025-01-15) score=1.000 (tfidf=0.295, dense=0.340)
    All remote access to company systems must go through the corporate VPN. Employees connect to the VPN
    using the GlobalProtect client with their network credentials, and multi-factor authentication (MFA)
    through the authenticator app is manda...

[2] Remote Access and VPN Policy (policy, it-security, 2025-01-15) score=0.789 (tfidf=0.185, dense=0.335)
    To connect to the VPN, install the GlobalProtect client from the IT software portal, enter the gateway
    address vpn.example-corp.com, sign in with your network username and password, and approve the MFA
    push notification. Split tunneling is ...

Answer:
All remote access to company systems must go through the corporate VPN. [1] Employees connect to the VPN
using the GlobalProtect client with their network credentials, and multi-factor authentication (MFA)
through the authenticator app is mandatory for every VPN session. [1] To connect to the VPN, install the
GlobalProtect client from the IT software portal, enter the gateway address vpn.example-corp.com, sign in
with your network username and password, and approve the MFA push notification. [1]
Sources:
  [1] Remote Access and VPN Policy
```
