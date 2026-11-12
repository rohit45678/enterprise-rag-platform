"""Tests for the Advanced Enterprise RAG Platform.

Runnable two ways:
    python tests/run_tests.py     (plain-assert runner, no dependencies)
    python -m pytest tests/       (if pytest is installed)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rag import answer, chunking, index, rerank, retriever

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DOCS = os.path.join(ROOT, "sample_data", "docs")

_CORPUS = None


def corpus():
    global _CORPUS
    if _CORPUS is None:
        _CORPUS = index.load_corpus(DOCS)
    return _CORPUS


# ---------------------------------------------------------------- chunking

def test_chunking_short_text_single_chunk():
    assert chunking.chunk_text("Hello world. This is short.") == [
        "Hello world. This is short."
    ]


def test_chunking_respects_max_chars():
    text = "\n\n".join(
        f"Paragraph {i} " + ("lorem ipsum dolor sit amet " * 10) for i in range(12)
    )
    chunks = chunking.chunk_text(text, max_chars=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert all(c.strip() for c in chunks)


def test_chunking_long_paragraph_overlap():
    unit = " ".join(f"word{i}" for i in range(300))  # one long paragraph
    assert len(unit) > 800
    chunks = chunking.chunk_text(unit, max_chars=400, overlap=80)
    assert len(chunks) >= 3
    assert all(len(c) <= 400 for c in chunks)
    # Sliding windows overlap by exactly `overlap` characters.
    assert chunks[0][-80:] == chunks[1][:80]
    assert chunks[1][-80:] == chunks[2][:80]


def test_chunking_rejects_bad_overlap():
    try:
        chunking.chunk_text("some text", max_chars=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for overlap >= max_chars")


# ---------------------------------------------------------------- corpus

def test_corpus_loaded_with_metadata():
    c = corpus()
    assert len(c) >= 12  # 6 documents, multiple chunks each
    docs = {chunk["doc"] for chunk in c.chunks}
    assert docs == {
        "vpn-remote-access-policy.md",
        "employee-onboarding-guide.md",
        "expense-reimbursement-policy.md",
        "incident-response-runbook.md",
        "data-retention-policy.md",
        "cloud-cost-optimization-guide.md",
    }
    for chunk in c.chunks:
        assert chunk["metadata"]["type"] in {"policy", "guide", "runbook"}
        assert chunk["metadata"]["source"]
        assert chunk["metadata"]["date"]


# ---------------------------------------------------------------- retrieval

def test_retrieval_vpn_query_ranks_vpn_doc_first():
    results = retriever.hybrid_search(corpus(), "How do I connect to the company VPN?")
    assert results
    assert results[0]["doc"] == "vpn-remote-access-policy.md"


def test_retrieval_expense_query_ranks_expense_doc_first():
    results = retriever.hybrid_search(
        corpus(), "What is the per diem limit for meals while traveling?"
    )
    assert results
    assert results[0]["doc"] == "expense-reimbursement-policy.md"


def test_retrieval_incident_query_ranks_runbook_first():
    results = retriever.hybrid_search(
        corpus(), "What happens during a severity 1 incident escalation?"
    )
    assert results
    assert results[0]["doc"] == "incident-response-runbook.md"


# ---------------------------------------------------------------- filters

def test_metadata_type_filter():
    query = "data retention policy records retained years"
    results = retriever.hybrid_search(corpus(), query, filters={"type": "policy"})
    assert results
    assert all(r["metadata"]["type"] == "policy" for r in results)
    assert any(r["doc"] == "data-retention-policy.md" for r in results)
    # Restricting the same query to guides must exclude the policy document.
    guide_results = retriever.hybrid_search(corpus(), query, filters={"type": "guide"})
    assert all(r["metadata"]["type"] == "guide" for r in guide_results)
    assert all(r["doc"] != "data-retention-policy.md" for r in guide_results)


def test_metadata_source_filter():
    results = retriever.hybrid_search(
        corpus(), "incident escalation", filters={"source": "it-security"}
    )
    assert results
    assert all(r["metadata"]["source"] == "it-security" for r in results)
    assert any(r["doc"] == "incident-response-runbook.md" for r in results)


def test_metadata_date_filter():
    results = retriever.hybrid_search(
        corpus(), "incident severity", filters={"after": "2025-06-01"}
    )
    assert results
    assert all(r["metadata"]["date"] >= "2025-06-01" for r in results)
    assert all(r["doc"] != "vpn-remote-access-policy.md" for r in results)


# ---------------------------------------------------------------- rerank

def test_rerank_returns_diverse_top_k():
    c = corpus()
    candidates = retriever.hybrid_search(c, "company policies and procedures", top_k=10)
    assert len(candidates) >= 3
    selected = rerank.rerank(candidates, c, top_k=3)
    assert len(selected) == 3
    chunk_ids = [r["chunk_id"] for r in selected]
    assert len(set(chunk_ids)) == 3
    assert all(0.0 <= r["score"] <= 1.0 for r in selected)


# ---------------------------------------------------------------- answer

def test_extractive_answer_cites_source():
    c = corpus()
    candidates = retriever.hybrid_search(
        c, "How do I connect to the company VPN?", top_k=10
    )
    passages = rerank.rerank(candidates, c, top_k=5)
    result = answer.extract_answer("How do I connect to the company VPN?", passages)
    assert "Remote Access and VPN Policy" in result["sources"]
    assert "[1]" in result["answer"]
    assert "VPN" in result["answer"]


def test_extractive_answer_no_passages():
    result = answer.extract_answer("anything", [])
    assert result["sources"] == []
    assert "No relevant passages" in result["answer"]
