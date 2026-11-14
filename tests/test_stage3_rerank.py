"""Stage 3 tests: FlashRank reranking."""
from backend.pipelines import answer
from backend.rag import make_filter, rerank, search

QUESTION = "What risks does Tesla report about battery supply?"


def test_rerank_keeps_5_of_20_without_inventing_chunks():
    # No LLM here: only Qdrant (to fetch 20 chunks) and FlashRank.
    docs = search(QUESTION, k=20, flt=make_filter(["TSLA"], [2025]))
    assert len(docs) == 20
    kept = rerank(QUESTION, docs, top_n=5)
    assert len(kept) == 5
    assert all(any(k.page_content == d.page_content for d in docs) for k in kept)  # a subset of the 20
    assert len({k.page_content for k in kept}) == 5


def test_rerank_really_reorders_the_chunks():
    # The reranker gives its own scores, so the order must differ from the search order.
    # (We do not claim it is always better: the model is tiny. We compare the answers by eye.)
    docs = search(QUESTION, k=20, flt=make_filter(["TSLA"], [2025]))
    kept = rerank(QUESTION, docs, top_n=5)
    assert [d.page_content for d in kept] != [d.page_content for d in docs[:5]]


def test_rerank_pipeline_keeps_the_filters():
    result = answer("What does GM say about cybersecurity in 2024?", "rerank", [], [], "test-stage3-gm")
    assert len(result["sources"]) == 5
    assert all(s["company"] == "GM" and s["year"] == 2024 for s in result["sources"])
    assert result["answer"].strip() and result["tools_used"] == []


def test_rerank_pipeline_sidebar_conflict():
    result = answer("What was GM's R&D expense in 2025?", "rerank", ["TSLA", "F"], [], "test-stage3-sb")
    assert "not selected in the sidebar" in result["answer"] and "General Motors" in result["answer"]
    assert result["sources"] == []
