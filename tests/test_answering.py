from personal_search_layer.answering import synthesize_extractive
from personal_search_layer.models import ScoredChunk
from personal_search_layer.router import PrimaryIntent


def test_synthesize_extractive_creates_claims_with_citations() -> None:
    chunks = [
        ScoredChunk(
            chunk_id="c1",
            doc_id="d1",
            score=1.0,
            chunk_text="Hybrid retrieval combines lexical and vector signals. Evidence must be traceable.",
            source_path="reference_docs/smoke_corpus/notes.md",
            page=None,
        )
    ]
    draft = synthesize_extractive(
        "how does hybrid retrieval work",
        chunks,
        PrimaryIntent.SYNTHESIS,
    )
    assert draft.claims
    assert all(claim.citations for claim in draft.claims)
    assert all(claim.citation_span_quality >= 0.0 for claim in draft.claims)
    assert all(claim.supportability_score >= 0.0 for claim in draft.claims)
    assert "- " in draft.answer_text


def test_synthesize_extractive_prefers_topical_claims() -> None:
    chunks = [
        ScoredChunk(
            chunk_id="c1",
            doc_id="d1",
            score=1.0,
            chunk_text="Hybrid retrieval combines lexical and vector signals. Reciprocal rank fusion merges candidate lists.",
            source_path="notes.md",
            page=1,
        ),
        ScoredChunk(
            chunk_id="c2",
            doc_id="d2",
            score=1.0,
            chunk_text="Bananas are yellow. Apples can be red.",
            source_path="fruit.md",
            page=1,
        ),
    ]
    draft = synthesize_extractive(
        "explain hybrid retrieval", chunks, PrimaryIntent.FACT
    )
    combined = " ".join(claim.text.lower() for claim in draft.claims)
    assert "hybrid retrieval" in combined


def test_synthesize_extractive_dedupes_semantic_duplicates() -> None:
    chunks = [
        ScoredChunk(
            chunk_id="c1",
            doc_id="d1",
            score=1.0,
            chunk_text="Hybrid retrieval combines lexical and vector signals for ranking.",
            source_path="a.md",
            page=1,
        ),
        ScoredChunk(
            chunk_id="c2",
            doc_id="d2",
            score=0.9,
            chunk_text="Hybrid retrieval combining lexical and vector signal improves ranking quality.",
            source_path="b.md",
            page=1,
        ),
    ]
    draft = synthesize_extractive(
        "explain hybrid retrieval ranking", chunks, PrimaryIntent.SYNTHESIS
    )
    matching = [
        claim for claim in draft.claims if "hybrid retrieval" in claim.text.lower()
    ]
    assert len(matching) == 1


def test_synthesize_extractive_prefers_multi_source_for_synthesis() -> None:
    chunks = [
        ScoredChunk(
            chunk_id="c1",
            doc_id="d1",
            score=1.0,
            chunk_text=(
                "Hybrid retrieval combines lexical and vector signals. "
                "Reciprocal rank fusion merges candidate lists."
            ),
            source_path="a.md",
            page=1,
        ),
        ScoredChunk(
            chunk_id="c2",
            doc_id="d2",
            score=0.95,
            chunk_text=(
                "Reciprocal rank fusion merges candidate lists from different retrievers."
            ),
            source_path="b.md",
            page=1,
        ),
    ]
    draft = synthesize_extractive(
        "how does reciprocal rank fusion work", chunks, PrimaryIntent.SYNTHESIS
    )
    rr_claims = [
        claim
        for claim in draft.claims
        if "reciprocal rank fusion merges candidate lists" in claim.text.lower()
    ]
    assert rr_claims
    assert rr_claims[0].source_count >= 2
    assert len(rr_claims[0].citations) >= 2
