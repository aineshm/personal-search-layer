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
    assert "- " in draft.answer_text
