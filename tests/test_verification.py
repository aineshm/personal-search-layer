from personal_search_layer.models import Citation, Claim, DraftAnswer, ScoredChunk
from personal_search_layer.router import VerifierMode
from personal_search_layer.verification import verify_answer


def test_verify_answer_flags_unsupported_claim() -> None:
    chunk = ScoredChunk(
        chunk_id="chunk-1",
        doc_id="doc-1",
        score=1.0,
        chunk_text="The project starts in April.",
        source_path="doc-a.txt",
        page=1,
    )
    claim = Claim(
        claim_id="c1",
        text="The project starts in December.",
        citations=[
            Citation(
                claim_id="c1",
                chunk_id="chunk-1",
                source_path="doc-a.txt",
                page=1,
                quote_span_start=0,
                quote_span_end=30,
            )
        ],
    )
    draft = DraftAnswer(answer_text="- claim", claims=[claim], searched_queries=["q"])
    result = verify_answer("q", draft, [chunk], VerifierMode.STRICT)

    assert result.abstain is True
    assert any(issue.type == "unsupported_claim" for issue in result.issues)


def test_verify_answer_detects_conflict() -> None:
    chunks = [
        ScoredChunk(
            chunk_id="chunk-a",
            doc_id="doc-a",
            score=1.0,
            chunk_text="Project alpha is 2024 according to source A.",
            source_path="source_a.txt",
            page=1,
        ),
        ScoredChunk(
            chunk_id="chunk-b",
            doc_id="doc-b",
            score=0.9,
            chunk_text="Project alpha is 2025 according to source B.",
            source_path="source_b.txt",
            page=1,
        ),
    ]
    draft = DraftAnswer(answer_text="", claims=[], searched_queries=["q"])
    result = verify_answer("q", draft, chunks, VerifierMode.STRICT_CONFLICT)

    assert result.abstain is True
    assert result.conflicts
