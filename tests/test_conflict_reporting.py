from personal_search_layer.answering import synthesize_extractive
from personal_search_layer.models import ScoredChunk
from personal_search_layer.router import PrimaryIntent, VerifierMode
from personal_search_layer.verification import verify_answer


def test_conflict_reporting_includes_both_sources() -> None:
    chunks = [
        ScoredChunk(
            chunk_id="a",
            doc_id="da",
            score=1.0,
            chunk_text="Project alpha is 2024 according to source A.",
            source_path="synthetic/source_a.txt",
            page=1,
        ),
        ScoredChunk(
            chunk_id="b",
            doc_id="db",
            score=0.9,
            chunk_text="Project alpha is 2025 according to source B.",
            source_path="synthetic/source_b.txt",
            page=1,
        ),
    ]
    draft = synthesize_extractive(
        "what year is project alpha", chunks, PrimaryIntent.FACT
    )
    draft.searched_queries = ["what year is project alpha"]
    result = verify_answer(
        "what year is project alpha",
        draft,
        chunks,
        VerifierMode.STRICT_CONFLICT,
    )

    assert result.conflicts
    conflict_text = " ".join(result.conflicts)
    assert "source_a.txt" in conflict_text
    assert "source_b.txt" in conflict_text
