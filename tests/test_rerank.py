from personal_search_layer.models import ScoredChunk
from personal_search_layer.rerank import rerank_chunks


def test_rerank_chunks_prefers_overlap() -> None:
    chunks = [
        ScoredChunk(
            chunk_id="1",
            doc_id="d1",
            score=0.5,
            chunk_text="alpha beta gamma",
            source_path="/tmp/a",
            page=None,
        ),
        ScoredChunk(
            chunk_id="2",
            doc_id="d2",
            score=0.6,
            chunk_text="delta epsilon",
            source_path="/tmp/b",
            page=None,
        ),
    ]
    reranked = rerank_chunks("alpha", chunks)
    assert reranked[0].chunk_id == "1"
