from personal_search_layer.ingestion.chunking import chunk_text
from personal_search_layer.models import TextBlock


def test_chunk_text_basic():
    blocks = [TextBlock(text="a" * 2500)]
    chunks = chunk_text(blocks, chunk_size=1000, overlap=100)
    assert len(chunks) == 3
    assert chunks[0].start_offset == 0
    assert chunks[1].start_offset == 900


def test_chunk_text_preserves_page_and_section() -> None:
    blocks = [TextBlock(text="hello world", page=2, section="intro")]
    chunks = chunk_text(blocks, chunk_size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0].page == 2
    assert chunks[0].section == "intro"
