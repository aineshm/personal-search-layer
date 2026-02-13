"""Shared data models for the personal search layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextBlock:
    text: str
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class ChunkSpan:
    text: str
    start_offset: int
    end_offset: int
    page: int | None = None
    section: str | None = None


@dataclass(frozen=True)
class LoadedDocument:
    source_path: str
    source_type: str
    title: str
    blocks: list[TextBlock]
    content_hash: str


@dataclass(frozen=True)
class LoadReport:
    source_path: str
    source_type: str | None
    bytes_total: int
    pages_total: int | None
    pages_loaded: int
    pages_skipped_empty: int
    pages_skipped_limit: int
    skip_reason: str | None


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_text: str
    start_offset: int
    end_offset: int
    section: str | None
    page: int | None


@dataclass
class IngestSummary:
    files_seen: int
    documents_added: int
    chunks_added: int
    duplicates_skipped: int
    files_skipped: int
    skip_reasons: dict[str, int]
    pages_skipped_empty: int
    pages_skipped_limit: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_seen": self.files_seen,
            "documents_added": self.documents_added,
            "chunks_added": self.chunks_added,
            "duplicates_skipped": self.duplicates_skipped,
            "files_skipped": self.files_skipped,
            "skip_reasons": self.skip_reasons,
            "pages_skipped_empty": self.pages_skipped_empty,
            "pages_skipped_limit": self.pages_skipped_limit,
        }


@dataclass(frozen=True)
class ScoredChunk:
    chunk_id: str
    doc_id: str
    score: float
    chunk_text: str
    source_path: str
    page: int | None


@dataclass
class SearchResult:
    query: str
    mode: str
    chunks: list[ScoredChunk]
    latency_ms: float


@dataclass
class IndexSummary:
    chunks_indexed: int
    model_name: str
    dim: int
    vectors_written: int
    elapsed_ms: float


@dataclass(frozen=True)
class Citation:
    claim_id: str
    chunk_id: str
    source_path: str
    page: int | None
    quote_span_start: int
    quote_span_end: int


@dataclass(frozen=True)
class Claim:
    claim_id: str
    text: str
    citations: list[Citation]
    overlap_score: float = 0.0
    citation_span_quality: float = 0.0
    source_count: int = 1
    supportability_score: float = 0.0


@dataclass
class DraftAnswer:
    answer_text: str
    claims: list[Claim]
    searched_queries: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class VerificationIssue:
    type: str
    claim_id: str | None
    detail: str


@dataclass
class VerificationResult:
    passed: bool
    issues: list[VerificationIssue]
    conflicts: list[str]
    abstain: bool
    abstain_reason: str | None
    verdict_code: str = "insufficient_evidence"
    confidence: float = 0.0
    decision_path: list[str] = field(default_factory=list)
    searched_queries: list[str] = field(default_factory=list)


@dataclass
class OrchestrationResult:
    mode: str
    intent: str
    chunks: list[ScoredChunk]
    draft_answer: DraftAnswer | None
    verification: VerificationResult | None
    tool_trace: dict[str, Any]
    latency_ms: float
