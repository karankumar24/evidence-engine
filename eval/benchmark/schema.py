"""Pydantic models for the EvidenceEngine evaluation benchmark."""

from typing import Literal

from pydantic import BaseModel, Field


class EvidenceSpanFixture(BaseModel):
    span_text: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    rank: int


class BenchmarkCase(BaseModel):
    id: str  # e.g. "gold-sup-001"
    verdict_class: Literal["supported", "contradicted", "insufficient_support", "needs_review"]
    claim_text: str
    evidence_spans: list[EvidenceSpanFixture]
    gold_verdict: Literal["supported", "contradicted", "insufficient_support", "needs_review"]
    source: Literal["gold", "synthetic"]  # EVAL-07: gold/synthetic tracked separately
    notes: str | None = None


class BenchmarkSuite(BaseModel):
    gold: list[BenchmarkCase]        # Only source=="gold" cases — used for gold-set metrics
    synthetic: list[BenchmarkCase]   # Only source=="synthetic" cases — excluded from gold metrics
