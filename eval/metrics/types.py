"""Pydantic models for benchmark results and eval report."""

from typing import Literal

from pydantic import BaseModel, Field

VerdictClass = Literal["supported", "contradicted", "insufficient_support", "needs_review"]


class BenchmarkResult(BaseModel):
    case_id: str
    gold_verdict: VerdictClass
    predicted_verdict: VerdictClass
    evidence_span_count: int  # number of evidence_spans retrieved for this case
    source: Literal["gold", "synthetic"]


class EvalReport(BaseModel):
    false_support_rate: float = Field(ge=0.0, le=1.0)
    contradiction_recall: float = Field(ge=0.0, le=1.0)
    insufficient_support_precision: float = Field(ge=0.0, le=1.0)
    evidence_retrieval_recall: float = Field(ge=0.0, le=1.0)
    # Denominator counts (for interpretability)
    n_gold: int
    n_gold_contradicted: int
    n_gold_insufficient: int
    n_predicted_insufficient: int
    n_gold_with_evidence: int
