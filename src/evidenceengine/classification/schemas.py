"""Pydantic schema for structured verdict classification output.

Field order is intentional: reasoning before verdict_type forces chain-of-thought.
The LLM must produce step-by-step reasoning before committing to a label.
"""

from typing import Literal

from pydantic import BaseModel, Field


class VerdictClassificationResponse(BaseModel):
    """Structured output schema for verdict classification.

    Field order is intentional: reasoning before verdict_type forces chain-of-thought.
    The model must complete its reasoning before assigning a label, reducing
    label-first anchoring bias.
    """

    reasoning: str = Field(
        description=(
            "Step-by-step analysis of how the evidence relates to the claim. "
            "Examine each evidence span individually before drawing a conclusion. "
            "This field MUST be completed before assigning verdict_type."
        )
    )
    verdict_type: Literal["supported", "contradicted", "insufficient_support", "needs_review"] = Field(
        description=(
            "Classification label for the claim-evidence relationship. "
            "Use exactly one of: "
            "'supported' — evidence explicitly and completely confirms ALL elements of the claim; "
            "'contradicted' — evidence actively states the opposite (e.g., claim: 10% increase, evidence: 5% decrease); "
            "'insufficient_support' — evidence is absent, partial, or does not directly address the claim (default for uncertain cases); "
            "'needs_review' — evidence internally conflicts OR claim is genuinely ambiguous."
        )
    )
    confidence_score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Calibrated confidence in the verdict assignment, from 0.0 to 1.0. "
            "Score below 0.7 for uncertain cases. "
            "Score above 0.85 only when evidence is unambiguous. "
            "A score below 0.7 will route to needs_review automatically — this is a feature, not a failure."
        ),
    )
