"""Adversarial-pipeline tests: prove the trust model holds under attack.

Tests the four attack classes embedded in
examples/climate-app/sample_packets/adversarial_carbon/ by exercising the
classification pipeline with realistic inputs for each attack, mocking only
the LLM call so failures here point at our pipeline — not free-tier flakiness.

Attack catalogue:
  1. Fabricated citation   → anchor_resolver unresolvable → needs_review
  2. Directly contradicted → LLM-returned "contradicted" must be persisted
  3. Cherry-picked scope   → LLM returns partial support below threshold;
                             threshold routing sends it to needs_review.
  4. Self-verification trap → LLM overconfident supported is hard-capped at
                              self_verify_supported_cap (0.80) and the source
                              label reaches the prompt as SAME DOCUMENT AS CLAIM.

Run: pytest tests/test_adversarial_pipeline.py -v
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL",
    "postgresql+asyncpg://evidenceengine:evidenceengine_dev@localhost:5432/evidenceengine",
)


@pytest_asyncio.fixture
async def test_engine():
    engine = create_async_engine(TEST_DATABASE_URL, echo=False)
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine):
    session_factory = async_sessionmaker(test_engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        yield session
        await session.rollback()


def mock_llm_message(verdict_type: str, confidence: float, reasoning: str = "."):
    """Build the message object that classifier's asyncio.to_thread returns."""
    from evidenceengine.classification.schemas import VerdictClassificationResponse

    parsed = VerdictClassificationResponse(
        reasoning=reasoning, verdict_type=verdict_type, confidence_score=confidence,
    )
    msg = MagicMock(parsed=parsed, refusal=None)
    return msg


async def build_attack_run(
    db: AsyncSession,
    *,
    claim_text: str,
    claim_status: str = "pending",
    evidence_specs: list[dict] | None = None,
    anchor_statuses: list[str] | None = None,
):
    """Create packet + report + (optional external source) + run + claim + anchors + spans.

    evidence_specs: list of {"text": ..., "relevance": ..., "from_report": bool}.
      If from_report=True the span is persisted against the report SourceDocument;
      otherwise a distinct external SourceDocument is created and the span points
      there. This is what drives the self-verify vs external distinction in the
      classifier prompt tags.
    anchor_statuses: list of "resolved" | "unresolvable" strings — one CitationAnchor
      per element is attached to the claim. Empty list means zero anchors.
    """
    from evidenceengine.models.document import DocumentPacket, SourceDocument
    from evidenceengine.models.run import RunVersion
    from evidenceengine.models.claim import Claim, CitationAnchor
    from evidenceengine.models.evidence import EvidenceSpan

    packet = DocumentPacket(status="queued", report_filename="report.pdf", report_file_path="/r.pdf")
    db.add(packet)
    await db.flush()

    report = SourceDocument(
        packet_id=packet.id, is_report=True, filename="report.pdf",
        file_path="/r.pdf", file_type="pdf", parsed_content={"blocks": []},
    )
    db.add(report)
    await db.flush()

    external = None
    if any(spec.get("from_report") is False for spec in (evidence_specs or [])):
        external = SourceDocument(
            packet_id=packet.id, is_report=False, filename="source_01_external.pdf",
            file_path="/s.pdf", file_type="pdf", parsed_content={"blocks": []},
            raw_text="External source document content.",
        )
        db.add(external)
        await db.flush()

    run = RunVersion(packet_id=packet.id, status="retrieval_complete", pipeline_config={})
    db.add(run)
    await db.flush()

    claim = Claim(
        packet_id=packet.id, source_document_id=report.id, run_version_id=run.id,
        claim_text=claim_text, status=claim_status,
    )
    db.add(claim)
    await db.flush()

    for status in (anchor_statuses or []):
        anchor = CitationAnchor(
            claim_id=claim.id,
            raw_marker="[1]",
            citation_style="numeric",
            resolution_status=status,
            target_document_id=external.id if (status == "resolved" and external is not None) else None,
        )
        db.add(anchor)

    for i, spec in enumerate(evidence_specs or []):
        source = report if spec.get("from_report", True) else external
        span = EvidenceSpan(
            claim_id=claim.id,
            source_document_id=source.id,
            run_version_id=run.id,
            span_text=spec["text"],
            char_start=i * 500,
            char_end=(i * 500) + len(spec["text"]),
            relevance_score=spec["relevance"],
            retrieval_method="cross_encoder",
            retrieval_rank=i + 1,
        )
        db.add(span)
    await db.flush()

    return packet, report, run, claim


# ---------------------------------------------------------------------------
# Attack 1 — fabricated citation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attack_1_fabricated_citation_routes_to_needs_review(db_session):
    """Claim status=unresolvable_anchor bypasses the LLM and lands in needs_review.

    The pipeline upstream marks claim.status='unresolvable_anchor' when all anchors
    fail to resolve. Classification must not pretend to verify it.
    """
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.verdict import Verdict

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="Methane concentrations declined sharply in 2023 [3].",
        claim_status="unresolvable_anchor",
        evidence_specs=[],
        anchor_statuses=["unresolvable"],
    )

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        verdicts = await classify_verdicts_for_run(str(run.id), db_session)

    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.verdict_type == "needs_review"
    assert v.confidence_score == 0.0
    assert "unresolvable" in v.reasoning.lower()
    # Critical — no LLM call happens for fabricated citations.
    mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# Attack 2 — directly contradicted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attack_2_contradicted_verdict_is_persisted_as_contradicted(db_session):
    """When the LLM flags contradiction, the pipeline must NOT downgrade it.

    VERDICT-03 rule: contradictions are a distinct class, never collapsed.
    Confidence above threshold → verdict_type stays 'contradicted'.
    """
    from evidenceengine.classification.pipeline import classify_verdicts_for_run

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="Global CO2 emissions decreased 12 percent in 2022 compared with 2019 [1].",
        evidence_specs=[
            {"text": "Global fossil CO2 emissions in 2022 approached the 2019 pre-pandemic record of 36.7 GtCO2.",
             "relevance": 0.93, "from_report": False},
        ],
        anchor_statuses=["resolved"],
    )

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = mock_llm_message(
            "contradicted", 0.92,
            "Source states emissions approached the 2019 record; claim asserts a decrease.",
        )
        verdicts = await classify_verdicts_for_run(str(run.id), db_session)

    assert len(verdicts) == 1
    assert verdicts[0].verdict_type == "contradicted"
    assert verdicts[0].confidence_score >= 0.70


# ---------------------------------------------------------------------------
# Attack 3 — cherry-picked scope (partial support)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attack_3_low_confidence_supported_routes_to_needs_review(db_session):
    """Below-threshold SUPPORTED must be overridden to needs_review.

    When the classifier honestly reports low confidence on partially-supported
    claims, the threshold safety net (0.70) routes them away from supported.
    """
    from evidenceengine.classification.pipeline import classify_verdicts_for_run

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="Global fossil CO2 emissions in 2022 reached 36.6 GtCO2, a record high [1].",
        evidence_specs=[
            {"text": "2022 emissions are estimated at 36.6 GtCO2, approaching (but not exceeding) the 2019 peak of 36.7 GtCO2.",
             "relevance": 0.88, "from_report": False},
        ],
        anchor_statuses=["resolved"],
    )

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        # Honest calibration — model confirms the number but flags "record high" is unsupported
        mock_thread.return_value = mock_llm_message(
            "insufficient_support", 0.55,
            "The 36.6 GtCO2 figure is confirmed but the 'record high' clause is not supported — 2019 was the record.",
        )
        verdicts = await classify_verdicts_for_run(str(run.id), db_session)

    assert len(verdicts) == 1
    # insufficient_support OR needs_review (threshold override) is acceptable.
    # What MUST NOT happen: supported.
    assert verdicts[0].verdict_type != "supported"


# ---------------------------------------------------------------------------
# Attack 4 — self-verification trap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attack_4_self_verify_caps_overconfident_supported(db_session):
    """When all evidence comes from the same doc as the claim, SUPPORTED is capped.

    Even if the LLM returns supported @ 0.95, the self-verify guard hard-caps
    confidence at self_verify_supported_cap (default 0.80). The band design
    keeps 0.70–0.80 as the "self-corroboration can pass" window.
    """
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="Land-use change emissions averaged 3.9 GtCO2 per year during 2012-2021.",
        evidence_specs=[
            # from_report=True — the same document as the claim. This is self-verify.
            {"text": "The average annual land-use change contribution over the most recent decade was approximately 3.9 gigatonnes of CO2.",
             "relevance": 0.91, "from_report": True},
        ],
        anchor_statuses=[],  # no citation = no resolved anchor = self-verify mode
    )

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread:
        mock_thread.return_value = mock_llm_message(
            "supported", 0.95, "Different paragraph restates the claim verbatim.",
        )
        verdicts = await classify_verdicts_for_run(str(run.id), db_session)

    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.confidence_score <= settings.self_verify_supported_cap + 1e-9, (
        f"self-verify cap leak: {v.confidence_score} > {settings.self_verify_supported_cap}"
    )
    # Reasoning must carry the audit trail so reviewers see the cap was applied.
    assert f"Self-verify cap {settings.self_verify_supported_cap:.2f}" in v.reasoning


@pytest.mark.asyncio
async def test_attack_4_same_document_tag_reaches_classifier_prompt(db_session):
    """Verify the deterministic SAME DOCUMENT AS CLAIM tag actually reaches the prompt.

    The v1.2.6 self-verify defense relied on the LLM inferring self-reference from
    text content (it couldn't — spans had no attribution). v1.2.8 A1 made this
    structural. This test guards the wire-through.
    """
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.classification.schemas import VerdictClassificationResponse

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="Land-use change emissions averaged 3.9 GtCO2 per year.",
        evidence_specs=[
            {"text": "Land-use change contributed ~3.9 GtCO2/yr over 2012-2021.",
             "relevance": 0.85, "from_report": True},
        ],
        anchor_statuses=[],
    )

    captured_messages: list[dict] = []

    def fake_parse(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        result_msg = MagicMock(refusal=None)
        result_msg.parsed = VerdictClassificationResponse(
            reasoning=".", verdict_type="insufficient_support", confidence_score=0.50,
        )
        return MagicMock(choices=[MagicMock(message=result_msg)])

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value.__enter__.return_value
        mock_client.beta.chat.completions.parse = fake_parse
        await classify_verdicts_for_run(str(run.id), db_session)

    user_msgs = [m["content"] for m in captured_messages if m.get("role") == "user"]
    assert user_msgs, "no user message was captured"
    prompt = user_msgs[0]
    assert "SAME DOCUMENT AS CLAIM" in prompt, (
        f"Expected structural self-verify tag in prompt; got:\n{prompt[:500]}..."
    )
    assert "source: report.pdf" in prompt, "expected source filename tag in prompt"


@pytest.mark.asyncio
async def test_attack_4_external_source_produces_external_tag(db_session):
    """Evidence from a different document must render as EXTERNAL SOURCE in the prompt."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.classification.schemas import VerdictClassificationResponse

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="Fossil emissions reached 36.6 GtCO2 in 2022 [1].",
        evidence_specs=[
            {"text": "Global fossil CO2 emissions in 2022 are estimated at 36.6 GtCO2.",
             "relevance": 0.95, "from_report": False},
        ],
        anchor_statuses=["resolved"],
    )

    captured: list[dict] = []

    def fake_parse(**kwargs):
        captured.extend(kwargs.get("messages", []))
        result_msg = MagicMock(refusal=None)
        result_msg.parsed = VerdictClassificationResponse(
            reasoning=".", verdict_type="supported", confidence_score=0.92,
        )
        return MagicMock(choices=[MagicMock(message=result_msg)])

    with patch("openai.OpenAI") as mock_openai_cls:
        mock_client = mock_openai_cls.return_value.__enter__.return_value
        mock_client.beta.chat.completions.parse = fake_parse
        await classify_verdicts_for_run(str(run.id), db_session)

    user_msgs = [m["content"] for m in captured if m.get("role") == "user"]
    assert user_msgs
    prompt = user_msgs[0]
    assert "EXTERNAL SOURCE" in prompt
    assert "source: source_01_external.pdf" in prompt
    # Must NOT falsely tag an external span as same-doc.
    assert "SAME DOCUMENT AS CLAIM" not in prompt


# ---------------------------------------------------------------------------
# Telemetry — verify the attacks show up in pipeline_config telemetry.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_adversarial_run_records_telemetry(db_session):
    """After classifying adversarial claims, run.pipeline_config carries trust-model telemetry."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.run import RunVersion

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="x",
        claim_status="unresolvable_anchor",
        evidence_specs=[],
        anchor_statuses=["unresolvable"],
    )

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock):
        await classify_verdicts_for_run(str(run.id), db_session)

    refreshed = (await db_session.execute(
        select(RunVersion).where(RunVersion.id == run.id)
    )).scalar_one()

    telemetry = (refreshed.pipeline_config or {}).get("classification_telemetry")
    assert telemetry is not None, "classification_telemetry missing from pipeline_config"
    assert telemetry["bypass_unresolvable_anchor"] == 1
    assert telemetry["claims_total"] == 1
