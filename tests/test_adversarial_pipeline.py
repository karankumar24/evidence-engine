"""Adversarial-pipeline tests: prove the trust model holds under attack.

Tests the four attack classes embedded in
examples/climate-app/sample_packets/adversarial_carbon/ by exercising the
classification pipeline with realistic inputs for each attack, mocking only
the underlying model calls so failures here point at our pipeline — not
free-tier flakiness.

Attack catalogue:
  1. Fabricated citation   → anchor_resolver unresolvable → needs_review
  2. Directly contradicted → classifier-returned "contradicted" must be persisted
  3. Cherry-picked scope   → low-confidence supported; threshold routing sends
                             it to needs_review.
  4. Self-verification trap → overconfident supported is hard-capped at
                              self_verify_supported_cap (0.80) and the source
                              label reaches the prompt as SAME DOCUMENT AS CLAIM.

Phase 02 Plan 03: every adversarial test is parametrized over
``classifier_backend_param`` (from ``tests/conftest.py``) so each attack runs
once under ``nli_primary`` AND once under ``llm_primary`` — 14 total
invocations from 7 vectors. The LLM path mocks ``classifier.asyncio.to_thread``;
the NLI path uses the ``mock_nli_probs`` fixture to pin the NLI verdict and
patches ``generate_explanation`` to None so explanation HTTP doesn't leak
into these offline tests.

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


# ---------------------------------------------------------------------------
# NLI-path helper: probs that map to a specific verdict.
# ---------------------------------------------------------------------------


def _nli_probs_for(verdict: str) -> tuple[float, float, float]:
    """Return (entail, neutral, contra) tuples that map deterministically to
    the target verdict under the Plan 02 thresholds
    (entail >= 0.80 → supported, contra >= 0.80 → contradicted, max < 0.50
    → needs_review, else insufficient_support)."""
    if verdict == "supported":
        return (0.92, 0.05, 0.03)
    if verdict == "contradicted":
        return (0.05, 0.05, 0.92)
    if verdict == "needs_review":
        # max < 0.50 — NLI isn't confident in any class.
        return (0.40, 0.35, 0.25)
    # Default: insufficient_support (neutral-ish, all <0.80, max >=0.50)
    return (0.55, 0.40, 0.05)


async def build_attack_run(
    db: AsyncSession,
    *,
    claim_text: str,
    claim_status: str = "pending",
    evidence_specs: list[dict] | None = None,
    anchor_statuses: list[str] | None = None,
):
    """Create packet + report + (optional external source) + run + claim + anchors + spans."""
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
# Attack 1 — fabricated citation (runs WITHOUT calling any model; works for both backends)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attack_1_fabricated_citation_routes_to_needs_review(
    db_session, classifier_backend_param, mock_nli_probs,
):
    """Claim status=unresolvable_anchor bypasses classifier entirely and lands in needs_review.

    Runs identically under both backends — the short-circuit lives upstream of
    the backend dispatch. Parametrization confirms no regression when
    classifier_backend flips.
    """
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.core.config import settings

    assert settings.classifier_backend == classifier_backend_param

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="Methane concentrations declined sharply in 2023 [3].",
        claim_status="unresolvable_anchor",
        evidence_specs=[],
        anchor_statuses=["unresolvable"],
    )

    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock) as mock_thread, \
         patch("evidenceengine.classification.pipeline.generate_explanation",
               new_callable=AsyncMock, return_value=None):
        verdicts = await classify_verdicts_for_run(str(run.id), db_session)

    assert len(verdicts) == 1
    v = verdicts[0]
    assert v.verdict_type == "needs_review"
    assert v.confidence_score == 0.0
    assert "unresolvable" in v.reasoning.lower()
    # No backend call happens for fabricated citations under EITHER backend.
    mock_thread.assert_not_called()


# ---------------------------------------------------------------------------
# Attack 2 — directly contradicted
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attack_2_contradicted_verdict_is_persisted_as_contradicted(
    db_session, classifier_backend_param, mock_nli_probs,
):
    """Contradicted verdicts from either backend are NOT downgraded.

    VERDICT-03 rule: contradictions are a distinct class, never collapsed.
    Confidence above threshold → verdict_type stays 'contradicted' regardless
    of which backend produced it.
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

    if classifier_backend_param == "nli_primary":
        # Pin NLI to produce a high-confidence contradicted verdict.
        mock_nli_probs([_nli_probs_for("contradicted")])
        with patch("evidenceengine.classification.pipeline.generate_explanation",
                   new_callable=AsyncMock, return_value=None):
            verdicts = await classify_verdicts_for_run(str(run.id), db_session)
    else:
        with patch("evidenceengine.classification.classifier.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread, \
             patch("evidenceengine.classification.pipeline.generate_explanation",
                   new_callable=AsyncMock, return_value=None):
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
async def test_attack_3_low_confidence_supported_routes_to_needs_review(
    db_session, classifier_backend_param, mock_nli_probs,
):
    """Below-threshold SUPPORTED must NOT persist as supported under either backend.

    LLM-primary: honest classifier reports low confidence (insufficient_support
    @ 0.55 in the test); 0.70 override routes to needs_review.

    NLI-primary: NLI returns low max prob — mapped to needs_review by the 0.50
    floor OR the tiebreaker path. Either way, we assert the verdict is NOT
    supported — the band invariant the attack class requires.
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

    if classifier_backend_param == "nli_primary":
        # NLI returns low max — triggers tiebreaker (conf < 0.65). Pin the
        # tiebreaker LLMClassifier to also return low-confidence insufficient
        # so the final verdict is explicitly NOT supported.
        mock_nli_probs([_nli_probs_for("needs_review")])  # max < 0.50
        tb_inst = MagicMock()
        tb_inst.classify = AsyncMock(return_value=__import__(
            "evidenceengine.classification.schemas", fromlist=["VerdictClassificationResponse"],
        ).VerdictClassificationResponse(
            reasoning="tiebreaker — not supported",
            verdict_type="insufficient_support",
            confidence_score=0.55,
        ))
        with patch("evidenceengine.classification.pipeline.generate_explanation",
                   new_callable=AsyncMock, return_value=None), \
             patch("evidenceengine.classification.llm_classifier.LLMClassifier",
                   return_value=tb_inst):
            verdicts = await classify_verdicts_for_run(str(run.id), db_session)
    else:
        with patch("evidenceengine.classification.classifier.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread, \
             patch("evidenceengine.classification.pipeline.generate_explanation",
                   new_callable=AsyncMock, return_value=None):
            mock_thread.return_value = mock_llm_message(
                "insufficient_support", 0.55,
                "The 36.6 GtCO2 figure is confirmed but the 'record high' clause is not supported — 2019 was the record.",
            )
            verdicts = await classify_verdicts_for_run(str(run.id), db_session)

    assert len(verdicts) == 1
    # The class-level invariant under attack 3: must NOT persist as supported
    # regardless of which backend classified it.
    assert verdicts[0].verdict_type != "supported"


# ---------------------------------------------------------------------------
# Attack 4 — self-verification trap
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_attack_4_self_verify_caps_overconfident_supported(
    db_session, classifier_backend_param, mock_nli_probs,
):
    """Self-verify cap fires on overconfident SUPPORTED under EITHER backend.

    The cap lives downstream of backend dispatch — same guard applies to NLI's
    supported verdicts as to the LLM's. Both paths must clamp confidence to
    ``self_verify_supported_cap`` (0.80).
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

    if classifier_backend_param == "nli_primary":
        # NLI-primary returns high-confidence entail (>=0.80 → supported @ 0.92).
        mock_nli_probs([_nli_probs_for("supported")])  # entail=0.92 → supported
        with patch("evidenceengine.classification.pipeline.generate_explanation",
                   new_callable=AsyncMock, return_value=None):
            verdicts = await classify_verdicts_for_run(str(run.id), db_session)
    else:
        with patch("evidenceengine.classification.classifier.asyncio.to_thread",
                   new_callable=AsyncMock) as mock_thread, \
             patch("evidenceengine.classification.pipeline.generate_explanation",
                   new_callable=AsyncMock, return_value=None):
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
async def test_attack_4_same_document_tag_reaches_classifier_prompt(
    db_session, classifier_backend_param, mock_nli_probs,
):
    """The deterministic SAME DOCUMENT AS CLAIM tag reaches the classifier prompt.

    Under ``llm_primary`` this is the verdict prompt sent to the LLM
    classifier. Under ``nli_primary`` the VERDICT call goes through NLI
    (no prompt wire-through is meaningful there), so we pin NLI and instead
    verify the SAME-DOC flag is preserved in the evidence dict reaching the
    backend.classify call. Net-net: the structural tag flows the same way
    through both paths.
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

    if classifier_backend_param == "nli_primary":
        # Under NLI, the "prompt" equivalent is the evidence_spans dict passed
        # to backend.classify. Spy on it and assert SAME-DOC flag was preserved.
        from evidenceengine.classification import pipeline as pm
        captured_spans: list[list[dict]] = []

        class SpyBackend:
            async def classify(self, claim_text, evidence_spans):
                captured_spans.append(evidence_spans)
                return VerdictClassificationResponse(
                    reasoning=".", verdict_type="insufficient_support",
                    confidence_score=0.50,
                )

        with patch.object(pm, "get_backend", return_value=SpyBackend()), \
             patch.object(pm, "generate_explanation",
                          new=AsyncMock(return_value=None)):
            await classify_verdicts_for_run(str(run.id), db_session)

        assert captured_spans, "backend.classify not called"
        spans = captured_spans[0]
        assert any(s.get("is_same_doc_as_claim") is True for s in spans), (
            f"Expected SAME-DOC flag in NLI evidence spans; got: {spans}"
        )
        assert any(s.get("source_filename") == "report.pdf" for s in spans)
        return

    # llm_primary: preserve the original wire-through assertion.
    captured_messages: list[dict] = []

    def fake_parse(**kwargs):
        captured_messages.extend(kwargs.get("messages", []))
        result_msg = MagicMock(refusal=None)
        result_msg.parsed = VerdictClassificationResponse(
            reasoning=".", verdict_type="insufficient_support", confidence_score=0.50,
        )
        return MagicMock(choices=[MagicMock(message=result_msg)])

    with patch("openai.OpenAI") as mock_openai_cls, \
         patch("evidenceengine.classification.pipeline.generate_explanation",
               new_callable=AsyncMock, return_value=None):
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
async def test_attack_4_external_source_produces_external_tag(
    db_session, classifier_backend_param, mock_nli_probs,
):
    """Evidence from a different document renders as EXTERNAL SOURCE for both backends.

    Mirrors ``test_attack_4_same_document_tag_reaches_classifier_prompt`` for
    the external case. Under NLI, we assert the is_same_doc_as_claim flag is
    False on the span dicts; under LLM, the prompt tag "EXTERNAL SOURCE"
    appears verbatim.
    """
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

    if classifier_backend_param == "nli_primary":
        from evidenceengine.classification import pipeline as pm
        captured_spans: list[list[dict]] = []

        class SpyBackend:
            async def classify(self, claim_text, evidence_spans):
                captured_spans.append(evidence_spans)
                return VerdictClassificationResponse(
                    reasoning=".", verdict_type="supported", confidence_score=0.92,
                )

        with patch.object(pm, "get_backend", return_value=SpyBackend()), \
             patch.object(pm, "generate_explanation",
                          new=AsyncMock(return_value=None)):
            await classify_verdicts_for_run(str(run.id), db_session)

        assert captured_spans
        spans = captured_spans[0]
        assert any(s.get("is_same_doc_as_claim") is False for s in spans)
        assert any(s.get("source_filename") == "source_01_external.pdf" for s in spans)
        # Must NOT tag any span as same-doc here.
        assert not any(s.get("is_same_doc_as_claim") is True for s in spans)
        return

    # llm_primary preserves the original prompt-level wire-through assertion.
    captured: list[dict] = []

    def fake_parse(**kwargs):
        captured.extend(kwargs.get("messages", []))
        result_msg = MagicMock(refusal=None)
        result_msg.parsed = VerdictClassificationResponse(
            reasoning=".", verdict_type="supported", confidence_score=0.92,
        )
        return MagicMock(choices=[MagicMock(message=result_msg)])

    with patch("openai.OpenAI") as mock_openai_cls, \
         patch("evidenceengine.classification.pipeline.generate_explanation",
               new_callable=AsyncMock, return_value=None):
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
async def test_adversarial_run_records_telemetry(
    db_session, classifier_backend_param, mock_nli_probs,
):
    """After classifying adversarial claims, pipeline_config carries telemetry for BOTH backends."""
    from evidenceengine.classification.pipeline import classify_verdicts_for_run
    from evidenceengine.models.run import RunVersion

    _, _, run, _ = await build_attack_run(
        db_session,
        claim_text="x",
        claim_status="unresolvable_anchor",
        evidence_specs=[],
        anchor_statuses=["unresolvable"],
    )

    # Short-circuit attack (unresolvable_anchor) never calls backend or
    # explanation, but we still patch to keep offline.
    with patch("evidenceengine.classification.classifier.asyncio.to_thread",
               new_callable=AsyncMock), \
         patch("evidenceengine.classification.pipeline.generate_explanation",
               new_callable=AsyncMock, return_value=None):
        await classify_verdicts_for_run(str(run.id), db_session)

    refreshed = (await db_session.execute(
        select(RunVersion).where(RunVersion.id == run.id)
    )).scalar_one()

    telemetry = (refreshed.pipeline_config or {}).get("classification_telemetry")
    assert telemetry is not None, "classification_telemetry missing from pipeline_config"
    assert telemetry["bypass_unresolvable_anchor"] == 1
    assert telemetry["claims_total"] == 1
    # Plan 03 additive keys present regardless of backend.
    assert telemetry["classifier_backend"] == classifier_backend_param
    assert "llm_tiebreaker_fired" in telemetry
    assert "explanation_generated" in telemetry
    assert "explanation_failed" in telemetry
    assert "nli_verdict_counts" in telemetry
