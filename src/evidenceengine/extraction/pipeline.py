"""Extraction pipeline orchestrator.

Wires all extraction components together:
  1. Load report SourceDocument from DB
  2. Pre-filter citation blocks from parsed_content
  3. Extract references section for numeric resolution
  4. LLM claim extraction (AsyncOpenAI)
  5. Position recovery (char offsets)
  6. Anchor resolution (RapidFuzz fuzzy matching)
  7. Persist Claim + CitationAnchor rows

Key invariant: CitationAnchor rows are ALWAYS written — never dropped.
Unresolvable anchors get resolution_status='unresolvable' and target_document_id=None.
When any anchor is unresolvable, Claim.status = 'unresolvable_anchor' (not 'completed').

Phase 5 can call extract_claims_for_document() directly from an async task worker.
RunVersion lifecycle is managed by the caller (API endpoint or task worker),
not by this function — run_version_id is passed as a parameter.
"""

import logging
import uuid as _uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.extraction.anchor_resolver import (
    extract_references_entries,
    resolve_to_source_document,
)
from evidenceengine.core.config import settings
from evidenceengine.extraction.claim_extractor import extract_claims_from_blocks
from evidenceengine.extraction.position_recovery import recover_position
from evidenceengine.models.claim import CitationAnchor, Claim
from evidenceengine.models.document import SourceDocument


def _stratified_sample(claims: list, n: int) -> list:
    """Sample n claims spread evenly across 5 document sections.

    Ensures results/discussion sections are represented rather than
    only introduction/background (which appear first in the document).
    """
    if not claims or n <= 0:
        return []
    section_count = min(5, len(claims))
    section_size = max(1, len(claims) // section_count)
    sections = [claims[i:i + section_size] for i in range(0, len(claims), section_size)]
    per_section = max(1, n // len(sections))
    result = []
    for section in sections:
        result.extend(section[:per_section])
    # Fill remaining slots from the start (prefer early document coverage)
    if len(result) < n:
        seen = set(id(c) for c in result)
        for claim in claims:
            if id(claim) not in seen:
                result.append(claim)
                if len(result) >= n:
                    break
    return result[:n]

logger = logging.getLogger(__name__)


async def extract_claims_for_document(
    report_document_id: str,
    run_version_id: str,
    db: AsyncSession,
) -> list[Claim]:
    """Full extraction pipeline: load report -> detect -> extract -> resolve -> persist.

    Args:
        report_document_id: UUID string of the SourceDocument with is_report=True.
        run_version_id: UUID string of the RunVersion tracking this extraction run.
        db: Active AsyncSession — caller is responsible for commit/rollback lifecycle.

    Returns:
        List of persisted Claim ORM objects (with .id populated after flush).
        Returns [] if no citation blocks found in the report.

    Raises:
        AssertionError: If the provided document_id is not a report (is_report=False).
        NoResultFound: If the report document doesn't exist in the DB.
    """
    # 1. Load report SourceDocument
    result = await db.execute(
        select(SourceDocument).where(
            SourceDocument.id == _uuid.UUID(report_document_id)
        )
    )
    report = result.scalar_one()
    assert report.is_report, (
        f"extract_claims_for_document requires is_report=True, "
        f"but document {report_document_id} has is_report=False"
    )

    # 2. Load all non-report source documents in the same packet
    result = await db.execute(
        select(SourceDocument).where(
            SourceDocument.packet_id == report.packet_id,
            SourceDocument.is_report.is_(False),
        )
    )
    sources = result.scalars().all()

    # 3. Collect paragraph blocks (open extraction — every factual claim, not just cited)
    parsed_blocks = (report.parsed_content or {}).get("blocks", [])
    content_blocks = [
        b for b in parsed_blocks
        if b.get("text", "").strip() and b.get("block_type") != "heading"
    ]

    if not content_blocks:
        logger.warning(
            "Report %s has no content paragraphs — returning empty claim list",
            report_document_id,
        )
        return []

    # 4. Extract references entries for numeric citation resolution (still useful
    #    when the LLM does find markers in cited academic text)
    references_entries = extract_references_entries(parsed_blocks)

    # 5. LLM extraction — returns ClaimExtractionResponse
    extraction_result = await extract_claims_from_blocks(content_blocks)

    # 5b. Scale-aware claim cap — over-eager extraction on dense tabular PDFs
    # can produce hundreds of low-value claims and exhaust the LLM quota
    # during classification. Cap = min(pages * per_page, absolute).
    total_pages = max(1, report.total_pages or 1)
    effective_cap = min(
        total_pages * settings.max_claims_per_page,
        settings.max_claims_absolute,
    )
    extracted_count = len(extraction_result.claims)

    # Citation-first sampling: cited claims (sentences with [N] or (Author, Year))
    # fill the cap first. These are the claims the tool exists to verify.
    # Uncited claims are added via stratified section sampling to ensure all
    # document sections (intro/methods/results/discussion) are represented.
    cited_claims = [
        c for c in extraction_result.claims
        if getattr(c, "citation_markers", [])
    ]
    uncited_claims = [
        c for c in extraction_result.claims
        if not getattr(c, "citation_markers", [])
    ]

    if len(cited_claims) >= effective_cap:
        sampled_claims = cited_claims[:effective_cap]
    else:
        remaining = effective_cap - len(cited_claims)
        sampled_claims = cited_claims + _stratified_sample(uncited_claims, remaining)

    discarded_count = max(0, extracted_count - len(sampled_claims))
    if discarded_count > 0:
        logger.warning(
            "Extraction produced %d claims for %d-page report %s — "
            "citation-first cap at %d (%d cited, %d uncited sampled, %d discarded)",
            extracted_count, total_pages, report_document_id, effective_cap,
            len(cited_claims), len(sampled_claims) - len(cited_claims), discarded_count,
        )
    else:
        logger.info(
            "Extraction: %d claims selected (%d cited + %d uncited) for report %s",
            len(sampled_claims), len(cited_claims),
            len(sampled_claims) - len(cited_claims), report_document_id,
        )
    extraction_result.claims = sampled_claims
    # 6. Persist each claim and its anchors
    claims: list[Claim] = []
    run_version_uuid = _uuid.UUID(run_version_id)

    # Persist extraction telemetry to RunVersion.pipeline_config. Visibility of
    # over-extraction is a trust-model signal: if the LLM routinely extracts
    # 3x the cap for short reports, verdicts may be skewed toward early pages.
    from evidenceengine.models.run import RunVersion  # noqa: PLC0415
    run_row = (await db.execute(
        select(RunVersion).where(RunVersion.id == run_version_uuid)
    )).scalar_one_or_none()
    if run_row is not None:
        config_patch: dict = {}
        if extracted_count > 0:
            config_patch["extraction_telemetry"] = {
                "extracted_count": extracted_count,
                "effective_cap": effective_cap,
                "cited_claim_count": len(cited_claims),
                "uncited_claim_count": len(uncited_claims),
                "sampled_count": len(sampled_claims),
                "discarded_count": discarded_count,
                "discard_ratio": round(discarded_count / extracted_count, 3),
                "total_pages": total_pages,
            }
        if extraction_result.diagnostic:
            config_patch["zero_claims_diagnostic"] = extraction_result.diagnostic
        if config_patch:
            run_row.pipeline_config = {
                **(run_row.pipeline_config or {}),
                **config_patch,
            }

    for extracted_claim in extraction_result.claims:
        # Recover position from raw_text
        position = recover_position(
            extracted_claim.claim_text,
            report.raw_text or "",
            parsed_blocks,
        )

        claim = Claim(
            packet_id=report.packet_id,
            source_document_id=report.id,
            run_version_id=run_version_uuid,
            claim_text=extracted_claim.claim_text,
            section_header=position["section_header"],
            page_number=position["page_number"],
            paragraph_index=position["paragraph_index"],
            char_start=position["char_start"],
            char_end=position["char_end"],
            status="pending",
        )
        db.add(claim)
        await db.flush()  # populate claim.id before adding anchors

        has_unresolvable = False

        for marker in extracted_claim.citation_markers:
            doc_id, resolution_status = resolve_to_source_document(
                raw_marker=marker.raw_marker,
                citation_style=marker.citation_style,
                source_documents=list(sources),
                references_entries=references_entries,
            )

            # Convert string doc_id back to UUID (or None)
            target_document_id = _uuid.UUID(doc_id) if doc_id else None

            anchor = CitationAnchor(
                claim_id=claim.id,
                raw_marker=marker.raw_marker,
                citation_style=marker.citation_style,
                target_document_id=target_document_id,
                resolution_status=resolution_status,
            )
            db.add(anchor)

            if resolution_status == "unresolvable":
                has_unresolvable = True

        # CLAIM-06: unresolvable anchors must be reflected in claim status
        claim.status = "unresolvable_anchor" if has_unresolvable else "completed"
        claims.append(claim)

    await db.commit()
    return claims
