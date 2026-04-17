#!/usr/bin/env python3
"""Seed the database with a complete demo run for portfolio/testing purposes.

Usage:
    PYTHONPATH=src python scripts/seed_demo.py

Inserts a DocumentPacket with 5 claims, evidence spans, and verdicts
(mix of supported/contradicted/insufficient_support/needs_review) so the
reviewer dashboard has working content on a fresh database.
"""

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
import uuid

# Allow running from project root without install
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evidenceengine.core.database import async_session_factory
from evidenceengine.models.claim import Claim
from evidenceengine.models.document import DocumentPacket, SourceDocument
from evidenceengine.models.evidence import EvidenceSpan
from evidenceengine.models.run import RunVersion
from evidenceengine.models.verdict import Verdict

_DEMO_RAW_TEXT = """
Global average temperatures have risen by approximately 1.1°C since the pre-industrial period.
The Arctic is warming at more than twice the global average rate.
Sea levels have risen by about 20 centimeters over the past century due to thermal expansion and melting ice.
Renewable energy sources now account for over 30% of global electricity generation.
Carbon dioxide concentrations in the atmosphere have exceeded 420 parts per million, the highest in over 800,000 years.
""".strip()

_DEMO_CLAIMS = [
    (
        "Global average temperatures have risen by approximately 1.1°C since the pre-industrial period.",
        "supported",
        0.94,
        "Multiple peer-reviewed studies and IPCC reports confirm a warming of ~1.1°C above pre-industrial baseline "
        "as of 2020. The evidence is robust and consistent across independent measurement datasets.",
    ),
    (
        "The Arctic is warming at more than twice the global average rate.",
        "supported",
        0.89,
        "Arctic amplification is a well-documented phenomenon. NASA and NOAA datasets show Arctic warming at 2–4× "
        "the global mean, driven by ice-albedo feedback and atmospheric circulation changes.",
    ),
    (
        "Sea levels have risen by about 20 centimeters over the past century due to thermal expansion and melting ice.",
        "insufficient_support",
        0.61,
        "The ~20 cm figure is broadly consistent with tide gauge records, but attribution to specific causes "
        "(thermal expansion vs. ice melt) varies by study. The claim is directionally correct but overly precise "
        "without additional sourcing.",
    ),
    (
        "Renewable energy sources now account for over 30% of global electricity generation.",
        "contradicted",
        0.72,
        "IEA data for 2022–2023 shows renewables at approximately 28–30% of global electricity generation, "
        "slightly below the claimed threshold. The claim may be aspirational or based on a specific region "
        "rather than global figures.",
    ),
    (
        "Carbon dioxide concentrations in the atmosphere have exceeded 420 parts per million, "
        "the highest in over 800,000 years.",
        "needs_review",
        0.55,
        "Mauna Loa Observatory confirmed CO₂ crossing 420 ppm in 2023. However, the claim that this is "
        "'the highest in over 800,000 years' requires ice core evidence verification which is not present "
        "in the provided source documents.",
    ),
]

_DEMO_EVIDENCE = [
    [
        ("The IPCC Sixth Assessment Report documents a human-caused warming of 1.1°C above pre-industrial levels.", 0.95),
        ("NASA GISS Surface Temperature Analysis confirms global mean temperature anomaly trends since 1880.", 0.88),
    ],
    [
        ("Arctic surface air temperature has increased at roughly twice the global rate over recent decades (IPCC AR6).", 0.91),
        ("Sea ice extent decline and permafrost thaw are consistent with accelerated Arctic warming signals.", 0.79),
    ],
    [
        ("Tide gauge records show global mean sea level rise of approximately 15–20 cm over the 20th century.", 0.82),
        ("Satellite altimetry since 1993 shows an accelerating rate of sea level rise of ~3.7 mm/year.", 0.74),
    ],
    [
        ("IEA World Energy Outlook 2023 reports renewables supplied 29% of global electricity in 2022.", 0.87),
        ("IRENA data shows variable renewable share differs significantly by region, with some countries exceeding 50%.", 0.68),
    ],
    [
        ("NOAA's Mauna Loa Observatory recorded monthly average CO₂ of 421.08 ppm in April 2023.", 0.93),
        ("Ice core records from Antarctica extend the CO₂ record back 800,000 years (Lüthi et al., 2008).", 0.76),
    ],
]


async def seed() -> None:
    now = datetime.now(timezone.utc)
    packet_id = uuid.uuid4()
    run_id = uuid.uuid4()

    async with async_session_factory() as session:
        packet = DocumentPacket(
            id=packet_id,
            status="completed",
            report_filename="climate_claims_sample.pdf",
            report_file_path="",
        )
        session.add(packet)
        await session.flush()

        source_doc = SourceDocument(
            packet_id=packet_id,
            is_report=True,
            filename="climate_claims_sample.pdf",
            file_path="",
            file_type="pdf",
            file_size_bytes=0,
            raw_text=_DEMO_RAW_TEXT,
            markdown_text=_DEMO_RAW_TEXT,
            total_pages=1,
            parse_status="completed",
        )
        session.add(source_doc)
        await session.flush()

        run = RunVersion(
            id=run_id,
            packet_id=packet_id,
            status="completed",
            started_at=now,
            completed_at=now,
            model_versions={
                "classification_model": "gpt-4o-mini",
                "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2",
            },
        )
        session.add(run)
        await session.flush()

        for i, (claim_text, verdict_type, confidence, reasoning) in enumerate(_DEMO_CLAIMS):
            char_start = i * 200
            claim = Claim(
                packet_id=packet_id,
                source_document_id=source_doc.id,
                run_version_id=run_id,
                claim_text=claim_text,
                page_number=1,
                paragraph_index=i,
                char_start=char_start,
                char_end=char_start + len(claim_text),
                status="extracted",
            )
            session.add(claim)
            await session.flush()

            for j, (span_text, relevance) in enumerate(_DEMO_EVIDENCE[i]):
                span = EvidenceSpan(
                    claim_id=claim.id,
                    source_document_id=source_doc.id,
                    run_version_id=run_id,
                    span_text=span_text,
                    page_number=1,
                    paragraph_index=j,
                    char_start=j * 100,
                    char_end=j * 100 + len(span_text),
                    relevance_score=relevance,
                    retrieval_method="bm25",
                    retrieval_rank=j + 1,
                )
                session.add(span)

            verdict = Verdict(
                claim_id=claim.id,
                run_version_id=run_id,
                verdict_type=verdict_type,
                confidence_score=confidence,
                reasoning=reasoning,
                model_name="gpt-4o-mini",
                prompt_version="v1",
            )
            session.add(verdict)

        await session.commit()
        print(f"✓ Seeded demo run")
        print(f"  Packet: {packet_id}")
        print(f"  Run:    {run_id}")
        print(f"  URL:    http://localhost:8000/dashboard/{packet_id}/{run_id}")


if __name__ == "__main__":
    asyncio.run(seed())
