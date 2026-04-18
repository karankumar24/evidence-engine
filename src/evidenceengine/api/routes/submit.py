"""Web routes for document submission, pipeline status, and demo seeding."""

import uuid as uuid_module
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.ext.asyncio import AsyncSession

from evidenceengine.api.dependencies import get_db, get_file_store
from evidenceengine.core.config import settings
from evidenceengine.ingestion.validation import validate_file_type
from evidenceengine.models.document import DocumentPacket, SourceDocument
from evidenceengine.models.run import RunVersion
from evidenceengine.pipeline.orchestrator import run_full_pipeline
from evidenceengine.storage.file_store import FileStore

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

router = APIRouter(tags=["submit"])

_MAX_FILE_BYTES = settings.max_file_size_mb * 1024 * 1024


# ── Upload form ───────────────────────────────────────────────────────────────

@router.get("/upload", response_class=HTMLResponse)
async def upload_form(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request=request,
        name="upload.html",
        context={"max_mb": settings.max_file_size_mb},
    )


@router.post("/upload", response_class=HTMLResponse)
async def upload_submit(
    request: Request,
    background_tasks: BackgroundTasks,
    report: Annotated[UploadFile, File(description="Report document (PDF or DOCX)")],
    sources: Annotated[
        list[UploadFile],
        File(description="Source documents (PDF or DOCX, optional)"),
    ] = [],
    db: AsyncSession = Depends(get_db),
    file_store: FileStore = Depends(get_file_store),
) -> HTMLResponse:
    packet_id = str(uuid_module.uuid4())
    all_files = [(report, "report")] + [(s, "source") for s in sources]

    # Read and size-check
    file_contents: list[tuple[UploadFile, str, bytes]] = []
    for upload_file, role in all_files:
        content = await upload_file.read()
        if len(content) > _MAX_FILE_BYTES:
            return templates.TemplateResponse(
                request=request,
                name="upload.html",
                context={"error": f"'{upload_file.filename}' exceeds {settings.max_file_size_mb}MB limit.", "max_mb": settings.max_file_size_mb},
                status_code=422,
            )
        file_contents.append((upload_file, role, content))

    # Validate file types
    for upload_file, role, content in file_contents:
        try:
            validate_file_type(content)
        except ValueError as exc:
            return templates.TemplateResponse(
                request=request,
                name="upload.html",
                context={"error": f"'{upload_file.filename}': {exc}", "max_mb": settings.max_file_size_mb},
                status_code=422,
            )

    # Save to disk
    saved_paths: dict[str, str] = {}
    for upload_file, role, content in file_contents:
        filename = upload_file.filename or "unknown"
        saved_paths[filename] = file_store.save(packet_id, filename, content)

    # Create DB records (parsing deferred to pipeline background task to keep upload fast)
    report_filename = report.filename or "report"
    packet = DocumentPacket(
        id=uuid_module.UUID(packet_id),
        status="completed",
        report_filename=report_filename,
        report_file_path=saved_paths.get(report_filename, ""),
    )
    db.add(packet)
    await db.flush()

    for upload_file, role, content in file_contents:
        filename = upload_file.filename or "unknown"
        source_doc = SourceDocument(
            packet_id=packet.id,
            is_report=(role == "report"),
            filename=filename,
            file_path=saved_paths[filename],
            file_type=_detect_file_type(filename),
            file_size_bytes=len(content),
            parse_status="pending",
        )
        db.add(source_doc)

    await db.flush()

    run = RunVersion(packet_id=packet.id, status="queued")
    db.add(run)
    await db.flush()

    # Commit now so the run/packet rows are durable before the background task starts.
    # BackgroundTasks run after the response but within the request exception scope —
    # any exception they raise would otherwise trigger the session's rollback handler.
    await db.commit()

    background_tasks.add_task(run_full_pipeline, str(run.id))

    return RedirectResponse(f"/runs/{run.id}/status", status_code=303)


# ── Pipeline status page ──────────────────────────────────────────────────────

@router.get("/runs/{run_id}/status", response_class=HTMLResponse)
async def run_status_page(
    request: Request,
    run_id: uuid_module.UUID,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    run = await db.get(RunVersion, run_id)
    if run is None:
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse(
        request=request,
        name="run_status.html",
        context={"run": run},
    )


@router.get("/runs/{run_id}/status/poll", response_class=HTMLResponse)
async def run_status_poll(
    request: Request,
    run_id: uuid_module.UUID,
    db: AsyncSession = Depends(get_db),
) -> HTMLResponse:
    run = await db.get(RunVersion, run_id)
    if run is None:
        return HTMLResponse("<p class='text-red-600 text-sm'>Run not found.</p>")

    response = templates.TemplateResponse(
        request=request,
        name="partials/run_status_fragment.html",
        context={"run": run},
    )

    if run.status == "completed":
        response.headers["HX-Redirect"] = f"/dashboard/{run.packet_id}/{run.id}"

    return response


# ── Demo seed endpoint (debug only) ──────────────────────────────────────────

if settings.debug:
    @router.post("/demo/seed", response_class=HTMLResponse)
    async def demo_seed(
        request: Request,
        db: AsyncSession = Depends(get_db),
    ) -> HTMLResponse:
        packet_id = uuid_module.uuid4()
        run_id = uuid_module.uuid4()

        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)

        packet = DocumentPacket(
            id=packet_id,
            status="completed",
            report_filename="climate_claims_sample.pdf",
            report_file_path="",
        )
        db.add(packet)
        await db.flush()

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
        db.add(source_doc)
        await db.flush()

        run = RunVersion(
            id=run_id,
            packet_id=packet_id,
            status="completed",
            started_at=now,
            completed_at=now,
            model_versions={"classification_model": "gpt-4o-mini", "reranker": "cross-encoder/ms-marco-MiniLM-L6-v2"},
        )
        db.add(run)
        await db.flush()

        from evidenceengine.models.claim import Claim
        from evidenceengine.models.evidence import EvidenceSpan
        from evidenceengine.models.verdict import Verdict

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
            db.add(claim)
            await db.flush()

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
                db.add(span)

            verdict = Verdict(
                claim_id=claim.id,
                run_version_id=run_id,
                verdict_type=verdict_type,
                confidence_score=confidence,
                reasoning=reasoning,
                model_name="gpt-4o-mini",
                prompt_version="v1",
            )
            db.add(verdict)

        await db.flush()
        return RedirectResponse(f"/dashboard/{packet_id}/{run_id}", status_code=303)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _detect_file_type(filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    return "pdf" if ext == "pdf" else "docx" if ext in ("docx", "doc") else ext


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
        "Multiple peer-reviewed studies and IPCC reports confirm a warming of ~1.1°C above pre-industrial baseline as of 2020. The evidence is robust and consistent across independent measurement datasets.",
    ),
    (
        "The Arctic is warming at more than twice the global average rate.",
        "supported",
        0.89,
        "Arctic amplification is a well-documented phenomenon. NASA and NOAA datasets show Arctic warming at 2–4× the global mean, driven by ice-albedo feedback and atmospheric circulation changes.",
    ),
    (
        "Sea levels have risen by about 20 centimeters over the past century due to thermal expansion and melting ice.",
        "insufficient_support",
        0.61,
        "The ~20 cm figure is broadly consistent with tide gauge records, but attribution to specific causes (thermal expansion vs. ice melt) varies by study. The claim is directionally correct but overly precise without additional sourcing.",
    ),
    (
        "Renewable energy sources now account for over 30% of global electricity generation.",
        "contradicted",
        0.72,
        "IEA data for 2022–2023 shows renewables at approximately 28–30% of global electricity generation, slightly below the claimed threshold. The claim may be aspirational or based on a specific region rather than global figures.",
    ),
    (
        "Carbon dioxide concentrations in the atmosphere have exceeded 420 parts per million.",
        "needs_review",
        0.55,
        "Mauna Loa Observatory confirmed CO₂ crossing 420 ppm in 2023. However, the claim that this is 'the highest in over 800,000 years' requires ice core evidence verification which is not present in the provided source documents.",
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
