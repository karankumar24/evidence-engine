"""Retrieval pipeline orchestrator.

Loads Claims and CitationAnchors for a run, builds/loads BM25 indexes per cited source
document (cached in-memory within this run to avoid re-indexing for repeated citations),
queries BM25 top-k, reranks with cross-encoder, and persists EvidenceSpan rows.

This function is designed to be called by Phase 5 orchestration. It accepts
run_version_id and db as parameters — it does NOT create or update RunVersion status.
The caller (API endpoint or Phase 5 task runner) is responsible for RunVersion lifecycle.
"""

import logging
import os
import uuid as uuid_lib
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from evidenceengine.core.config import settings
from evidenceengine.models.claim import Claim
from evidenceengine.models.document import SourceDocument
from evidenceengine.models.evidence import EvidenceSpan
from evidenceengine.retrieval.bm25_retriever import load_or_build_index, query_index
from evidenceengine.retrieval.dense_retriever import load_or_build_dense, query_dense
from evidenceengine.retrieval.indexer import extract_spans
from evidenceengine.retrieval.recall_logger import log_retrieval_metrics
from evidenceengine.retrieval.reranker import rerank

logger = logging.getLogger(__name__)


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    """True if [a_start, a_end) and [b_start, b_end) overlap at all."""
    return a_start < b_end and b_start < a_end


async def retrieve_evidence_for_run(
    run_version_id: str,
    db: AsyncSession,
) -> list[EvidenceSpan]:
    """Run the full retrieval pipeline for all claims in a run.

    For each claim with at least one resolved CitationAnchor:
    1. Load the cited SourceDocument (scoped per anchor — RETRIEV-02)
    2. Build/load BM25 index (cached by source_document_id in this run)
    3. Query BM25 top-k candidates
    4. Rerank with cross-encoder
    5. Persist top-k reranked EvidenceSpan rows

    Claims with ALL anchors unresolvable are skipped for retrieval but counted
    in the recall denominator.

    After all claims processed, logs recall@k to RunVersion.pipeline_config.

    Returns: list of all persisted EvidenceSpan objects.
    """
    # 1. Load all claims for this run with their citation anchors
    result = await db.execute(
        select(Claim)
        .where(Claim.run_version_id == run_version_id)
        .options(selectinload(Claim.citation_anchors))
    )
    claims = result.scalars().all()

    if not claims:
        logger.warning("No claims found for run_version_id=%s", run_version_id)
        return []

    total_claim_count = len(claims)
    retrieved_claim_count = 0
    all_spans: list[EvidenceSpan] = []

    # 1b. Load cited (non-report) source documents for this packet.
    # Used as fallback corpus when a claim has no resolved citation anchors —
    # searching cited sources is strictly better than self-verifying the report.
    packet_id = claims[0].packet_id
    cited_result = await db.execute(
        select(SourceDocument.id).where(
            SourceDocument.packet_id == packet_id,
            SourceDocument.is_report == False,  # noqa: E712
        )
    )
    cited_source_ids: list[str] = [str(row[0]) for row in cited_result.fetchall()]

    # In-run index cache: {source_document_id_str -> (retriever, span_dicts, dense_embs|None)}
    # Avoids re-indexing/re-encoding when multiple claims cite the same source document.
    index_cache: dict[str, tuple[Any, list[dict], Any]] = {}

    for claim in claims:
        resolved_anchors = [
            a for a in claim.citation_anchors
            if a.resolution_status == "resolved" and a.target_document_id is not None
        ]

        # Build the list of (target_doc_id, anchor_or_None) pairs to search.
        # Priority 1: resolved citation anchors → search only the anchored docs.
        # Priority 2: uploaded cited sources → search all non-report docs in packet.
        # Priority 3: no cited sources → self-verification (report corpus).
        # Distinguish "claim has citations but none resolve" from "claim has no citations".
        # When a claim cites papers that aren't in the packet (e.g. cites BERT but only
        # Transformer was uploaded), searching the uploaded paper for evidence produces
        # false contradictions — NLI sees unrelated content with different numbers and
        # confidently rules "contradicted". Skip retrieval; classification short-circuits
        # to needs_review with reason "cited sources unavailable".
        has_citation_markers = bool(claim.citation_anchors)
        if resolved_anchors:
            search_targets = [(str(a.target_document_id), a) for a in resolved_anchors]
        elif has_citation_markers:
            logger.debug(
                "Claim %s has citation markers but none resolve to uploaded sources — skipping retrieval",
                claim.id,
            )
            continue  # no spans → classification marks needs_review
        elif cited_source_ids:
            search_targets = [(doc_id, None) for doc_id in cited_source_ids]
            logger.debug(
                "Claim %s has no citation markers — searching %d cited source(s) for context",
                claim.id,
                len(cited_source_ids),
            )
        else:
            search_targets = [(str(claim.source_document_id), None)]
            logger.debug(
                "Claim %s self-verifying against report (no citations, no cited sources)",
                claim.id,
            )

        claim_has_spans = False
        # Rank across ALL searched documents for this claim, so downstream
        # ordering is meaningful when a claim cites multiple sources.
        claim_global_rank = 0

        for doc_id_str, anchor in search_targets:

            target_doc_uuid = uuid_lib.UUID(doc_id_str)

            # Load and cache BM25 index for this source document
            if doc_id_str not in index_cache:
                source_doc_result = await db.execute(
                    select(SourceDocument).where(
                        SourceDocument.id == target_doc_uuid
                    )
                )
                source_doc = source_doc_result.scalar_one_or_none()
                if source_doc is None or not source_doc.parsed_content:
                    logger.warning(
                        "Source document %s has no parsed_content — skipping anchor",
                        doc_id_str,
                    )
                    continue

                span_dicts = extract_spans(source_doc.parsed_content)
                if not span_dicts:
                    logger.warning("No spans extracted from document %s", doc_id_str)
                    continue

                span_texts = [s["text"] for s in span_dicts]
                index_path = os.path.join(settings.index_dir, doc_id_str)
                retriever = load_or_build_index(span_texts, index_path)
                dense_embs = (
                    load_or_build_dense(span_texts, index_path)
                    if settings.dense_retrieval_enabled
                    else None
                )
                index_cache[doc_id_str] = (retriever, span_dicts, dense_embs)

            if doc_id_str not in index_cache:
                continue  # failed to build index above

            retriever, span_dicts, dense_embs = index_cache[doc_id_str]
            span_texts = [s["text"] for s in span_dicts]

            # BM25 top-k query
            bm25_texts, _bm25_scores = query_index(
                retriever,
                claim.claim_text,
                k=settings.retrieval_top_k_bm25,
            )

            # In self-verification mode, exclude any span that overlaps the
            # claim's own text position in the source document. Without this
            # filter, BM25 rank-1 is always the paragraph CONTAINING the claim
            # (not just the exact claim text), producing circular verdicts like
            # "SUPPORTED 95%" where the only evidence is the claim itself.
            # overlapping_texts is kept outside the if-block so the dense union
            # below can apply the same filter without re-computing overlap.
            # Guard: only apply when searching the claim's own source document.
            # In multi-doc fallback mode anchor=None but doc_id differs from
            # claim.source_document_id — char positions are document-local and
            # comparing across documents produces spurious overlap hits.
            overlapping_texts: set[str] = set()
            if anchor is None and doc_id_str == str(claim.source_document_id):
                claim_start = claim.char_start or 0
                claim_end = claim.char_end or 0
                overlapping_texts = {
                    sd["text"]
                    for sd in span_dicts
                    if _overlaps(
                        sd.get("char_start", 0), sd.get("char_end", 0),
                        claim_start, claim_end,
                    )
                }
                bm25_texts = [t for t in bm25_texts if t not in overlapping_texts]

            # Dense retrieval union — appends semantically similar candidates
            # that keyword search misses (e.g. paraphrased claims). Applies the
            # same overlap filter as BM25 to prevent self-citation in self-verify
            # mode. New candidates are appended after BM25 results so the
            # cross-encoder reranker sees the full candidate set.
            if settings.dense_retrieval_enabled and dense_embs is not None:
                dense_k = max(2, settings.retrieval_top_k_bm25 // 2)
                dense_candidates = query_dense(
                    claim.claim_text, dense_embs, span_texts, k=dense_k,
                )
                if overlapping_texts:
                    dense_candidates = [t for t in dense_candidates if t not in overlapping_texts]
                bm25_set = set(bm25_texts)
                for t in dense_candidates:
                    if t not in bm25_set:
                        bm25_texts.append(t)

            if not bm25_texts:
                continue

            # Cross-encoder reranking over BM25 ∪ dense candidates.
            ranked = await rerank(claim.claim_text, bm25_texts)
            final_spans = ranked[: settings.retrieval_top_k_final]

            if not final_spans:
                continue

            # Build a lookup from span text → span metadata dict
            text_to_meta: dict[str, dict] = {s["text"]: s for s in span_dicts}

            # Persist EvidenceSpan rows
            for rank_idx, item in enumerate(final_spans):
                meta = text_to_meta.get(item["text"])
                if meta is None:
                    logger.warning(
                        "span text not in index for claim %s rank %d — position metadata zeroed",
                        claim.id, rank_idx,
                    )
                    meta = {}
                claim_global_rank += 1
                span = EvidenceSpan(
                    claim_id=claim.id,
                    source_document_id=target_doc_uuid,
                    run_version_id=uuid_lib.UUID(run_version_id),
                    span_text=item["text"],
                    page_number=meta.get("page"),
                    paragraph_index=meta.get("paragraph"),
                    char_start=meta.get("char_start", 0),
                    char_end=meta.get("char_end", 0),
                    section_header=meta.get("section_header"),
                    relevance_score=item["score"],
                    retrieval_method="cross_encoder",
                    retrieval_rank=claim_global_rank,
                )
                db.add(span)
                all_spans.append(span)
            claim_has_spans = True

        if claim_has_spans:
            retrieved_claim_count += 1

    await db.flush()

    # Log recall@k to RunVersion.pipeline_config
    await log_retrieval_metrics(
        run_version_id=run_version_id,
        claim_count=total_claim_count,
        retrieved_count=retrieved_claim_count,
        k=settings.retrieval_top_k_final,
        db=db,
    )

    await db.commit()
    logger.info(
        "Retrieval complete: %d spans from %d/%d claims for run %s",
        len(all_spans),
        retrieved_claim_count,
        total_claim_count,
        run_version_id,
    )
    return all_spans
