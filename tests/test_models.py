"""Tests for all SQLAlchemy ORM models (provenance schema)."""

import pytest
from sqlalchemy import inspect


# --------------------------------------------------------------------------- #
# Test 1: All model classes are importable from evidenceengine.models
# --------------------------------------------------------------------------- #
def test_all_models_importable():
    from evidenceengine.models import (  # noqa: F401
        Base,
        CitationAnchor,
        Claim,
        DocumentPacket,
        EvidenceSpan,
        ReviewDecision,
        RunVersion,
        SourceDocument,
        Verdict,
        VerdictEvidence,
    )


# --------------------------------------------------------------------------- #
# Test 2: DocumentPacket has UUID PK, status field, timestamps, and relationship
# --------------------------------------------------------------------------- #
def test_document_packet_columns():
    from evidenceengine.models import DocumentPacket

    table = DocumentPacket.__table__
    col_names = {c.name for c in table.columns}
    assert "id" in col_names
    assert "status" in col_names
    assert "created_at" in col_names
    assert "updated_at" in col_names
    assert "report_filename" in col_names
    assert "report_file_path" in col_names


def test_document_packet_relationship():
    from evidenceengine.models import DocumentPacket

    assert hasattr(DocumentPacket, "source_documents"), "DocumentPacket must have source_documents relationship"


# --------------------------------------------------------------------------- #
# Test 3: SourceDocument has packet_id FK, file_type, raw_text, parsed_content,
#          total_pages, parse_status fields
# --------------------------------------------------------------------------- #
def test_source_document_columns():
    from evidenceengine.models import SourceDocument

    table = SourceDocument.__table__
    col_names = {c.name for c in table.columns}
    assert "id" in col_names
    assert "packet_id" in col_names
    assert "is_report" in col_names
    assert "filename" in col_names
    assert "file_path" in col_names
    assert "file_type" in col_names
    assert "file_size_bytes" in col_names
    assert "raw_text" in col_names
    assert "parsed_content" in col_names
    assert "markdown_text" in col_names
    assert "total_pages" in col_names
    assert "parse_status" in col_names
    assert "parse_error" in col_names


def test_source_document_fk_to_packet():
    from evidenceengine.models import SourceDocument

    fk_cols = {fk.column.table.name for fk in SourceDocument.__table__.foreign_keys}
    assert "document_packets" in fk_cols, "SourceDocument.packet_id must FK to document_packets"


# --------------------------------------------------------------------------- #
# Test 4: All 8+ tables have correct FK chains
# --------------------------------------------------------------------------- #
def test_claim_fk_chains():
    from evidenceengine.models import CitationAnchor, Claim

    claim_fks = {fk.column.table.name for fk in Claim.__table__.foreign_keys}
    assert "document_packets" in claim_fks
    assert "source_documents" in claim_fks
    assert "run_versions" in claim_fks

    anchor_fks = {fk.column.table.name for fk in CitationAnchor.__table__.foreign_keys}
    assert "claims" in anchor_fks


def test_evidence_span_fk_chains():
    from evidenceengine.models import EvidenceSpan

    fks = {fk.column.table.name for fk in EvidenceSpan.__table__.foreign_keys}
    assert "claims" in fks
    assert "source_documents" in fks
    assert "run_versions" in fks


def test_verdict_fk_chains():
    from evidenceengine.models import Verdict, VerdictEvidence

    verdict_fks = {fk.column.table.name for fk in Verdict.__table__.foreign_keys}
    assert "claims" in verdict_fks
    assert "run_versions" in verdict_fks

    ve_fks = {fk.column.table.name for fk in VerdictEvidence.__table__.foreign_keys}
    assert "verdicts" in ve_fks
    assert "evidence_spans" in ve_fks


def test_review_decision_fk_chains():
    from evidenceengine.models import ReviewDecision

    fks = {fk.column.table.name for fk in ReviewDecision.__table__.foreign_keys}
    assert "claims" in fks
    assert "verdicts" in fks


def test_run_version_fk_to_packet():
    from evidenceengine.models import RunVersion

    fks = {fk.column.table.name for fk in RunVersion.__table__.foreign_keys}
    assert "document_packets" in fks


# --------------------------------------------------------------------------- #
# Test 5: All 8+ tables exist in Base.metadata
# --------------------------------------------------------------------------- #
def test_all_tables_in_metadata():
    from evidenceengine.models import Base

    table_names = set(Base.metadata.tables.keys())
    expected = {
        "document_packets",
        "source_documents",
        "claims",
        "citation_anchors",
        "evidence_spans",
        "verdicts",
        "verdict_evidence",
        "review_decisions",
        "run_versions",
    }
    missing = expected - table_names
    assert not missing, f"Missing tables in metadata: {missing}"
