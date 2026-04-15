"""initial_schema

Revision ID: 001
Revises:
Create Date: 2026-04-15 04:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create all provenance schema tables."""

    # --- document_packets ---
    op.create_table(
        "document_packets",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("status", sa.String(50), nullable=False, server_default="queued"),
        sa.Column("report_filename", sa.Text(), nullable=False),
        sa.Column("report_file_path", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_document_packets")),
    )

    # --- source_documents ---
    op.create_table(
        "source_documents",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("packet_id", sa.UUID(), nullable=False),
        sa.Column("is_report", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("file_path", sa.Text(), nullable=False),
        sa.Column("file_type", sa.String(10), nullable=False),
        sa.Column("file_size_bytes", sa.BigInteger(), nullable=True),
        sa.Column("parsed_content", postgresql.JSONB(), nullable=True),
        sa.Column("raw_text", sa.Text(), nullable=True),
        sa.Column("markdown_text", sa.Text(), nullable=True),
        sa.Column("total_pages", sa.Integer(), nullable=True),
        sa.Column("parse_status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column("parse_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["packet_id"],
            ["document_packets.id"],
            name=op.f("fk_source_documents_packet_id_document_packets"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_source_documents")),
    )
    op.create_index(
        op.f("ix_source_documents_packet_id"),
        "source_documents",
        ["packet_id"],
    )

    # --- run_versions ---
    op.create_table(
        "run_versions",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("packet_id", sa.UUID(), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="queued"),
        sa.Column("pipeline_config", postgresql.JSONB(), nullable=True),
        sa.Column("model_versions", postgresql.JSONB(), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error_summary", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["packet_id"],
            ["document_packets.id"],
            name=op.f("fk_run_versions_packet_id_document_packets"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_run_versions")),
    )
    op.create_index(
        op.f("ix_run_versions_packet_id"),
        "run_versions",
        ["packet_id"],
    )

    # --- claims ---
    op.create_table(
        "claims",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("packet_id", sa.UUID(), nullable=False),
        sa.Column("source_document_id", sa.UUID(), nullable=False),
        sa.Column("run_version_id", sa.UUID(), nullable=False),
        sa.Column("claim_text", sa.Text(), nullable=False),
        sa.Column("section_header", sa.Text(), nullable=True),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("paragraph_index", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("status", sa.String(30), nullable=False, server_default="pending"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["packet_id"],
            ["document_packets.id"],
            name=op.f("fk_claims_packet_id_document_packets"),
        ),
        sa.ForeignKeyConstraint(
            ["run_version_id"],
            ["run_versions.id"],
            name=op.f("fk_claims_run_version_id_run_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["source_document_id"],
            ["source_documents.id"],
            name=op.f("fk_claims_source_document_id_source_documents"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_claims")),
    )
    op.create_index(op.f("ix_claims_packet_id"), "claims", ["packet_id"])
    op.create_index(
        op.f("ix_claims_source_document_id"), "claims", ["source_document_id"]
    )
    op.create_index(op.f("ix_claims_run_version_id"), "claims", ["run_version_id"])

    # --- citation_anchors ---
    op.create_table(
        "citation_anchors",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("claim_id", sa.UUID(), nullable=False),
        sa.Column("raw_marker", sa.Text(), nullable=False),
        sa.Column("citation_style", sa.String(30), nullable=False),
        sa.Column("target_document_id", sa.UUID(), nullable=True),
        sa.Column(
            "resolution_status", sa.String(30), nullable=False, server_default="pending"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["claim_id"],
            ["claims.id"],
            name=op.f("fk_citation_anchors_claim_id_claims"),
        ),
        sa.ForeignKeyConstraint(
            ["target_document_id"],
            ["source_documents.id"],
            name=op.f("fk_citation_anchors_target_document_id_source_documents"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_citation_anchors")),
    )
    op.create_index(
        op.f("ix_citation_anchors_claim_id"), "citation_anchors", ["claim_id"]
    )

    # --- evidence_spans ---
    op.create_table(
        "evidence_spans",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("claim_id", sa.UUID(), nullable=False),
        sa.Column("source_document_id", sa.UUID(), nullable=False),
        sa.Column("run_version_id", sa.UUID(), nullable=False),
        sa.Column("span_text", sa.Text(), nullable=False),
        sa.Column("page_number", sa.Integer(), nullable=True),
        sa.Column("paragraph_index", sa.Integer(), nullable=True),
        sa.Column("char_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("section_header", sa.Text(), nullable=True),
        sa.Column("relevance_score", sa.Float(), nullable=True),
        sa.Column(
            "retrieval_method", sa.String(30), nullable=False, server_default="bm25"
        ),
        sa.Column("retrieval_rank", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["claim_id"],
            ["claims.id"],
            name=op.f("fk_evidence_spans_claim_id_claims"),
        ),
        sa.ForeignKeyConstraint(
            ["run_version_id"],
            ["run_versions.id"],
            name=op.f("fk_evidence_spans_run_version_id_run_versions"),
        ),
        sa.ForeignKeyConstraint(
            ["source_document_id"],
            ["source_documents.id"],
            name=op.f("fk_evidence_spans_source_document_id_source_documents"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_evidence_spans")),
    )
    op.create_index(op.f("ix_evidence_spans_claim_id"), "evidence_spans", ["claim_id"])
    op.create_index(
        op.f("ix_evidence_spans_source_document_id"),
        "evidence_spans",
        ["source_document_id"],
    )
    op.create_index(
        op.f("ix_evidence_spans_run_version_id"), "evidence_spans", ["run_version_id"]
    )

    # --- verdicts ---
    op.create_table(
        "verdicts",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("claim_id", sa.UUID(), nullable=False),
        sa.Column("run_version_id", sa.UUID(), nullable=False),
        sa.Column("verdict_type", sa.String(30), nullable=False),
        sa.Column("confidence_score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("reasoning", sa.Text(), nullable=False),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("prompt_version", sa.String(50), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["claim_id"],
            ["claims.id"],
            name=op.f("fk_verdicts_claim_id_claims"),
        ),
        sa.ForeignKeyConstraint(
            ["run_version_id"],
            ["run_versions.id"],
            name=op.f("fk_verdicts_run_version_id_run_versions"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_verdicts")),
        sa.UniqueConstraint(
            "claim_id",
            "run_version_id",
            name="uq_verdicts_claim_id_run_version_id",
        ),
    )
    op.create_index(op.f("ix_verdicts_claim_id"), "verdicts", ["claim_id"])
    op.create_index(op.f("ix_verdicts_run_version_id"), "verdicts", ["run_version_id"])

    # --- verdict_evidence ---
    op.create_table(
        "verdict_evidence",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("verdict_id", sa.UUID(), nullable=False),
        sa.Column("evidence_span_id", sa.UUID(), nullable=False),
        sa.Column("weight", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["evidence_span_id"],
            ["evidence_spans.id"],
            name=op.f("fk_verdict_evidence_evidence_span_id_evidence_spans"),
        ),
        sa.ForeignKeyConstraint(
            ["verdict_id"],
            ["verdicts.id"],
            name=op.f("fk_verdict_evidence_verdict_id_verdicts"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_verdict_evidence")),
    )
    op.create_index(
        op.f("ix_verdict_evidence_verdict_id"), "verdict_evidence", ["verdict_id"]
    )
    op.create_index(
        op.f("ix_verdict_evidence_evidence_span_id"),
        "verdict_evidence",
        ["evidence_span_id"],
    )

    # --- review_decisions ---
    op.create_table(
        "review_decisions",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("claim_id", sa.UUID(), nullable=False),
        sa.Column("verdict_id", sa.UUID(), nullable=False),
        sa.Column("reviewer_id", sa.String(100), nullable=False),
        sa.Column("action", sa.String(30), nullable=False),
        sa.Column("verdict_at_decision", sa.String(30), nullable=False),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["claim_id"],
            ["claims.id"],
            name=op.f("fk_review_decisions_claim_id_claims"),
        ),
        sa.ForeignKeyConstraint(
            ["verdict_id"],
            ["verdicts.id"],
            name=op.f("fk_review_decisions_verdict_id_verdicts"),
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_review_decisions")),
    )
    op.create_index(
        op.f("ix_review_decisions_claim_id"), "review_decisions", ["claim_id"]
    )
    op.create_index(
        op.f("ix_review_decisions_verdict_id"), "review_decisions", ["verdict_id"]
    )


def downgrade() -> None:
    """Drop all provenance schema tables in reverse dependency order."""
    op.drop_table("review_decisions")
    op.drop_table("verdict_evidence")
    op.drop_table("verdicts")
    op.drop_table("evidence_spans")
    op.drop_table("citation_anchors")
    op.drop_table("claims")
    op.drop_table("run_versions")
    op.drop_table("source_documents")
    op.drop_table("document_packets")
