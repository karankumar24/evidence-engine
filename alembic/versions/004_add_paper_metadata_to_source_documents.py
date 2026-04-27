"""add paper_metadata to source_documents

Revision ID: 004
Revises: 003
Create Date: 2026-04-26

Adds a nullable JSONB column paper_metadata to source_documents.
Used for citation anchor resolution: matching citation markers to uploaded
papers by title+authors is more reliable than filename fuzzy-matching.

Schema: {"title": str, "authors": [str, ...], "year": str}

Existing rows get NULL (handled gracefully in anchor_resolver.py).
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "source_documents",
        sa.Column("paper_metadata", JSONB, nullable=True),
    )


def downgrade() -> None:
    op.drop_column("source_documents", "paper_metadata")
