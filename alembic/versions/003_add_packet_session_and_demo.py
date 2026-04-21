"""add packet session_id + is_demo for visitor-scoped dashboard visibility

Adds:
  - document_packets.session_id (nullable, indexed) — the session cookie that
    owns this packet. NULL means legacy/unclaimed (invisible to every new
    visitor since no cookie will match).
  - document_packets.is_demo (boolean, default False, indexed) — when True the
    packet is visible to every visitor regardless of session_id. Used for
    seeding a public demo so the dashboard isn't empty on first visit.

Safe on existing rows: session_id defaults NULL, is_demo defaults False.
Historic rows become invisible to new visitors (desired behavior).

Revision ID: 003
Revises: 002
Create Date: 2026-04-21
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "003"
down_revision: str | None = "002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "document_packets",
        sa.Column("session_id", sa.String(length=64), nullable=True),
    )
    op.add_column(
        "document_packets",
        sa.Column("is_demo", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.create_index(
        "ix_document_packets_session_id", "document_packets", ["session_id"]
    )
    op.create_index(
        "ix_document_packets_is_demo", "document_packets", ["is_demo"]
    )


def downgrade() -> None:
    op.drop_index("ix_document_packets_is_demo", table_name="document_packets")
    op.drop_index("ix_document_packets_session_id", table_name="document_packets")
    op.drop_column("document_packets", "is_demo")
    op.drop_column("document_packets", "session_id")
