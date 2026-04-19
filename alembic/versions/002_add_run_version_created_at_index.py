"""add_run_version_created_at_index

Revision ID: 002
Revises: 001
Create Date: 2026-04-19

"""

from collections.abc import Sequence

from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index("ix_run_versions_created_at", "run_versions", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_run_versions_created_at", table_name="run_versions")
