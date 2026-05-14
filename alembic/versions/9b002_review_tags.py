"""add review.tags

Revision ID: 9b002
Revises: 9b001
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "9b002"
down_revision = "9b001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = sa.inspect(bind)
    cols = {c["name"] for c in insp.get_columns("reviews")}
    if "tags" not in cols:
        op.add_column("reviews", sa.Column("tags", sa.JSON(), nullable=False, server_default=sa.text("'[]'")))


def downgrade() -> None:
    op.drop_column("reviews", "tags")
