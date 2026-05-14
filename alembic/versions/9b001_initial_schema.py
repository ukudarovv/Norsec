"""initial schema

Revision ID: 9b001
Revises:
Create Date: 2026-05-14

"""

from __future__ import annotations

from alembic import op

from api.db.base import Base

revision = "9b001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    bind = op.get_bind()
    Base.metadata.drop_all(bind=bind)
