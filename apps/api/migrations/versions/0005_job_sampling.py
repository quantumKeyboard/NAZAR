"""job frame sampling

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-05 10:00:00
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: str | None = "0004"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.add_column(sa.Column("fps", sa.Float(), nullable=False, server_default="1.0"))
        batch_op.add_column(sa.Column("frame_count", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("fps_effective", sa.Float(), nullable=True))
        batch_op.add_column(
            sa.Column("frames_clamped", sa.Boolean(), nullable=False, server_default=sa.false())
        )


def downgrade() -> None:
    with op.batch_alter_table("jobs", schema=None) as batch_op:
        batch_op.drop_column("frames_clamped")
        batch_op.drop_column("fps_effective")
        batch_op.drop_column("frame_count")
        batch_op.drop_column("fps")
