"""add error_message column and performance indexes

Revision ID: a1b2c3d4e5f6
Revises: 5635c001a8d7
Create Date: 2026-03-28 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "5635c001a8d7"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add error_message column to videos table
    op.add_column("videos", sa.Column("error_message", sa.Text(), nullable=True))

    # Add performance indexes
    op.create_index("ix_attempts_date", "attempts", ["date"])
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])
    op.create_index("ix_goals_metric_name", "goals", ["metric_name"])


def downgrade() -> None:
    op.drop_index("ix_goals_metric_name", "goals")
    op.drop_index("ix_analyses_created_at", "analyses")
    op.drop_index("ix_attempts_date", "attempts")
    op.drop_column("videos", "error_message")
