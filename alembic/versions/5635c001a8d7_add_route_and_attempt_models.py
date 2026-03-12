"""add route and attempt models

Revision ID: 5635c001a8d7
Revises: 9b96d9a4c73a
Create Date: 2026-03-12 13:32:13.515470

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision: str = "5635c001a8d7"
down_revision: Union[str, Sequence[str], None] = "9b96d9a4c73a"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _table_exists(conn, table_name: str) -> bool:
    return inspect(conn).has_table(table_name)


def _column_exists(conn, table_name: str, column_name: str) -> bool:
    cols = [c["name"] for c in inspect(conn).get_columns(table_name)]
    return column_name in cols


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    # --- routes table ---
    if not _table_exists(bind, "routes"):
        op.create_table(
            "routes",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("user_id", sa.Integer(), nullable=False),
            sa.Column("name", sa.String(length=255), nullable=False),
            sa.Column("grade", sa.String(length=20), nullable=False),
            sa.Column("grade_system", sa.String(length=20), nullable=False),
            sa.Column("type", sa.String(length=20), nullable=False),
            sa.Column("location", sa.String(length=255), nullable=True),
            sa.Column("status", sa.String(length=20), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.Column("updated_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(
                ["user_id"],
                ["users.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("routes", schema=None) as batch_op:
            batch_op.create_index(batch_op.f("ix_routes_id"), ["id"], unique=False)
            batch_op.create_index(
                batch_op.f("ix_routes_user_id"), ["user_id"], unique=False
            )

    # --- attempts table ---
    if not _table_exists(bind, "attempts"):
        op.create_table(
            "attempts",
            sa.Column("id", sa.Integer(), nullable=False),
            sa.Column("route_id", sa.Integer(), nullable=False),
            sa.Column("video_id", sa.Integer(), nullable=False),
            sa.Column("session_id", sa.Integer(), nullable=True),
            sa.Column("analysis_id", sa.Integer(), nullable=True),
            sa.Column("notes", sa.Text(), nullable=True),
            sa.Column("date", sa.DateTime(), nullable=False),
            sa.Column("created_at", sa.DateTime(), nullable=False),
            sa.ForeignKeyConstraint(
                ["analysis_id"],
                ["analyses.id"],
            ),
            sa.ForeignKeyConstraint(
                ["route_id"],
                ["routes.id"],
            ),
            sa.ForeignKeyConstraint(
                ["session_id"],
                ["climb_sessions.id"],
            ),
            sa.ForeignKeyConstraint(
                ["video_id"],
                ["videos.id"],
            ),
            sa.PrimaryKeyConstraint("id"),
        )
        with op.batch_alter_table("attempts", schema=None) as batch_op:
            batch_op.create_index(
                batch_op.f("ix_attempts_analysis_id"), ["analysis_id"], unique=False
            )
            batch_op.create_index(batch_op.f("ix_attempts_id"), ["id"], unique=False)
            batch_op.create_index(
                batch_op.f("ix_attempts_route_id"), ["route_id"], unique=False
            )
            batch_op.create_index(
                batch_op.f("ix_attempts_session_id"), ["session_id"], unique=False
            )
            batch_op.create_index(
                batch_op.f("ix_attempts_video_id"), ["video_id"], unique=False
            )

    # --- climb_sessions.name nullable ---
    with op.batch_alter_table("climb_sessions", schema=None) as batch_op:
        batch_op.alter_column(
            "name", existing_type=sa.VARCHAR(length=255), nullable=True
        )

    # --- goals.route_id column + index + FK ---
    if not _column_exists(bind, "goals", "route_id"):
        with op.batch_alter_table("goals", schema=None) as batch_op:
            batch_op.add_column(sa.Column("route_id", sa.Integer(), nullable=True))
            batch_op.create_index(
                batch_op.f("ix_goals_route_id"), ["route_id"], unique=False
            )
            batch_op.create_foreign_key(
                "fk_goals_route_id", "routes", ["route_id"], ["id"]
            )


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table("goals", schema=None) as batch_op:
        batch_op.drop_constraint("fk_goals_route_id", type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_goals_route_id"))
        batch_op.drop_column("route_id")

    with op.batch_alter_table("climb_sessions", schema=None) as batch_op:
        batch_op.alter_column(
            "name", existing_type=sa.VARCHAR(length=255), nullable=False
        )

    with op.batch_alter_table("attempts", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_attempts_video_id"))
        batch_op.drop_index(batch_op.f("ix_attempts_session_id"))
        batch_op.drop_index(batch_op.f("ix_attempts_route_id"))
        batch_op.drop_index(batch_op.f("ix_attempts_id"))
        batch_op.drop_index(batch_op.f("ix_attempts_analysis_id"))

    op.drop_table("attempts")
    with op.batch_alter_table("routes", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_routes_user_id"))
        batch_op.drop_index(batch_op.f("ix_routes_id"))

    op.drop_table("routes")
