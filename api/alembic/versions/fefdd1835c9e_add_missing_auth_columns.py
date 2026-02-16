"""Add missing auth columns to users table

Revision ID: fefdd1835c9e
Revises: fefdd1835b7d
Create Date: 2026-02-16 14:20:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "fefdd1835c9e"
down_revision: Union[str, None] = "34c8537dfde5"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add email and hashed_password columns to users table
    op.add_column("users", sa.Column("email", sa.String(), nullable=True))
    op.add_column("users", sa.Column("hashed_password", sa.String(), nullable=True))
    
    # Create index on email column
    op.create_index(op.f("ix_users_email"), "users", ["email"], unique=True)


def downgrade() -> None:
    # Remove index and columns
    op.drop_index(op.f("ix_users_email"), table_name="users")
    op.drop_column("users", "hashed_password")
    op.drop_column("users", "email")
