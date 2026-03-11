"""Database configuration and session management.

This module provides database engine configuration, session management,
and FastAPI dependency injection utilities.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./climbsensei.db")

# Create engine
# For SQLite, need to enable foreign keys
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions.

    Yields:
        Session: SQLAlchemy database session

    Example:
        ```python
        @app.get("/users")
        def get_users(db: Session = Depends(get_db)):
            return db.query(User).all()
        ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Initialize database by creating all tables.

    This should be called once when setting up the application
    or can be replaced by Alembic migrations in production.
    """
    from climb_sensei.database.models import Base

    Base.metadata.create_all(bind=engine)


def drop_all_tables() -> None:
    """Drop all tables from the database.

    WARNING: This will delete all data! Only use in development/testing.
    """
    from climb_sensei.database.models import Base

    Base.metadata.drop_all(bind=engine)
