"""Database configuration and session management.

This module provides database engine configuration, session management,
and FastAPI dependency injection utilities.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator, AsyncGenerator

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./climbsensei.db")

# Map sync DB URL to async equivalent for fastapi-users
_ASYNC_DRIVER_MAP = {
    "sqlite://": "sqlite+aiosqlite://",
    "postgresql://": "postgresql+asyncpg://",
    "postgresql+psycopg2://": "postgresql+asyncpg://",
}


def _to_async_url(url: str) -> str:
    """Convert a sync database URL to its async equivalent."""
    for sync_prefix, async_prefix in _ASYNC_DRIVER_MAP.items():
        if url.startswith(sync_prefix):
            return url.replace(sync_prefix, async_prefix, 1)
    raise ValueError(
        f"Unsupported DATABASE_URL scheme for async: {url.split('://')[0]}. "
        f"Supported: {', '.join(_ASYNC_DRIVER_MAP.keys())}"
    )


ASYNC_DATABASE_URL = _to_async_url(DATABASE_URL)

# Create sync engine (for existing code)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

# Create async engine (required by fastapi-users)
# Note: Most application code uses sync sessions via SessionLocal.
# The async engine is used exclusively by fastapi-users for auth operations.
async_engine = create_async_engine(ASYNC_DATABASE_URL, echo=False)

# Create session factories
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
AsyncSessionLocal = async_sessionmaker(
    async_engine, class_=AsyncSession, expire_on_commit=False
)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency for database sessions (sync).

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


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for async database sessions.

    Yields:
        AsyncSession: SQLAlchemy async database session
    """
    async with AsyncSessionLocal() as session:
        yield session


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
