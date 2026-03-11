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

# For async support, convert sqlite:/// to sqlite+aiosqlite:///
ASYNC_DATABASE_URL = DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")

# Create sync engine (for existing code)
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, echo=False)

# Create async engine (for fastapi-users)
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
