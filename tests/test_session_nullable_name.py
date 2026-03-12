"""Tests for ClimbSession.name being nullable."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climb_sensei.database.models import Base, ClimbSession, User


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def user(db_session):
    u = User(
        email="climber@example.com",
        hashed_password="hashed",
        is_active=True,
        is_superuser=False,
        is_verified=False,
    )
    db_session.add(u)
    db_session.commit()
    db_session.refresh(u)
    return u


def test_session_with_name(db_session, user):
    """ClimbSession can still be created with a name."""
    s = ClimbSession(
        user_id=user.id,
        name="Morning Gym Session",
        date=datetime(2026, 3, 12, 9, 0, tzinfo=timezone.utc),
    )
    db_session.add(s)
    db_session.commit()
    db_session.refresh(s)

    assert s.name == "Morning Gym Session"


def test_session_without_name(db_session, user):
    """ClimbSession can be created without a name (nullable)."""
    s = ClimbSession(
        user_id=user.id,
        date=datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc),
    )
    db_session.add(s)
    db_session.commit()
    db_session.refresh(s)

    assert s.id is not None
    assert s.name is None


def test_session_name_explicitly_none(db_session, user):
    """ClimbSession.name can be explicitly set to None."""
    s = ClimbSession(
        user_id=user.id,
        name=None,
        date=datetime(2026, 3, 12, 11, 0, tzinfo=timezone.utc),
    )
    db_session.add(s)
    db_session.commit()
    db_session.refresh(s)

    assert s.name is None


def test_session_name_can_be_cleared(db_session, user):
    """ClimbSession.name can be set to None after creation."""
    s = ClimbSession(
        user_id=user.id,
        name="Temporary Name",
        date=datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc),
    )
    db_session.add(s)
    db_session.commit()

    s.name = None
    db_session.commit()
    db_session.refresh(s)

    assert s.name is None
