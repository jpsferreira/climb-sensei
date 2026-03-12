"""Tests for Attempt model."""

from datetime import datetime, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climb_sensei.database.models import (
    Attempt,
    Base,
    ClimbSession,
    Route,
    User,
    Video,
)
from climb_sensei.types import VideoStatus


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


@pytest.fixture
def route(db_session, user):
    r = Route(
        user_id=user.id,
        name="Test Route",
        grade="V5",
        grade_system="hueco",
        type="boulder",
    )
    db_session.add(r)
    db_session.commit()
    db_session.refresh(r)
    return r


@pytest.fixture
def video(db_session, user):
    v = Video(
        user_id=user.id,
        filename="climb.mp4",
        file_path="/uploads/climb.mp4",
        status=VideoStatus.UPLOADED,
    )
    db_session.add(v)
    db_session.commit()
    db_session.refresh(v)
    return v


@pytest.fixture
def session_obj(db_session, user):
    s = ClimbSession(
        user_id=user.id,
        name="Morning Session",
        date=datetime(2026, 3, 12, 9, 0, tzinfo=timezone.utc),
    )
    db_session.add(s)
    db_session.commit()
    db_session.refresh(s)
    return s


def test_attempt_creation(db_session, route, video):
    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime(2026, 3, 12, 10, 0, tzinfo=timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.id is not None
    assert attempt.route_id == route.id
    assert attempt.video_id == video.id
    assert attempt.session_id is None
    assert attempt.analysis_id is None
    assert attempt.notes is None


def test_attempt_with_session(db_session, route, video, session_obj):
    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        session_id=session_obj.id,
        date=datetime(2026, 3, 12, 10, 30, tzinfo=timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.session_id == session_obj.id
    assert attempt.session is not None
    assert attempt.session.name == "Morning Session"


def test_attempt_with_notes(db_session, route, video):
    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime(2026, 3, 12, 11, 0, tzinfo=timezone.utc),
        notes="Felt strong, nearly sent it.",
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.notes == "Felt strong, nearly sent it."


def test_attempt_route_relationship(db_session, route, video):
    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime(2026, 3, 12, 12, 0, tzinfo=timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.route is not None
    assert attempt.route.name == "Test Route"


def test_attempt_video_relationship(db_session, route, video):
    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime(2026, 3, 12, 13, 0, tzinfo=timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.video is not None
    assert attempt.video.filename == "climb.mp4"


def test_route_attempts_relationship(db_session, route, video):
    attempt1 = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
    )
    attempt2 = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc),
    )
    db_session.add_all([attempt1, attempt2])
    db_session.commit()
    db_session.refresh(route)

    assert len(route.attempts) == 2


def test_session_attempts_relationship(db_session, route, video, session_obj):
    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        session_id=session_obj.id,
        date=datetime(2026, 3, 12, 14, 0, tzinfo=timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(session_obj)

    assert len(session_obj.attempts) == 1
    assert session_obj.attempts[0].route_id == route.id


def test_attempt_created_at_set(db_session, route, video):
    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime(2026, 3, 12, 15, 0, tzinfo=timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.created_at is not None
    assert isinstance(attempt.created_at, datetime)


def test_attempt_repr(db_session, route, video):
    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime(2026, 3, 12, 16, 0, tzinfo=timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    r = repr(attempt)
    assert "Attempt" in r
    assert str(route.id) in r
