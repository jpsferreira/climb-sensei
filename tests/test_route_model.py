"""Tests for Route model."""

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climb_sensei.database.models import Base, Route, User


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


def test_route_creation(db_session, user):
    route = Route(
        user_id=user.id,
        name="The Nose",
        grade="V7",
        grade_system="hueco",
        type="boulder",
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    assert route.id is not None
    assert route.name == "The Nose"
    assert route.grade == "V7"
    assert route.grade_system == "hueco"
    assert route.type == "boulder"
    assert route.user_id == user.id


def test_route_default_status(db_session, user):
    route = Route(
        user_id=user.id,
        name="Project Zero",
        grade="5.12a",
        grade_system="yds",
        type="sport",
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    assert route.status == "projecting"


def test_route_sent_status(db_session, user):
    route = Route(
        user_id=user.id,
        name="Old Faithful",
        grade="6a",
        grade_system="french",
        type="sport",
        status="sent",
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    assert route.status == "sent"


def test_route_optional_location(db_session, user):
    route_with = Route(
        user_id=user.id,
        name="Crag Climb",
        grade="V3",
        grade_system="hueco",
        type="boulder",
        location="Magic Wood",
    )
    route_without = Route(
        user_id=user.id,
        name="Gym Problem",
        grade="V3",
        grade_system="hueco",
        type="boulder",
    )
    db_session.add_all([route_with, route_without])
    db_session.commit()

    assert route_with.location == "Magic Wood"
    assert route_without.location is None


def test_route_timestamps_set_on_creation(db_session, user):
    route = Route(
        user_id=user.id,
        name="Time Test",
        grade="V5",
        grade_system="hueco",
        type="boulder",
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    assert route.created_at is not None
    assert route.updated_at is not None
    assert isinstance(route.created_at, datetime)


def test_route_user_relationship(db_session, user):
    route = Route(
        user_id=user.id,
        name="Relationship Test",
        grade="V4",
        grade_system="hueco",
        type="boulder",
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    assert route.user is not None
    assert route.user.id == user.id
    assert route.user.email == "climber@example.com"


def test_user_routes_relationship(db_session, user):
    route1 = Route(
        user_id=user.id,
        name="First Route",
        grade="V1",
        grade_system="hueco",
        type="boulder",
    )
    route2 = Route(
        user_id=user.id,
        name="Second Route",
        grade="V2",
        grade_system="hueco",
        type="boulder",
    )
    db_session.add_all([route1, route2])
    db_session.commit()
    db_session.refresh(user)

    assert len(user.routes) == 2


def test_route_repr(db_session, user):
    route = Route(
        user_id=user.id,
        name="Repr Test",
        grade="V6",
        grade_system="hueco",
        type="boulder",
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    r = repr(route)
    assert "Route" in r
    assert "Repr Test" in r
    assert "V6" in r
