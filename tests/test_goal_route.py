"""Tests for Goal model with route_id field."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climb_sensei.database.models import Base, Goal, Route, User


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
        name="Goal Route",
        grade="V6",
        grade_system="hueco",
        type="boulder",
    )
    db_session.add(r)
    db_session.commit()
    db_session.refresh(r)
    return r


def test_goal_without_route_id(db_session, user):
    """Goal can be created without a route_id (null for migration compatibility)."""
    goal = Goal(
        user_id=user.id,
        metric_name="max_velocity",
        target_value=2.5,
    )
    db_session.add(goal)
    db_session.commit()
    db_session.refresh(goal)

    assert goal.id is not None
    assert goal.route_id is None
    assert goal.route is None


def test_goal_with_route_id(db_session, user, route):
    """Goal can be linked to a specific route."""
    goal = Goal(
        user_id=user.id,
        route_id=route.id,
        metric_name="send_route",
        target_value=1.0,
    )
    db_session.add(goal)
    db_session.commit()
    db_session.refresh(goal)

    assert goal.route_id == route.id
    assert goal.route is not None
    assert goal.route.name == "Goal Route"
    assert goal.route.grade == "V6"


def test_route_goals_relationship(db_session, user, route):
    """Route can have multiple goals linked to it."""
    goal1 = Goal(
        user_id=user.id,
        route_id=route.id,
        metric_name="send_route",
        target_value=1.0,
    )
    goal2 = Goal(
        user_id=user.id,
        route_id=route.id,
        metric_name="attempts_before_send",
        target_value=5.0,
    )
    db_session.add_all([goal1, goal2])
    db_session.commit()
    db_session.refresh(route)

    assert len(route.goals) == 2


def test_mixed_goals_with_and_without_route(db_session, user, route):
    """User can have goals both with and without route_id."""
    goal_with_route = Goal(
        user_id=user.id,
        route_id=route.id,
        metric_name="send_route",
        target_value=1.0,
    )
    goal_without_route = Goal(
        user_id=user.id,
        metric_name="avg_velocity",
        target_value=3.0,
    )
    db_session.add_all([goal_with_route, goal_without_route])
    db_session.commit()
    db_session.refresh(user)

    assert len(user.goals) == 2
    linked = [g for g in user.goals if g.route_id is not None]
    unlinked = [g for g in user.goals if g.route_id is None]
    assert len(linked) == 1
    assert len(unlinked) == 1
