"""Tests for Route CRUD API endpoints.

Covers:
- GET /api/routes (list, filter by type, search by name, sort)
- POST /api/routes (create)
- GET /api/routes/{id} (get detail)
- PATCH /api/routes/{id} (update)
- DELETE /api/routes/{id} (delete)
"""

import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from climb_sensei.database.models import Base, User, Route, Attempt, Video
from climb_sensei.database.config import get_db
from climb_sensei.auth import get_current_active_user, get_password_hash


# ========== Fixtures ==========


@pytest.fixture
def test_engine():
    """Create in-memory SQLite database for testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture
def db_session(test_engine):
    """Create a test database session."""
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_engine
    )
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def test_user(db_session):
    """Create and persist a test user."""
    user = User(
        email="climber@example.com",
        hashed_password=get_password_hash("secret123"),
        full_name="Climber One",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user(db_session):
    """Create a second user to verify isolation."""
    user = User(
        email="other@example.com",
        hashed_password=get_password_hash("secret123"),
        full_name="Climber Two",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def client(test_engine, db_session, test_user):
    """Create TestClient with DB and auth overrides."""
    # Import app here to avoid issues with module-level app creation
    from app.main import create_app

    application = create_app()

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    def override_get_current_active_user():
        return test_user

    auth_dep = get_current_active_user
    application.dependency_overrides[get_db] = override_get_db
    application.dependency_overrides[auth_dep] = override_get_current_active_user

    with TestClient(application) as c:
        yield c

    application.dependency_overrides.clear()


@pytest.fixture
def sample_route(db_session, test_user):
    """Create a sample route for test_user."""
    route = Route(
        user_id=test_user.id,
        name="The Nose",
        grade="V5",
        grade_system="hueco",
        type="boulder",
        location="Yosemite",
        status="projecting",
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)
    return route


@pytest.fixture
def sample_video(db_session, test_user):
    """Create a sample video record."""
    video = Video(
        user_id=test_user.id,
        filename="climb.mp4",
        file_path="/uploads/climb.mp4",
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    return video


# ========== List Routes Tests ==========


class TestListRoutes:
    def test_list_routes_empty(self, client):
        """Should return empty list when user has no routes."""
        response = client.get("/api/v1/routes")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_routes_returns_own_routes(self, client, sample_route):
        """Should return routes belonging to the current user."""
        response = client.get("/api/v1/routes")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == sample_route.id
        assert data[0]["name"] == "The Nose"

    def test_list_routes_excludes_other_users(
        self, client, db_session, other_user, sample_route
    ):
        """Should not return routes from other users."""
        other_route = Route(
            user_id=other_user.id,
            name="Other Route",
            grade="V3",
            grade_system="hueco",
            type="boulder",
        )
        db_session.add(other_route)
        db_session.commit()

        response = client.get("/api/v1/routes")
        assert response.status_code == 200
        ids = [r["id"] for r in response.json()]
        assert other_route.id not in ids

    def test_list_routes_filter_by_type(self, client, db_session, test_user):
        """Should filter routes by type."""
        boulder = Route(
            user_id=test_user.id,
            name="Boulder Problem",
            grade="V4",
            grade_system="hueco",
            type="boulder",
        )
        sport = Route(
            user_id=test_user.id,
            name="Sport Route",
            grade="5.11a",
            grade_system="yds",
            type="sport",
        )
        db_session.add_all([boulder, sport])
        db_session.commit()

        response = client.get("/api/v1/routes?type=boulder")
        assert response.status_code == 200
        data = response.json()
        assert all(r["type"] == "boulder" for r in data)
        names = [r["name"] for r in data]
        assert "Boulder Problem" in names
        assert "Sport Route" not in names

    def test_list_routes_search_by_name(self, client, db_session, test_user):
        """Should search routes by name substring (case-insensitive)."""
        r1 = Route(
            user_id=test_user.id,
            name="Midnight Lightning",
            grade="V8",
            grade_system="hueco",
            type="boulder",
        )
        r2 = Route(
            user_id=test_user.id,
            name="Separate Reality",
            grade="5.11d",
            grade_system="yds",
            type="sport",
        )
        db_session.add_all([r1, r2])
        db_session.commit()

        response = client.get("/api/v1/routes?search=midnight")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "Midnight Lightning"

    def test_list_routes_sort_by_attempts(
        self, client, db_session, test_user, sample_video
    ):
        """Should sort routes by attempt count descending."""
        r1 = Route(
            user_id=test_user.id,
            name="Few Attempts",
            grade="V2",
            grade_system="hueco",
            type="boulder",
        )
        r2 = Route(
            user_id=test_user.id,
            name="Many Attempts",
            grade="V6",
            grade_system="hueco",
            type="boulder",
        )
        db_session.add_all([r1, r2])
        db_session.commit()

        # Add 3 attempts to r2, 1 to r1
        for _ in range(3):
            attempt = Attempt(
                route_id=r2.id,
                video_id=sample_video.id,
                date=datetime.now(timezone.utc),
            )
            db_session.add(attempt)
        single_attempt = Attempt(
            route_id=r1.id,
            video_id=sample_video.id,
            date=datetime.now(timezone.utc),
        )
        db_session.add(single_attempt)
        db_session.commit()

        response = client.get("/api/v1/routes?sort=attempts")
        assert response.status_code == 200
        data = response.json()
        counts = [r["attempt_count"] for r in data]
        assert counts == sorted(counts, reverse=True)

    def test_list_routes_includes_sparkline(self, client, sample_route):
        """Should include sparkline list (empty when no attempts with analyses)."""
        response = client.get("/api/v1/routes")
        assert response.status_code == 200
        data = response.json()
        assert "sparkline" in data[0]
        assert isinstance(data[0]["sparkline"], list)


# ========== Create Route Tests ==========


class TestCreateRoute:
    def test_create_route_success(self, client):
        """Should create a route and return 201."""
        payload = {
            "name": "New Problem",
            "grade": "V7",
            "grade_system": "hueco",
            "type": "boulder",
            "location": "Bishop",
        }
        response = client.post("/api/v1/routes", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "New Problem"
        assert data["grade"] == "V7"
        assert data["grade_system"] == "hueco"
        assert data["type"] == "boulder"
        assert data["location"] == "Bishop"
        assert data["status"] == "projecting"
        assert data["attempt_count"] == 0
        assert data["last_attempt_date"] is None

    def test_create_route_without_location(self, client):
        """Should create a route with optional location omitted."""
        payload = {
            "name": "No Location Route",
            "grade": "6a",
            "grade_system": "french",
            "type": "sport",
        }
        response = client.post("/api/v1/routes", json=payload)
        assert response.status_code == 201
        data = response.json()
        assert data["location"] is None

    def test_create_route_invalid_grade_system(self, client):
        """Should reject invalid grade_system value."""
        payload = {
            "name": "Bad Route",
            "grade": "V5",
            "grade_system": "vscale",  # invalid
            "type": "boulder",
        }
        response = client.post("/api/v1/routes", json=payload)
        assert response.status_code == 422

    def test_create_route_invalid_type(self, client):
        """Should reject invalid route type."""
        payload = {
            "name": "Bad Route",
            "grade": "V5",
            "grade_system": "hueco",
            "type": "deepwater",  # invalid
        }
        response = client.post("/api/v1/routes", json=payload)
        assert response.status_code == 422


# ========== Get Route Tests ==========


class TestGetRoute:
    def test_get_route_success(self, client, sample_route):
        """Should return route detail for owned route."""
        response = client.get(f"/api/v1/routes/{sample_route.id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_route.id
        assert data["name"] == "The Nose"
        assert "attempt_count" in data
        assert "last_attempt_date" in data

    def test_get_route_not_found(self, client):
        """Should return 404 for non-existent route."""
        response = client.get("/api/v1/routes/99999")
        assert response.status_code == 404

    def test_get_route_other_user_not_found(self, client, db_session, other_user):
        """Should return 404 for a route owned by another user."""
        other_route = Route(
            user_id=other_user.id,
            name="Someone Else Route",
            grade="V3",
            grade_system="hueco",
            type="boulder",
        )
        db_session.add(other_route)
        db_session.commit()

        response = client.get(f"/api/v1/routes/{other_route.id}")
        assert response.status_code == 404


# ========== Update Route Tests ==========


class TestUpdateRoute:
    def test_update_route_name(self, client, sample_route):
        """Should update route name."""
        response = client.patch(
            f"/api/v1/routes/{sample_route.id}", json={"name": "Renamed Route"}
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Renamed Route"

    def test_update_route_status_to_sent(self, client, sample_route):
        """Should update status to sent."""
        response = client.patch(
            f"/api/v1/routes/{sample_route.id}", json={"status": "sent"}
        )
        assert response.status_code == 200
        assert response.json()["status"] == "sent"

    def test_update_route_partial(self, client, sample_route):
        """Should update only the provided fields."""
        response = client.patch(
            f"/api/v1/routes/{sample_route.id}", json={"location": "Updated Crag"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["location"] == "Updated Crag"
        assert data["name"] == "The Nose"  # unchanged

    def test_update_route_invalid_status(self, client, sample_route):
        """Should reject invalid status value."""
        response = client.patch(
            f"/api/v1/routes/{sample_route.id}", json={"status": "climbed"}
        )
        assert response.status_code == 422

    def test_update_route_not_found(self, client):
        """Should return 404 for non-existent route."""
        response = client.patch("/api/v1/routes/99999", json={"name": "Ghost"})
        assert response.status_code == 404


# ========== Delete Route Tests ==========


class TestDeleteRoute:
    def test_delete_route_success(self, client, db_session, sample_route):
        """Should delete route and return 204."""
        response = client.delete(f"/api/v1/routes/{sample_route.id}")
        assert response.status_code == 204

        # Verify it's gone from DB
        from climb_sensei.database.models import Route as RouteModel

        deleted = db_session.get(RouteModel, sample_route.id)
        assert deleted is None

    def test_delete_route_not_found(self, client):
        """Should return 404 for non-existent route."""
        response = client.delete("/api/v1/routes/99999")
        assert response.status_code == 404

    def test_delete_route_other_user_not_found(self, client, db_session, other_user):
        """Should return 404 when trying to delete another user's route."""
        other_route = Route(
            user_id=other_user.id,
            name="Protected Route",
            grade="V5",
            grade_system="hueco",
            type="boulder",
        )
        db_session.add(other_route)
        db_session.commit()

        response = client.delete(f"/api/v1/routes/{other_route.id}")
        assert response.status_code == 404
