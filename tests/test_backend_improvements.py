"""Tests for backend improvements: auth rate limiting, error messages, indexes.

Covers:
- Auth rate limiting middleware (5 req/min on login/register)
- Error message storage on failed video analysis
- Video status endpoint returns error_message when failed
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, inspect
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from climb_sensei.database.models import Base, User, Video, VideoStatus
from climb_sensei.database.config import get_db
from climb_sensei.auth import get_password_hash, create_access_token

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
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def auth_token(test_user):
    """Create JWT token for test user."""
    return create_access_token(data={"sub": str(test_user.id)})


@pytest.fixture
def client(db_session):
    """Create test client with DB override."""
    from app.main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Reset auth rate limiter counters for test isolation
    for middleware in app.middleware_stack.__dict__.get("app", app).__dict__.get(
        "middleware_stack", []
    ):
        if hasattr(middleware, "reset"):
            middleware.reset()

    with TestClient(app) as c:
        # Also try resetting via the app's middleware stack
        _reset_auth_rate_limiter(app)
        yield c
    app.dependency_overrides.clear()


def _reset_auth_rate_limiter(application):
    """Walk middleware stack to find and reset AuthRateLimitMiddleware."""
    current = getattr(application, "middleware_stack", None)
    while current is not None:
        if hasattr(current, "reset"):
            current.reset()
        current = getattr(current, "app", None)


# ========== Auth Rate Limiting ==========


class TestAuthRateLimiting:
    """Tests for authentication rate limiting."""

    def test_login_allows_normal_requests(self, client):
        """Normal login attempts should not be rate-limited."""
        for _ in range(3):
            response = client.post(
                "/api/v1/auth/jwt/login",
                data={"username": "nonexist@test.com", "password": "wrong"},
            )
            assert response.status_code != 429

    def test_login_rate_limited_after_many_attempts(self, client):
        """Should return 429 after exceeding rate limit."""
        for _ in range(6):
            response = client.post(
                "/api/v1/auth/jwt/login",
                data={"username": "attack@test.com", "password": "brute"},
            )
        # The 6th request should be rate limited
        assert response.status_code == 429
        assert "Too many attempts" in response.json()["detail"]

    def test_register_rate_limited_after_many_attempts(self, client):
        """Should rate limit registration spam."""
        for i in range(6):
            response = client.post(
                "/api/v1/auth/register",
                json={
                    "email": f"spam{i}@test.com",
                    "password": "password123",
                },
            )
        assert response.status_code == 429


# ========== Error Message Storage ==========


class TestErrorMessageStorage:
    """Tests for failed analysis error message storage."""

    def test_failed_video_has_error_message(self, db_session, test_user):
        """Video model should store error_message when analysis fails."""
        video = Video(
            user_id=test_user.id,
            filename="test.mp4",
            file_path="/tmp/test.mp4",
            status=VideoStatus.FAILED,
            error_message="MediaPipe failed to initialize",
        )
        db_session.add(video)
        db_session.commit()
        db_session.refresh(video)

        assert video.error_message == "MediaPipe failed to initialize"
        assert video.status == VideoStatus.FAILED

    def test_status_endpoint_returns_error_message(
        self, client, db_session, test_user, auth_token
    ):
        """GET /api/videos/{id}/status should include error_message when failed."""
        video = Video(
            user_id=test_user.id,
            filename="test.mp4",
            file_path="/tmp/test.mp4",
            status=VideoStatus.FAILED,
            error_message="Video too short for analysis",
        )
        db_session.add(video)
        db_session.commit()
        db_session.refresh(video)

        response = client.get(
            f"/api/v1/videos/{video.id}/status",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "failed"
        assert "error_message" in data

    def test_status_endpoint_no_error_on_success(
        self, client, db_session, test_user, auth_token
    ):
        """Successful videos should not have error_message in response."""
        video = Video(
            user_id=test_user.id,
            filename="test.mp4",
            file_path="/tmp/test.mp4",
            status=VideoStatus.COMPLETED,
        )
        db_session.add(video)
        db_session.commit()
        db_session.refresh(video)

        response = client.get(
            f"/api/v1/videos/{video.id}/status",
            headers={"Authorization": f"Bearer {auth_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "error_message" not in data


# ========== Database Indexes ==========


class TestDatabaseIndexes:
    """Tests that key indexes exist on the database models."""

    def test_attempt_date_has_index(self, test_engine):
        """Attempt.date should be indexed for sorting/filtering."""
        inspector = inspect(test_engine)
        indexes = inspector.get_indexes("attempts")
        indexed_columns = {col for idx in indexes for col in idx["column_names"]}
        assert "date" in indexed_columns

    def test_analysis_created_at_has_index(self, test_engine):
        """Analysis.created_at should be indexed for time-range queries."""
        inspector = inspect(test_engine)
        indexes = inspector.get_indexes("analyses")
        indexed_columns = {col for idx in indexes for col in idx["column_names"]}
        assert "created_at" in indexed_columns

    def test_goal_metric_name_has_index(self, test_engine):
        """Goal.metric_name should be indexed for filtering."""
        inspector = inspect(test_engine)
        indexes = inspector.get_indexes("goals")
        indexed_columns = {col for idx in indexes for col in idx["column_names"]}
        assert "metric_name" in indexed_columns
