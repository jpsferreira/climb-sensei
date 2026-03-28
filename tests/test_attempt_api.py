"""Tests for Attempt API endpoints.

Covers:
- GET /api/routes/{route_id}/attempts            — list attempts
- GET /api/routes/{route_id}/attempts/{id}       — attempt detail with deltas
- PATCH /api/routes/{route_id}/attempts/{id}     — update notes
- DELETE /api/routes/{route_id}/attempts/{id}    — delete attempt only
- GET /api/routes/{route_id}/progress/{metric}   — metric trend
"""

import pytest
from datetime import datetime, timedelta, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from climb_sensei.database.models import Base, User, Route, Attempt, Analysis, Video
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


@pytest.fixture
def sample_analysis(db_session, sample_video):
    """Create a sample analysis with denormalized metrics."""
    analysis = Analysis(
        video_id=sample_video.id,
        run_metrics=True,
        run_video=False,
        run_quality=False,
        avg_velocity=1.5,
        max_velocity=3.0,
        max_height=0.8,
        total_vertical_progress=0.6,
        avg_sway=0.05,
        avg_movement_economy=0.9,
        lock_off_count=2,
        rest_count=1,
        fatigue_score=0.3,
        summary={"analysis_id": "abc"},
        history={},
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)
    return analysis


@pytest.fixture
def sample_attempt(db_session, sample_route, sample_video, sample_analysis):
    """Create a sample attempt."""
    attempt = Attempt(
        route_id=sample_route.id,
        video_id=sample_video.id,
        analysis_id=sample_analysis.id,
        date=datetime.now(timezone.utc),
        notes="First try",
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)
    return attempt


# ========== List Attempts Tests ==========


class TestListAttempts:
    def test_list_attempts_empty(self, client, sample_route):
        """Should return empty list when route has no attempts."""
        response = client.get(f"/api/v1/v1/routes/{sample_route.id}/attempts")
        assert response.status_code == 200
        assert response.json() == []

    def test_list_attempts_returns_attempts(self, client, sample_attempt, sample_route):
        """Should return attempts for the route."""
        response = client.get(f"/api/v1/v1/routes/{sample_route.id}/attempts")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 1
        assert data[0]["id"] == sample_attempt.id
        assert data[0]["route_id"] == sample_route.id

    def test_list_attempts_includes_inline_metrics(
        self, client, sample_attempt, sample_route
    ):
        """Should include avg_velocity, avg_sway, avg_movement_economy from analysis."""
        response = client.get(f"/api/v1/v1/routes/{sample_route.id}/attempts")
        data = response.json()[0]
        assert data["avg_velocity"] == pytest.approx(1.5)
        assert data["avg_sway"] == pytest.approx(0.05)
        assert data["avg_movement_economy"] == pytest.approx(0.9)

    def test_list_attempts_most_recent_first(
        self, client, db_session, sample_route, sample_video, sample_analysis
    ):
        """Should return attempts sorted by date descending."""
        now = datetime.now(timezone.utc)
        old_attempt = Attempt(
            route_id=sample_route.id,
            video_id=sample_video.id,
            date=now - timedelta(days=5),
        )
        new_attempt = Attempt(
            route_id=sample_route.id,
            video_id=sample_video.id,
            date=now - timedelta(days=1),
        )
        db_session.add_all([old_attempt, new_attempt])
        db_session.commit()

        response = client.get(f"/api/v1/v1/routes/{sample_route.id}/attempts")
        assert response.status_code == 200
        dates = [a["date"] for a in response.json()]
        assert dates == sorted(dates, reverse=True)

    def test_list_attempts_route_not_found(self, client):
        """Should return 404 for a non-existent route."""
        response = client.get("/api/v1/routes/99999/attempts")
        assert response.status_code == 404

    def test_list_attempts_other_user_route(
        self, client, db_session, other_user, sample_video
    ):
        """Should return 404 when the route belongs to another user."""
        other_route = Route(
            user_id=other_user.id,
            name="Other Route",
            grade="V3",
            grade_system="hueco",
            type="boulder",
        )
        db_session.add(other_route)
        db_session.commit()

        response = client.get(f"/api/v1/v1/routes/{other_route.id}/attempts")
        assert response.status_code == 404


# ========== Get Attempt Detail Tests ==========


class TestGetAttemptDetail:
    def test_get_attempt_detail_success(
        self, client, sample_route, sample_attempt, sample_analysis
    ):
        """Should return full attempt detail."""
        response = client.get(
            f"/api/v1/v1/routes/{sample_route.id}/attempts/{sample_attempt.id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_attempt.id
        assert data["analysis_id"] == sample_analysis.id
        assert data["notes"] == "First try"

    def test_get_attempt_detail_includes_analysis_data(
        self, client, sample_route, sample_attempt, sample_analysis
    ):
        """Should include summary and quality data from analysis."""
        response = client.get(
            f"/api/v1/v1/routes/{sample_route.id}/attempts/{sample_attempt.id}"
        )
        data = response.json()
        assert data["summary"] == {"analysis_id": "abc"}
        assert data["history"] == {}

    def test_get_attempt_detail_no_previous(self, client, sample_route, sample_attempt):
        """Should return null deltas and prev_attempt_id when no prior attempt."""
        response = client.get(
            f"/api/v1/v1/routes/{sample_route.id}/attempts/{sample_attempt.id}"
        )
        data = response.json()
        assert data["prev_attempt_id"] is None
        assert data["deltas"] is None

    def test_get_attempt_detail_with_deltas(
        self, client, db_session, sample_route, sample_video
    ):
        """Should compute deltas against previous attempt's analysis."""
        # First attempt (older)
        analysis1 = Analysis(
            video_id=sample_video.id,
            run_metrics=True,
            run_video=False,
            run_quality=False,
            avg_velocity=1.0,
            avg_sway=0.1,
            avg_movement_economy=0.7,
            max_velocity=2.0,
            max_height=0.5,
            total_vertical_progress=0.4,
            lock_off_count=1,
            rest_count=0,
            fatigue_score=0.2,
            summary={},
            history={},
        )
        db_session.add(analysis1)
        db_session.flush()

        attempt1 = Attempt(
            route_id=sample_route.id,
            video_id=sample_video.id,
            analysis_id=analysis1.id,
            date=datetime.now(timezone.utc) - timedelta(days=3),
        )
        db_session.add(attempt1)
        db_session.flush()

        # Second attempt (newer)
        analysis2 = Analysis(
            video_id=sample_video.id,
            run_metrics=True,
            run_video=False,
            run_quality=False,
            avg_velocity=1.5,
            avg_sway=0.05,
            avg_movement_economy=0.9,
            max_velocity=3.0,
            max_height=0.8,
            total_vertical_progress=0.6,
            lock_off_count=2,
            rest_count=1,
            fatigue_score=0.3,
            summary={},
            history={},
        )
        db_session.add(analysis2)
        db_session.flush()

        attempt2 = Attempt(
            route_id=sample_route.id,
            video_id=sample_video.id,
            analysis_id=analysis2.id,
            date=datetime.now(timezone.utc) - timedelta(days=1),
        )
        db_session.add(attempt2)
        db_session.commit()

        response = client.get(
            f"/api/v1/v1/routes/{sample_route.id}/attempts/{attempt2.id}"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["prev_attempt_id"] == attempt1.id
        deltas = data["deltas"]
        assert deltas is not None
        assert deltas["avg_velocity"] == pytest.approx(0.5)
        assert deltas["avg_sway"] == pytest.approx(-0.05)

    def test_get_attempt_not_found(self, client, sample_route):
        """Should return 404 for non-existent attempt."""
        response = client.get(f"/api/v1/v1/routes/{sample_route.id}/attempts/99999")
        assert response.status_code == 404

    def test_get_attempt_route_not_found(self, client, sample_attempt):
        """Should return 404 when route doesn't exist."""
        response = client.get(f"/api/v1/v1/routes/99999/attempts/{sample_attempt.id}")
        assert response.status_code == 404


# ========== Update Attempt Tests ==========


class TestUpdateAttempt:
    def test_update_attempt_notes(self, client, sample_route, sample_attempt):
        """Should update attempt notes."""
        response = client.patch(
            f"/api/v1/v1/routes/{sample_route.id}/attempts/{sample_attempt.id}",
            json={"notes": "Updated notes after reflection"},
        )
        assert response.status_code == 200
        assert response.json()["notes"] == "Updated notes after reflection"

    def test_update_attempt_ignores_unknown_fields(
        self, client, sample_route, sample_attempt
    ):
        """Should apply only notes; other fields silently ignored."""
        original_video_id = sample_attempt.video_id
        response = client.patch(
            f"/api/v1/v1/routes/{sample_route.id}/attempts/{sample_attempt.id}",
            json={"notes": "New note", "video_id": 9999},
        )
        assert response.status_code == 200
        # video_id should remain unchanged
        assert response.json()["video_id"] == original_video_id

    def test_update_attempt_not_found(self, client, sample_route):
        """Should return 404 for non-existent attempt."""
        response = client.patch(
            f"/api/v1/v1/routes/{sample_route.id}/attempts/99999",
            json={"notes": "ghost"},
        )
        assert response.status_code == 404

    def test_update_attempt_route_not_found(self, client, sample_attempt):
        """Should return 404 when route doesn't exist."""
        response = client.patch(
            f"/api/v1/v1/routes/99999/attempts/{sample_attempt.id}",
            json={"notes": "ghost"},
        )
        assert response.status_code == 404


# ========== Delete Attempt Tests ==========


class TestDeleteAttempt:
    def test_delete_attempt_success(
        self,
        client,
        db_session,
        sample_route,
        sample_attempt,
        sample_analysis,
        sample_video,
    ):
        """Should delete attempt and return 204; video and analysis preserved."""
        from climb_sensei.database.models import Attempt as AttemptModel

        response = client.delete(
            f"/api/v1/v1/routes/{sample_route.id}/attempts/{sample_attempt.id}"
        )
        assert response.status_code == 204

        # Attempt is gone
        deleted = db_session.get(AttemptModel, sample_attempt.id)
        assert deleted is None

        # Video and Analysis still exist
        from climb_sensei.database.models import (
            Video as VideoModel,
            Analysis as AnalysisModel,
        )

        assert db_session.get(VideoModel, sample_video.id) is not None
        assert db_session.get(AnalysisModel, sample_analysis.id) is not None

    def test_delete_attempt_not_found(self, client, sample_route):
        """Should return 404 for non-existent attempt."""
        response = client.delete(f"/api/v1/v1/routes/{sample_route.id}/attempts/99999")
        assert response.status_code == 404

    def test_delete_attempt_route_not_found(self, client, sample_attempt):
        """Should return 404 when route doesn't exist."""
        response = client.delete(
            f"/api/v1/v1/routes/99999/attempts/{sample_attempt.id}"
        )
        assert response.status_code == 404


# ========== Metric Trend Tests ==========


class TestMetricTrend:
    def test_get_metric_trend_empty(self, client, sample_route):
        """Should return empty data when no attempts have analyses."""
        response = client.get(
            f"/api/v1/v1/routes/{sample_route.id}/progress/avg_velocity"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "avg_velocity"
        assert data["data"] == []
        assert data["count"] == 0

    def test_get_metric_trend_with_data(
        self, client, sample_route, sample_attempt, sample_analysis
    ):
        """Should return data points for each attempt with the metric."""
        response = client.get(
            f"/api/v1/v1/routes/{sample_route.id}/progress/avg_velocity"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert data["data"][0]["value"] == pytest.approx(1.5)
        assert data["data"][0]["attempt_id"] == sample_attempt.id

    def test_get_metric_trend_ordered_by_date_asc(
        self, client, db_session, sample_route, sample_video
    ):
        """Should return data points ordered by attempt date ascending."""
        now = datetime.now(timezone.utc)
        for i, days_ago in enumerate([5, 3, 1]):
            analysis = Analysis(
                video_id=sample_video.id,
                run_metrics=True,
                run_video=False,
                run_quality=False,
                avg_velocity=float(i + 1),
                summary={},
                history={},
            )
            db_session.add(analysis)
            db_session.flush()
            attempt = Attempt(
                route_id=sample_route.id,
                video_id=sample_video.id,
                analysis_id=analysis.id,
                date=now - timedelta(days=days_ago),
            )
            db_session.add(attempt)
        db_session.commit()

        response = client.get(
            f"/api/v1/v1/routes/{sample_route.id}/progress/avg_velocity"
        )
        assert response.status_code == 200
        values = [p["value"] for p in response.json()["data"]]
        assert values == sorted(values)  # ascending

    def test_get_metric_trend_invalid_metric(self, client, sample_route):
        """Should return 400 for a metric not in ALLOWED_METRICS."""
        response = client.get(
            f"/api/v1/v1/routes/{sample_route.id}/progress/hacked_field"
        )
        assert response.status_code == 400
        assert "Invalid metric" in response.json()["detail"]

    def test_get_metric_trend_all_allowed_metrics(self, client, sample_route):
        """All metrics in ALLOWED_METRICS should return 200."""
        from climb_sensei.progress.attempt_routes import ALLOWED_METRICS

        for metric in ALLOWED_METRICS:
            response = client.get(
                f"/api/v1/v1/routes/{sample_route.id}/progress/{metric}"
            )
            assert response.status_code == 200, f"Failed for metric: {metric}"

    def test_get_metric_trend_route_not_found(self, client):
        """Should return 404 for non-existent route."""
        response = client.get("/api/v1/routes/99999/progress/avg_velocity")
        assert response.status_code == 404
