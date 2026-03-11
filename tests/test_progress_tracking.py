"""Comprehensive tests for Phase 4: Progress Tracking.

Tests cover:
- Progress metric tracking
- Goal management
- Analysis comparisons
- Climbing session management
- Auto-recording of metrics from analyses
"""

import pytest
from datetime import datetime, timedelta
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app
from climb_sensei.database.config import get_db
from climb_sensei.database.models import (
    Base,
    User,
    Video,
    Analysis,
    ProgressMetric,
    Goal,
    ClimbSession,
)
from climb_sensei.auth import get_password_hash, create_access_token


# ========== Test Fixtures ==========


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
    """Create a test user."""
    user = User(
        email="testuser@example.com",
        password_hash=get_password_hash("testpass123"),
        full_name="Test User",
        is_active=True,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_user_token(test_user):
    """Generate JWT token for test user."""
    return create_access_token(data={"sub": test_user.email})


@pytest.fixture
def client(test_engine, db_session):
    """Create test client with overridden database dependency."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


@pytest.fixture
def sample_video(db_session, test_user):
    """Create a sample video record."""
    video = Video(
        user_id=test_user.id,
        filename="test_video.mp4",
        file_path="/uploads/test_video.mp4",
        duration_seconds=30.0,
        fps=30.0,
        width=1920,
        height=1080,
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    return video


@pytest.fixture
def sample_analysis(db_session, sample_video):
    """Create a sample analysis record with metrics."""
    analysis = Analysis(
        video_id=sample_video.id,
        summary={"avg_velocity": 0.15, "lock_off_count": 5},
        history={"velocity": [0.1, 0.15, 0.2]},
        total_frames=100,
        avg_velocity=0.15,
        max_velocity=0.25,
        max_height=2.5,
        total_vertical_progress=2.0,
        avg_sway=0.03,
        avg_movement_economy=0.75,
        lock_off_count=5,
        rest_count=2,
        fatigue_score=0.4,
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)
    return analysis


# ========== Progress Metrics Tests ==========


class TestProgressMetrics:
    """Test progress metric tracking endpoints."""

    def test_get_metric_progress_empty(self, client, test_user_token):
        """Should return empty data for metric with no history."""
        response = client.get(
            "/api/progress/avg_velocity",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "avg_velocity"
        assert data["count"] == 0
        assert data["data"] == []

    def test_get_metric_progress_with_data(
        self, client, test_user_token, db_session, test_user, sample_analysis
    ):
        """Should return progress data for tracked metric."""
        # Create progress metrics
        for i in range(5):
            metric = ProgressMetric(
                user_id=test_user.id,
                analysis_id=sample_analysis.id,
                metric_name="avg_velocity",
                value=0.1 + i * 0.02,
            )
            db_session.add(metric)
        db_session.commit()

        response = client.get(
            "/api/progress/avg_velocity",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["metric"] == "avg_velocity"
        assert data["count"] == 5
        assert len(data["data"]) == 5
        assert data["min_value"] == pytest.approx(0.1)
        assert data["max_value"] == pytest.approx(0.18)

    def test_get_metric_progress_with_days_filter(
        self, client, test_user_token, db_session, test_user, sample_analysis
    ):
        """Should filter metrics by days parameter."""
        # Create old and recent metrics
        old_date = datetime.utcnow() - timedelta(days=45)
        recent_date = datetime.utcnow() - timedelta(days=5)

        old_metric = ProgressMetric(
            user_id=test_user.id,
            analysis_id=sample_analysis.id,
            metric_name="lock_off_count",
            value=3.0,
            recorded_at=old_date,
        )
        recent_metric = ProgressMetric(
            user_id=test_user.id,
            analysis_id=sample_analysis.id,
            metric_name="lock_off_count",
            value=5.0,
            recorded_at=recent_date,
        )
        db_session.add_all([old_metric, recent_metric])
        db_session.commit()

        # Request last 30 days
        response = client.get(
            "/api/progress/lock_off_count?days=30",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1  # Only recent one

    def test_get_all_metrics_progress(
        self, client, test_user_token, db_session, test_user, sample_analysis
    ):
        """Should return all tracked metrics."""
        metrics_data = [
            ("avg_velocity", 0.15),
            ("avg_velocity", 0.18),
            ("lock_off_count", 5.0),
            ("lock_off_count", 7.0),
            ("fatigue_score", 0.3),
        ]

        for metric_name, value in metrics_data:
            metric = ProgressMetric(
                user_id=test_user.id,
                analysis_id=sample_analysis.id,
                metric_name=metric_name,
                value=value,
            )
            db_session.add(metric)
        db_session.commit()

        response = client.get(
            "/api/progress", headers={"Authorization": f"Bearer {test_user_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "avg_velocity" in data
        assert "lock_off_count" in data
        assert "fatigue_score" in data
        assert data["avg_velocity"]["count"] == 2
        assert data["lock_off_count"]["count"] == 2
        assert data["fatigue_score"]["count"] == 1

    def test_progress_metrics_user_isolation(
        self, client, test_user_token, db_session, test_user, sample_analysis
    ):
        """Should only return metrics for current user."""
        # Create another user with metrics
        other_user = User(
            email="other@example.com",
            password_hash=get_password_hash("pass123"),
        )
        db_session.add(other_user)
        db_session.commit()

        other_metric = ProgressMetric(
            user_id=other_user.id,
            analysis_id=sample_analysis.id,
            metric_name="avg_velocity",
            value=0.99,
        )
        user_metric = ProgressMetric(
            user_id=test_user.id,
            analysis_id=sample_analysis.id,
            metric_name="avg_velocity",
            value=0.15,
        )
        db_session.add_all([other_metric, user_metric])
        db_session.commit()

        response = client.get(
            "/api/progress/avg_velocity",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        data = response.json()
        assert data["count"] == 1
        assert data["data"][0]["value"] == 0.15


# ========== Analysis Comparison Tests ==========


class TestAnalysisComparison:
    """Test analysis comparison endpoints."""

    def test_compare_analyses(self, client, test_user_token, db_session, sample_video):
        """Should compare multiple analyses."""
        # Create multiple analyses
        analysis1 = Analysis(
            video_id=sample_video.id,
            avg_velocity=0.15,
            max_velocity=0.25,
            lock_off_count=5,
        )
        analysis2 = Analysis(
            video_id=sample_video.id,
            avg_velocity=0.18,
            max_velocity=0.30,
            lock_off_count=7,
        )
        db_session.add_all([analysis1, analysis2])
        db_session.commit()

        response = client.post(
            "/api/compare",
            json={"analysis_ids": [analysis1.id, analysis2.id]},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 2
        assert len(data["comparisons"]) == 2
        assert data["comparisons"][0]["id"] == analysis1.id
        assert data["comparisons"][1]["id"] == analysis2.id

    def test_compare_with_session_names(
        self, client, test_user_token, db_session, sample_video, test_user
    ):
        """Should include session names in comparisons."""
        # Create session
        session = ClimbSession(
            user_id=test_user.id, name="Morning Session", date=datetime.utcnow()
        )
        db_session.add(session)
        db_session.commit()

        # Create 2 analyses (minimum for comparison)
        analysis1 = Analysis(
            video_id=sample_video.id, session_id=session.id, avg_velocity=0.15
        )
        analysis2 = Analysis(
            video_id=sample_video.id, session_id=session.id, avg_velocity=0.17
        )
        db_session.add_all([analysis1, analysis2])
        db_session.commit()

        response = client.post(
            "/api/compare",
            json={"analysis_ids": [analysis1.id, analysis2.id]},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        data = response.json()
        assert data["comparisons"][0]["session_id"] == session.id
        assert data["comparisons"][0]["session_name"] == "Morning Session"

    def test_compare_requires_min_analyses(
        self, client, test_user_token, sample_analysis
    ):
        """Should require at least 2 analyses to compare."""
        response = client.post(
            "/api/compare",
            json={"analysis_ids": [sample_analysis.id]},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 422  # Validation error

    def test_compare_non_existent_analysis(
        self, client, test_user_token, sample_analysis
    ):
        """Should return 404 if analysis not found."""
        response = client.post(
            "/api/compare",
            json={"analysis_ids": [sample_analysis.id, 99999]},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404

    def test_compare_user_isolation(
        self, client, test_user_token, db_session, test_user
    ):
        """Should not allow comparing other users' analyses."""
        # Create another user
        other_user = User(
            email="other@example.com",
            password_hash=get_password_hash("pass123"),
        )
        db_session.add(other_user)
        db_session.commit()

        # Create video and analysis for other user
        other_video = Video(
            user_id=other_user.id, filename="other.mp4", file_path="/uploads/other.mp4"
        )
        db_session.add(other_video)
        db_session.commit()

        other_analysis = Analysis(video_id=other_video.id, avg_velocity=0.20)
        db_session.add(other_analysis)
        db_session.commit()

        # Try to access with test_user's token
        response = client.post(
            "/api/compare",
            json={"analysis_ids": [other_analysis.id, other_analysis.id]},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404


# ========== Goal Management Tests ==========


class TestGoalManagement:
    """Test goal setting and tracking endpoints."""

    def test_create_goal(self, client, test_user_token):
        """Should create a new goal."""
        deadline = (datetime.utcnow() + timedelta(days=30)).isoformat()
        response = client.post(
            "/api/goals",
            json={
                "metric_name": "avg_velocity",
                "target_value": 0.20,
                "deadline": deadline,
                "notes": "Improve speed",
            },
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["metric_name"] == "avg_velocity"
        assert data["target_value"] == 0.20
        assert data["achieved"] is False

    def test_create_goal_with_current_value(
        self, client, test_user_token, db_session, test_user, sample_analysis
    ):
        """Should set current_value from latest metric."""
        # Create progress metric
        metric = ProgressMetric(
            user_id=test_user.id,
            analysis_id=sample_analysis.id,
            metric_name="lock_off_count",
            value=5.0,
        )
        db_session.add(metric)
        db_session.commit()

        response = client.post(
            "/api/goals",
            json={"metric_name": "lock_off_count", "target_value": 10.0},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        data = response.json()
        assert data["current_value"] == 5.0
        assert data["progress_percentage"] == 50.0

    def test_list_goals(self, client, test_user_token, db_session, test_user):
        """Should list all user's goals."""
        goal1 = Goal(
            user_id=test_user.id, metric_name="avg_velocity", target_value=0.20
        )
        goal2 = Goal(
            user_id=test_user.id,
            metric_name="lock_off_count",
            target_value=10.0,
            achieved=True,
        )
        db_session.add_all([goal1, goal2])
        db_session.commit()

        response = client.get(
            "/api/goals", headers={"Authorization": f"Bearer {test_user_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_list_active_goals_only(
        self, client, test_user_token, db_session, test_user
    ):
        """Should filter to show only active goals."""
        goal1 = Goal(
            user_id=test_user.id,
            metric_name="avg_velocity",
            target_value=0.20,
            achieved=False,
        )
        goal2 = Goal(
            user_id=test_user.id,
            metric_name="lock_off_count",
            target_value=10.0,
            achieved=True,
        )
        db_session.add_all([goal1, goal2])
        db_session.commit()

        response = client.get(
            "/api/goals?active_only=true",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        data = response.json()
        assert len(data) == 1
        assert data[0]["achieved"] is False

    def test_get_goal_by_id(self, client, test_user_token, db_session, test_user):
        """Should get specific goal by ID."""
        goal = Goal(
            user_id=test_user.id,
            metric_name="avg_velocity",
            target_value=0.20,
            current_value=0.15,
        )
        db_session.add(goal)
        db_session.commit()

        response = client.get(
            f"/api/goals/{goal.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == goal.id
        assert data["progress_percentage"] == pytest.approx(75.0)

    def test_update_goal(self, client, test_user_token, db_session, test_user):
        """Should update goal fields."""
        goal = Goal(user_id=test_user.id, metric_name="avg_velocity", target_value=0.20)
        db_session.add(goal)
        db_session.commit()

        response = client.patch(
            f"/api/goals/{goal.id}",
            json={"target_value": 0.25, "notes": "Increased target"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["target_value"] == 0.25
        assert data["notes"] == "Increased target"

    def test_mark_goal_achieved(self, client, test_user_token, db_session, test_user):
        """Should auto-set achieved_at when marking goal as achieved."""
        goal = Goal(user_id=test_user.id, metric_name="avg_velocity", target_value=0.20)
        db_session.add(goal)
        db_session.commit()

        response = client.patch(
            f"/api/goals/{goal.id}",
            json={"achieved": True},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        data = response.json()
        assert data["achieved"] is True
        assert data["achieved_at"] is not None

    def test_delete_goal(self, client, test_user_token, db_session, test_user):
        """Should delete a goal."""
        goal = Goal(user_id=test_user.id, metric_name="avg_velocity", target_value=0.20)
        db_session.add(goal)
        db_session.commit()
        goal_id = goal.id

        response = client.delete(
            f"/api/goals/{goal_id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 204

        # Verify deleted
        check_response = client.get(
            f"/api/goals/{goal_id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert check_response.status_code == 404

    def test_goal_user_isolation(self, client, test_user_token, db_session, test_user):
        """Should not allow accessing other users' goals."""
        other_user = User(
            email="other@example.com",
            password_hash=get_password_hash("pass123"),
        )
        db_session.add(other_user)
        db_session.commit()

        other_goal = Goal(
            user_id=other_user.id, metric_name="avg_velocity", target_value=0.30
        )
        db_session.add(other_goal)
        db_session.commit()

        response = client.get(
            f"/api/goals/{other_goal.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404


# ========== ClimbSession Management Tests ==========


class TestSessionManagement:
    """Test climbing session CRUD endpoints."""

    def test_create_session(self, client, test_user_token):
        """Should create a new climbing session."""
        session_date = datetime.utcnow().isoformat()
        response = client.post(
            "/api/sessions",
            json={
                "name": "Morning Bouldering",
                "date": session_date,
                "location": "Local Gym",
                "notes": "Focused on overhangs",
            },
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Morning Bouldering"
        assert data["location"] == "Local Gym"
        assert data["total_videos"] == 0

    def test_list_sessions(self, client, test_user_token, db_session, test_user):
        """Should list all sessions for user."""
        session1 = ClimbSession(
            user_id=test_user.id, name="Session 1", date=datetime.utcnow()
        )
        session2 = ClimbSession(
            user_id=test_user.id,
            name="Session 2",
            date=datetime.utcnow() - timedelta(days=1),
        )
        db_session.add_all([session1, session2])
        db_session.commit()

        response = client.get(
            "/api/sessions", headers={"Authorization": f"Bearer {test_user_token}"}
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2

    def test_list_sessions_pagination(
        self, client, test_user_token, db_session, test_user
    ):
        """Should support pagination."""
        for i in range(5):
            session = ClimbSession(
                user_id=test_user.id,
                name=f"Session {i}",
                date=datetime.utcnow() - timedelta(days=i),
            )
            db_session.add(session)
        db_session.commit()

        response = client.get(
            "/api/sessions?skip=2&limit=2",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        data = response.json()
        assert len(data) == 2

    def test_get_session_by_id(self, client, test_user_token, db_session, test_user):
        """Should get specific session by ID."""
        session = ClimbSession(
            user_id=test_user.id,
            name="Test Session",
            date=datetime.utcnow(),
            location="Gym A",
        )
        db_session.add(session)
        db_session.commit()

        response = client.get(
            f"/api/sessions/{session.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == session.id
        assert data["name"] == "Test Session"

    def test_update_session(self, client, test_user_token, db_session, test_user):
        """Should update session fields."""
        session = ClimbSession(
            user_id=test_user.id, name="Original Name", date=datetime.utcnow()
        )
        db_session.add(session)
        db_session.commit()

        response = client.patch(
            f"/api/sessions/{session.id}",
            json={"name": "Updated Name", "notes": "Great session!"},
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Updated Name"
        assert data["notes"] == "Great session!"

    def test_delete_session(self, client, test_user_token, db_session, test_user):
        """Should delete session without deleting analyses."""
        session = ClimbSession(
            user_id=test_user.id, name="To Delete", date=datetime.utcnow()
        )
        db_session.add(session)
        db_session.commit()

        # Create analysis linked to session
        video = Video(
            user_id=test_user.id, filename="test.mp4", file_path="/uploads/test.mp4"
        )
        db_session.add(video)
        db_session.commit()

        analysis = Analysis(video_id=video.id, session_id=session.id)
        db_session.add(analysis)
        db_session.commit()
        session_id = session.id
        analysis_id = analysis.id

        # Delete session
        response = client.delete(
            f"/api/sessions/{session_id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 204

        # Verify session deleted
        session_check = db_session.query(ClimbSession).filter_by(id=session_id).first()
        assert session_check is None

        # Verify analysis still exists but unlinked
        analysis_check = db_session.query(Analysis).filter_by(id=analysis_id).first()
        assert analysis_check is not None
        assert analysis_check.session_id is None

    def test_get_session_analyses(
        self, client, test_user_token, db_session, test_user, sample_video
    ):
        """Should get all analysis IDs for a session."""
        session = ClimbSession(
            user_id=test_user.id, name="Test Session", date=datetime.utcnow()
        )
        db_session.add(session)
        db_session.commit()

        # Create analyses linked to session
        analysis1 = Analysis(video_id=sample_video.id, session_id=session.id)
        analysis2 = Analysis(video_id=sample_video.id, session_id=session.id)
        db_session.add_all([analysis1, analysis2])
        db_session.commit()

        response = client.get(
            f"/api/sessions/{session.id}/analyses",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert analysis1.id in data
        assert analysis2.id in data

    def test_session_user_isolation(
        self, client, test_user_token, db_session, test_user
    ):
        """Should not allow accessing other users' sessions."""
        other_user = User(
            email="other@example.com",
            password_hash=get_password_hash("pass123"),
        )
        db_session.add(other_user)
        db_session.commit()

        other_session = ClimbSession(
            user_id=other_user.id, name="Other Session", date=datetime.utcnow()
        )
        db_session.add(other_session)
        db_session.commit()

        response = client.get(
            f"/api/sessions/{other_session.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404


# ========== Authentication Tests ==========


class TestProgressAuthentication:
    """Test that all progress endpoints require authentication."""

    def test_progress_endpoint_requires_auth(self, client):
        """Should return 401 without authentication."""
        response = client.get("/api/progress/avg_velocity")
        assert response.status_code == 401

    def test_compare_endpoint_requires_auth(self, client):
        """Should return 401 without authentication."""
        response = client.post("/api/compare", json={"analysis_ids": [1, 2]})
        assert response.status_code == 401

    def test_goals_endpoint_requires_auth(self, client):
        """Should return 401 without authentication."""
        response = client.get("/api/goals")
        assert response.status_code == 401

    def test_sessions_endpoint_requires_auth(self, client):
        """Should return 401 without authentication."""
        response = client.get("/api/sessions")
        assert response.status_code == 401
