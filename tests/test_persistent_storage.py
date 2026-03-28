"""Tests for persistent analysis storage (Phase 3).

This module tests authenticated video uploads, database persistence,
and retrieval of videos and analyses.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from climb_sensei.database.models import Base, User, Video, Analysis
from climb_sensei.database.config import get_db
from climb_sensei.auth import get_password_hash


@pytest.fixture
def test_engine():
    """Create a test database engine."""
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
        email="testuser@example.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_user_token(test_user):
    """Create an access token for the test user."""
    from climb_sensei.auth import create_access_token

    return create_access_token(data={"sub": str(test_user.id)})


@pytest.fixture
def client(db_session):
    """Create a FastAPI test client with database override."""
    from main import app

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


class TestVideoStorage:
    """Test video database storage."""

    def test_video_record_created_on_upload(self, client, test_user_token, db_session):
        """Should create a Video record when file is uploaded."""
        # This would require a real video file and MediaPipe processing
        # For now, we'll test the database model directly
        video = Video(
            user_id=1,
            filename="test_climb.mp4",
            file_path="/path/to/test_climb.mp4",
            status="processing",
        )
        db_session.add(video)
        db_session.commit()

        # Verify video was created
        saved_video = (
            db_session.query(Video).filter(Video.filename == "test_climb.mp4").first()
        )
        assert saved_video is not None
        assert saved_video.user_id == 1
        assert saved_video.status == "processing"

    def test_video_metadata_stored(self, db_session):
        """Should store video metadata."""
        video = Video(
            user_id=1,
            filename="test_climb.mp4",
            file_path="/path/to/test_climb.mp4",
            status="completed",
            metadata={
                "fps": 30.0,
                "width": 1920,
                "height": 1080,
                "frame_count": 900,
                "duration": 30.0,
            },
        )
        db_session.add(video)
        db_session.commit()

        saved_video = db_session.query(Video).first()
        assert saved_video.metadata["fps"] == 30.0
        assert saved_video.metadata["width"] == 1920
        assert saved_video.metadata["height"] == 1080

    def test_video_status_updates(self, db_session):
        """Should update video status after processing."""
        video = Video(
            user_id=1,
            filename="test_climb.mp4",
            file_path="/path/to/test_climb.mp4",
            status="processing",
        )
        db_session.add(video)
        db_session.commit()

        # Update status
        video.status = "completed"
        db_session.commit()

        saved_video = db_session.query(Video).first()
        assert saved_video.status == "completed"


class TestAnalysisStorage:
    """Test analysis database storage."""

    def test_analysis_record_created(self, db_session):
        """Should create an Analysis record after processing."""
        # Create video first
        video = Video(
            user_id=1,
            filename="test_climb.mp4",
            file_path="/path/to/test_climb.mp4",
            status="completed",
        )
        db_session.add(video)
        db_session.flush()

        # Create analysis
        analysis = Analysis(
            video_id=video.id,
            summary={
                "total_frames": 900,
                "avg_velocity": 0.15,
                "max_height": 0.85,
            },
            history={
                "com_velocity": [0.1, 0.15, 0.2, 0.18],
                "height": [0.5, 0.6, 0.7, 0.85],
            },
        )
        db_session.add(analysis)
        db_session.commit()

        # Verify analysis was created
        saved_analysis = db_session.query(Analysis).first()
        assert saved_analysis is not None
        assert saved_analysis.video_id == video.id
        assert saved_analysis.summary["total_frames"] == 900

    def test_analysis_stores_quality_reports(self, db_session):
        """Should store quality reports in analysis."""
        video = Video(
            user_id=1,
            filename="test_climb.mp4",
            file_path="/path/to/test_climb.mp4",
        )
        db_session.add(video)
        db_session.flush()

        analysis = Analysis(
            video_id=video.id,
            summary={"total_frames": 900},
            video_quality={
                "is_valid": True,
                "resolution": "1920x1080",
                "fps_quality": "good",
            },
            tracking_quality={
                "is_trackable": True,
                "detection_rate": 0.95,
                "quality_level": "excellent",
            },
        )
        db_session.add(analysis)
        db_session.commit()

        saved_analysis = db_session.query(Analysis).first()
        assert saved_analysis.video_quality is not None
        assert saved_analysis.video_quality["is_valid"] is True
        assert saved_analysis.tracking_quality["detection_rate"] == 0.95

    def test_analysis_linked_to_video(self, db_session):
        """Should link analysis to video through relationship."""
        video = Video(
            user_id=1,
            filename="test_climb.mp4",
            file_path="/path/to/test_climb.mp4",
        )
        db_session.add(video)
        db_session.flush()

        analysis1 = Analysis(
            video_id=video.id,
            summary={"total_frames": 900},
        )
        analysis2 = Analysis(
            video_id=video.id,
            summary={"total_frames": 900},
        )
        db_session.add_all([analysis1, analysis2])
        db_session.commit()

        # Access analyses through video relationship
        saved_video = db_session.query(Video).first()
        assert len(saved_video.analyses) == 2
        assert all(a.video_id == video.id for a in saved_video.analyses)


class TestVideoRetrievalAPI:
    """Test video retrieval API endpoints."""

    def test_list_videos_requires_auth(self, client):
        """Should require authentication to list videos."""
        response = client.get("/api/v1/videos")
        assert response.status_code == 401

    def test_list_videos_returns_user_videos(
        self, client, test_user_token, test_user, db_session
    ):
        """Should return only the current user's videos."""
        # Create videos for test user
        video1 = Video(
            user_id=test_user.id,
            filename="climb1.mp4",
            file_path="/path/to/climb1.mp4",
            status="completed",
        )
        video2 = Video(
            user_id=test_user.id,
            filename="climb2.mp4",
            file_path="/path/to/climb2.mp4",
            status="processing",
        )
        # Create video for another user
        other_user = User(
            email="other@example.com",
            hashed_password=get_password_hash("password123"),
        )
        db_session.add(other_user)
        db_session.flush()

        video3 = Video(
            user_id=other_user.id,
            filename="other_climb.mp4",
            file_path="/path/to/other_climb.mp4",
        )
        db_session.add_all([video1, video2, video3])
        db_session.commit()

        # List videos
        response = client.get(
            "/api/v1/videos", headers={"Authorization": f"Bearer {test_user_token}"}
        )

        assert response.status_code == 200
        videos = response.json()
        assert len(videos) == 2
        assert all(v["filename"] in ["climb1.mp4", "climb2.mp4"] for v in videos)

    def test_get_video_details(self, client, test_user_token, test_user, db_session):
        """Should return detailed video information."""
        video = Video(
            user_id=test_user.id,
            filename="climb1.mp4",
            file_path="/path/to/climb1.mp4",
            status="completed",
            metadata={"fps": 30.0, "width": 1920, "height": 1080},
        )
        db_session.add(video)
        db_session.flush()

        analysis = Analysis(
            video_id=video.id,
            summary={"total_frames": 900},
        )
        db_session.add(analysis)
        db_session.commit()

        # Get video details
        response = client.get(
            f"/api/v1/v1/videos/{video.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["filename"] == "climb1.mp4"
        assert data["status"] == "completed"
        assert len(data["analyses"]) == 1

    def test_get_video_not_found(self, client, test_user_token):
        """Should return 404 for non-existent video."""
        response = client.get(
            "/api/v1/videos/999", headers={"Authorization": f"Bearer {test_user_token}"}
        )
        assert response.status_code == 404

    def test_cannot_access_other_user_video(
        self, client, test_user_token, test_user, db_session
    ):
        """Should not allow access to other user's videos."""
        # Create another user and their video
        other_user = User(
            email="other@example.com",
            hashed_password=get_password_hash("password123"),
        )
        db_session.add(other_user)
        db_session.flush()

        video = Video(
            user_id=other_user.id,
            filename="other_climb.mp4",
            file_path="/path/to/other_climb.mp4",
        )
        db_session.add(video)
        db_session.commit()

        # Try to access other user's video
        response = client.get(
            f"/api/v1/v1/videos/{video.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404


class TestAnalysisRetrievalAPI:
    """Test analysis retrieval API endpoints."""

    def test_list_analyses_requires_auth(self, client):
        """Should require authentication to list analyses."""
        response = client.get("/api/v1/analyses")
        assert response.status_code == 401

    def test_list_analyses_returns_user_analyses(
        self, client, test_user_token, test_user, db_session
    ):
        """Should return only the current user's analyses."""
        # Create video and analyses
        video = Video(
            user_id=test_user.id,
            filename="climb1.mp4",
            file_path="/path/to/climb1.mp4",
        )
        db_session.add(video)
        db_session.flush()

        analysis1 = Analysis(
            video_id=video.id,
            summary={"total_frames": 900},
        )
        analysis2 = Analysis(
            video_id=video.id,
            summary={"total_frames": 1200},
        )
        db_session.add_all([analysis1, analysis2])
        db_session.commit()

        # List analyses
        response = client.get(
            "/api/v1/analyses", headers={"Authorization": f"Bearer {test_user_token}"}
        )

        assert response.status_code == 200
        analyses = response.json()
        assert len(analyses) == 2
        assert all(a["video_filename"] == "climb1.mp4" for a in analyses)

    def test_get_analysis_details(self, client, test_user_token, test_user, db_session):
        """Should return detailed analysis information."""
        video = Video(
            user_id=test_user.id,
            filename="climb1.mp4",
            file_path="/path/to/climb1.mp4",
        )
        db_session.add(video)
        db_session.flush()

        analysis = Analysis(
            video_id=video.id,
            summary={
                "total_frames": 900,
                "avg_velocity": 0.15,
            },
            history={
                "com_velocity": [0.1, 0.15, 0.2],
            },
            video_quality={"is_valid": True},
            tracking_quality={"is_trackable": True},
        )
        db_session.add(analysis)
        db_session.commit()

        # Get analysis details
        response = client.get(
            f"/api/v1/v1/analyses/{analysis.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["video_filename"] == "climb1.mp4"
        assert data["summary"]["total_frames"] == 900
        assert data["history"]["com_velocity"] == [0.1, 0.15, 0.2]
        assert data["video_quality"]["is_valid"] is True

    def test_cannot_access_other_user_analysis(
        self, client, test_user_token, test_user, db_session
    ):
        """Should not allow access to other user's analyses."""
        # Create another user and their analysis
        other_user = User(
            email="other@example.com",
            hashed_password=get_password_hash("password123"),
        )
        db_session.add(other_user)
        db_session.flush()

        video = Video(
            user_id=other_user.id,
            filename="other_climb.mp4",
            file_path="/path/to/other_climb.mp4",
        )
        db_session.add(video)
        db_session.flush()

        analysis = Analysis(
            video_id=video.id,
            summary={"total_frames": 900},
        )
        db_session.add(analysis)
        db_session.commit()

        # Try to access other user's analysis
        response = client.get(
            f"/api/v1/v1/analyses/{analysis.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404


class TestCascadeDeletes:
    """Test cascade delete behavior."""

    def test_deleting_video_deletes_analyses(self, db_session):
        """Should delete analyses when video is deleted."""
        video = Video(
            user_id=1,
            filename="climb1.mp4",
            file_path="/path/to/climb1.mp4",
        )
        db_session.add(video)
        db_session.flush()

        analysis1 = Analysis(video_id=video.id, summary={})
        analysis2 = Analysis(video_id=video.id, summary={})
        db_session.add_all([analysis1, analysis2])
        db_session.commit()

        # Delete video
        db_session.delete(video)
        db_session.commit()

        # Analyses should be deleted
        remaining_analyses = db_session.query(Analysis).all()
        assert len(remaining_analyses) == 0

    def test_deleting_user_deletes_videos_and_analyses(self, db_session):
        """Should delete videos and analyses when user is deleted."""
        user = User(
            email="testuser@example.com",
            hashed_password=get_password_hash("password123"),
        )
        db_session.add(user)
        db_session.flush()

        video = Video(
            user_id=user.id,
            filename="climb1.mp4",
            file_path="/path/to/climb1.mp4",
        )
        db_session.add(video)
        db_session.flush()

        analysis = Analysis(video_id=video.id, summary={})
        db_session.add(analysis)
        db_session.commit()

        # Delete user
        db_session.delete(user)
        db_session.commit()

        # Videos and analyses should be deleted
        remaining_videos = db_session.query(Video).all()
        remaining_analyses = db_session.query(Analysis).all()
        assert len(remaining_videos) == 0
        assert len(remaining_analyses) == 0
