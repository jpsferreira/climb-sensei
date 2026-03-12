"""Tests for database models and operations.

This module tests the SQLAlchemy models and database operations
for the ClimbSensei platform.
"""

import sys
from pathlib import Path
from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climb_sensei.database.models import Base, User, Video, Analysis, ClimbSession


@pytest.fixture
def test_engine():
    """Create a test database engine using in-memory SQLite."""
    engine = create_engine("sqlite:///:memory:")
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


class TestUserModel:
    """Tests for the User model."""

    def test_create_user(self, db_session):
        """Should create a user with basic fields."""
        user = User(
            email="test@example.com",
            hashed_password="hashed_password_here",
            full_name="Test User",
        )
        db_session.add(user)
        db_session.commit()

        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.full_name == "Test User"
        assert user.is_active is True
        assert user.is_superuser is False
        assert user.created_at is not None

    def test_user_email_unique(self, db_session):
        """Should enforce unique email constraint."""
        user1 = User(email="test@example.com", hashed_password="hash1")
        user2 = User(email="test@example.com", hashed_password="hash2")

        db_session.add(user1)
        db_session.commit()

        db_session.add(user2)
        with pytest.raises(Exception):  # IntegrityError
            db_session.commit()

    def test_user_relationships(self, db_session):
        """Should have videos and sessions relationships."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        # Test videos relationship
        video = Video(
            user_id=user.id,
            filename="test.mp4",
            file_path="/path/to/test.mp4",
        )
        db_session.add(video)
        db_session.commit()

        assert len(user.videos) == 1
        assert user.videos[0].filename == "test.mp4"

    def test_user_repr(self, db_session):
        """Should have a readable repr."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        assert "test@example.com" in repr(user)


class TestVideoModel:
    """Tests for the Video model."""

    def test_create_video(self, db_session):
        """Should create a video with metadata."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        video = Video(
            user_id=user.id,
            filename="climb.mp4",
            file_path="/uploads/climb.mp4",
            duration_seconds=120.5,
            fps=30.0,
            width=1920,
            height=1080,
            file_size_bytes=50000000,
        )
        db_session.add(video)
        db_session.commit()

        assert video.id is not None
        assert video.filename == "climb.mp4"
        assert video.duration_seconds == 120.5
        assert video.fps == 30.0
        assert video.status == "uploaded"
        assert video.uploaded_at is not None

    def test_video_status_default(self, db_session):
        """Should default status to 'uploaded'."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        video = Video(user_id=user.id, filename="test.mp4", file_path="/path/test.mp4")
        db_session.add(video)
        db_session.commit()

        assert video.status == "uploaded"

    def test_video_user_relationship(self, db_session):
        """Should link to user."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        video = Video(user_id=user.id, filename="test.mp4", file_path="/path/test.mp4")
        db_session.add(video)
        db_session.commit()

        assert video.user.email == "test@example.com"
        assert len(user.videos) == 1

    def test_video_cascade_delete(self, db_session):
        """Should cascade delete when user is deleted."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        video = Video(user_id=user.id, filename="test.mp4", file_path="/path/test.mp4")
        db_session.add(video)
        db_session.commit()

        video_id = video.id

        db_session.delete(user)
        db_session.commit()

        # Video should be deleted
        assert db_session.query(Video).filter(Video.id == video_id).first() is None


class TestAnalysisModel:
    """Tests for the Analysis model."""

    def test_create_analysis(self, db_session):
        """Should create an analysis with full data."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        video = Video(user_id=user.id, filename="test.mp4", file_path="/path/test.mp4")
        db_session.add(video)
        db_session.commit()

        analysis = Analysis(
            video_id=video.id,
            run_metrics=True,
            run_video=False,
            run_quality=True,
            summary={"total_frames": 100, "avg_velocity": 0.5},
            history={"velocity": [0.5, 0.6, 0.4]},
            total_frames=100,
            avg_velocity=0.5,
        )
        db_session.add(analysis)
        db_session.commit()

        assert analysis.id is not None
        assert analysis.summary["total_frames"] == 100
        assert len(analysis.history["velocity"]) == 3
        assert analysis.created_at is not None

    def test_analysis_configuration_defaults(self, db_session):
        """Should have default configuration values."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.flush()  # Flush to get user.id

        video = Video(user_id=user.id, filename="test.mp4", file_path="/path/test.mp4")
        db_session.add(video)
        db_session.commit()

        analysis = Analysis(video_id=video.id)
        db_session.add(analysis)
        db_session.commit()

        assert analysis.run_metrics is True
        assert analysis.run_video is False
        assert analysis.run_quality is True
        assert analysis.dashboard_position == "right"

    def test_analysis_video_relationship(self, db_session):
        """Should link to video."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.flush()  # Flush to get user.id

        video = Video(user_id=user.id, filename="test.mp4", file_path="/path/test.mp4")
        db_session.add(video)
        db_session.commit()

        analysis = Analysis(video_id=video.id)
        db_session.add(analysis)
        db_session.commit()

        assert analysis.video.filename == "test.mp4"
        assert len(video.analyses) == 1

    def test_analysis_cascade_delete(self, db_session):
        """Should cascade delete when video is deleted."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.flush()  # Flush to get user.id

        video = Video(user_id=user.id, filename="test.mp4", file_path="/path/test.mp4")
        db_session.add(video)
        db_session.commit()

        analysis = Analysis(video_id=video.id)
        db_session.add(analysis)
        db_session.commit()

        analysis_id = analysis.id

        db_session.delete(video)
        db_session.commit()

        # Analysis should be deleted
        assert (
            db_session.query(Analysis).filter(Analysis.id == analysis_id).first()
            is None
        )


class TestClimbSessionModel:
    """Tests for the ClimbSession model."""

    def test_create_session(self, db_session):
        """Should create a climbing session."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        session = ClimbSession(
            user_id=user.id,
            name="Morning Bouldering",
            date=datetime(2026, 1, 28, 10, 0),
            location="Local Gym",
            notes="Great session, focused on overhangs",
        )
        db_session.add(session)
        db_session.commit()

        assert session.id is not None
        assert session.name == "Morning Bouldering"
        assert session.location == "Local Gym"
        assert session.notes == "Great session, focused on overhangs"
        assert session.created_at is not None

    def test_session_user_relationship(self, db_session):
        """Should link to user."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        session = ClimbSession(
            user_id=user.id,
            name="Test Session",
            date=datetime(2026, 1, 28),
        )
        db_session.add(session)
        db_session.commit()

        assert session.user.email == "test@example.com"
        assert len(user.sessions) == 1

    def test_session_cascade_delete(self, db_session):
        """Should cascade delete when user is deleted."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        session = ClimbSession(
            user_id=user.id,
            name="Test Session",
            date=datetime(2026, 1, 28),
        )
        db_session.add(session)
        db_session.commit()

        session_id = session.id

        db_session.delete(user)
        db_session.commit()

        # Session should be deleted
        assert (
            db_session.query(ClimbSession).filter(ClimbSession.id == session_id).first()
            is None
        )


class TestDatabaseOperations:
    """Tests for common database operations."""

    def test_query_user_videos(self, db_session):
        """Should query all videos for a user."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        video1 = Video(
            user_id=user.id, filename="climb1.mp4", file_path="/path/climb1.mp4"
        )
        video2 = Video(
            user_id=user.id, filename="climb2.mp4", file_path="/path/climb2.mp4"
        )
        db_session.add_all([video1, video2])
        db_session.commit()

        videos = db_session.query(Video).filter(Video.user_id == user.id).all()
        assert len(videos) == 2

    def test_query_video_analyses(self, db_session):
        """Should query all analyses for a video."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.flush()  # Flush to get user.id

        video = Video(user_id=user.id, filename="test.mp4", file_path="/path/test.mp4")
        db_session.add(video)
        db_session.commit()

        analysis1 = Analysis(video_id=video.id, total_frames=100)
        analysis2 = Analysis(video_id=video.id, total_frames=150)
        db_session.add_all([analysis1, analysis2])
        db_session.commit()

        analyses = (
            db_session.query(Analysis).filter(Analysis.video_id == video.id).all()
        )
        assert len(analyses) == 2

    def test_update_user_data(self, db_session):
        """Should update user data."""
        user = User(
            email="test@example.com", hashed_password="hash", full_name="Old Name"
        )
        db_session.add(user)
        db_session.commit()

        user.full_name = "New Name"
        db_session.commit()

        updated_user = (
            db_session.query(User).filter(User.email == "test@example.com").first()
        )
        assert updated_user.full_name == "New Name"

    def test_filter_videos_by_status(self, db_session):
        """Should filter videos by status."""
        user = User(email="test@example.com", hashed_password="hash")
        db_session.add(user)
        db_session.commit()

        video1 = Video(
            user_id=user.id,
            filename="v1.mp4",
            file_path="/v1.mp4",
            status="uploaded",
        )
        video2 = Video(
            user_id=user.id,
            filename="v2.mp4",
            file_path="/v2.mp4",
            status="processing",
        )
        video3 = Video(
            user_id=user.id,
            filename="v3.mp4",
            file_path="/v3.mp4",
            status="completed",
        )
        db_session.add_all([video1, video2, video3])
        db_session.commit()

        completed = db_session.query(Video).filter(Video.status == "completed").all()
        assert len(completed) == 1
        assert completed[0].filename == "v3.mp4"
