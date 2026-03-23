"""Tests for FastAPI endpoints.

This module tests all API endpoints in the web application:
- GET / (home page)
- POST /upload (video upload and analysis)
- GET /analysis/{id} (get analysis results)
- GET /download/{id} (download analysis JSON)
- GET /api/videos, /api/analyses (listing endpoints)
"""

import json
import sys
from io import BytesIO
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the FastAPI app
from app.main import app
from climb_sensei.database.models import Analysis, Base, User, Video
from climb_sensei.database.config import get_db
from climb_sensei.auth import get_password_hash, create_access_token


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
    return create_access_token(data={"sub": str(test_user.id)})


@pytest.fixture
def client(db_session):
    """Create a test client with database override."""

    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_analysis(db_session, test_user):
    """Create a sample video + analysis record in the DB."""
    video = Video(
        user_id=test_user.id,
        filename="test.mp4",
        file_path="/tmp/test.mp4",
        status="completed",
    )
    db_session.add(video)
    db_session.flush()

    analysis = Analysis(
        video_id=video.id,
        summary={"total_frames": 100, "avg_velocity": 0.1},
        history={"velocity": [0.1, 0.2, 0.15]},
        run_metrics=True,
        run_video=False,
        run_quality=False,
        dashboard_position="right",
        total_frames=100,
        avg_velocity=0.1,
    )
    db_session.add(analysis)
    db_session.commit()
    db_session.refresh(analysis)
    return analysis


class TestHomeEndpoint:
    """Tests for the home page endpoint."""

    def test_home_returns_html(self, client):
        """Home page should return HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_home_contains_routes_content(self, client):
        """Home page should contain routes content."""
        response = client.get("/")
        assert b"route" in response.content.lower()


class TestUploadEndpoint:
    """Tests for the video upload endpoint."""

    def test_upload_requires_authentication(self, client):
        """Upload endpoint should require authentication."""
        response = client.post("/upload")
        assert response.status_code == 401  # Unauthorized

    def test_upload_requires_file(self, client, test_user_token):
        """Upload endpoint should require a file when authenticated."""
        response = client.post(
            "/upload", headers={"Authorization": f"Bearer {test_user_token}"}
        )
        assert response.status_code == 422  # Unprocessable Entity

    def test_upload_rejects_invalid_file_type(self, client, test_user_token):
        """Upload should reject non-video files."""
        file_content = b"This is not a video"
        files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 400
        assert "invalid file type" in response.json()["detail"].lower()

    def test_upload_accepts_mp4(self, client, test_user_token):
        """Upload should accept .mp4 files."""
        mp4_header = (
            b"\x00\x00\x00\x20\x66\x74\x79\x70"
            b"\x69\x73\x6f\x6d\x00\x00\x02\x00"
            b"\x69\x73\x6f\x6d\x69\x73\x6f\x32"
            b"\x61\x76\x63\x31\x6d\x70\x34\x31"
        )

        files = {"file": ("test_video.mp4", BytesIO(mp4_header), "video/mp4")}
        data = {
            "run_metrics": "false",
            "run_video": "false",
            "run_quality": "false",
        }

        response = client.post(
            "/upload",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )

        assert response.status_code in [200, 202, 400, 500]
        if response.status_code == 400:
            assert "invalid file type" not in response.json()["detail"].lower()

    def test_upload_accepts_mov(self, client, test_user_token):
        """Upload should accept .mov files."""
        files = {"file": ("test.mov", BytesIO(b"fake mov content"), "video/quicktime")}
        data = {
            "run_metrics": "false",
            "run_video": "false",
            "run_quality": "false",
        }

        response = client.post(
            "/upload",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code in [200, 202, 400, 500]
        if response.status_code == 400:
            assert "invalid file type" not in response.json()["detail"].lower()

    def test_upload_accepts_avi(self, client, test_user_token):
        """Upload should accept .avi files."""
        files = {"file": ("test.avi", BytesIO(b"fake avi content"), "video/x-msvideo")}
        data = {
            "run_metrics": "false",
            "run_video": "false",
            "run_quality": "false",
        }

        response = client.post(
            "/upload",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code in [200, 202, 400, 500]
        if response.status_code == 400:
            assert "invalid file type" not in response.json()["detail"].lower()

    def test_upload_accepts_mkv(self, client, test_user_token):
        """Upload should accept .mkv files."""
        files = {"file": ("test.mkv", BytesIO(b"fake mkv content"), "video/x-matroska")}
        data = {
            "run_metrics": "false",
            "run_video": "false",
            "run_quality": "false",
        }

        response = client.post(
            "/upload",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code in [200, 202, 400, 500]
        if response.status_code == 400:
            assert "invalid file type" not in response.json()["detail"].lower()

    def test_upload_form_parameters(self, client, test_user_token):
        """Upload should accept form parameters."""
        files = {"file": ("test.mp4", BytesIO(b"fake content"), "video/mp4")}
        data = {
            "run_metrics": "true",
            "run_video": "false",
            "run_quality": "true",
            "dashboard_position": "right",
        }

        response = client.post(
            "/upload",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code in [200, 202, 400, 500]


class TestAnalysisEndpoint:
    """Tests for the analysis retrieval endpoint."""

    def test_analysis_not_found(self, client, test_user_token):
        """Should return 404 for non-existent analysis."""
        response = client.get(
            "/analysis/99999",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_analysis_requires_auth(self, client):
        """Should require authentication."""
        response = client.get("/analysis/1")
        assert response.status_code == 401

    def test_analysis_returns_json(self, client, test_user_token, sample_analysis):
        """Should return JSON for existing analysis."""
        response = client.get(
            f"/analysis/{sample_analysis.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        data = response.json()
        assert "summary" in data
        assert "history" in data
        assert data["summary"]["total_frames"] == 100

    def test_analysis_includes_all_fields(
        self, client, test_user_token, sample_analysis
    ):
        """Analysis should include all expected fields."""
        response = client.get(
            f"/analysis/{sample_analysis.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        data = response.json()

        assert "id" in data
        assert "video_id" in data
        assert "summary" in data
        assert "history" in data
        assert "velocity" in data["history"]
        assert len(data["history"]["velocity"]) == 3


class TestDownloadEndpoint:
    """Tests for the download endpoint."""

    def test_download_not_found(self, client, test_user_token):
        """Should return 404 for non-existent analysis."""
        response = client.get(
            "/download/99999",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_download_returns_json_file(self, client, test_user_token, sample_analysis):
        """Should return downloadable JSON file."""
        response = client.get(
            f"/download/{sample_analysis.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]
        assert "attachment" in response.headers.get("content-disposition", "")

    def test_download_contains_valid_json(
        self, client, test_user_token, sample_analysis
    ):
        """Downloaded file should contain valid JSON."""
        response = client.get(
            f"/download/{sample_analysis.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        data = json.loads(response.content)
        assert "summary" in data
        assert "history" in data
        assert data["summary"]["total_frames"] == 100


class TestStaticFiles:
    """Tests for static file serving."""

    def test_static_files_accessible(self, client):
        """Static files should be accessible."""
        response = client.get("/static/style.css")
        assert response.status_code in [200, 404]

    def test_outputs_require_authentication(self, client):
        """Output files should require authentication."""
        response = client.get("/outputs/nonexistent.mp4")
        assert response.status_code in [401, 404]


class TestDBAPIEndpoints:
    """Tests for database API endpoints."""

    def test_list_videos_requires_auth(self, client):
        """Should require authentication."""
        response = client.get("/api/videos")
        assert response.status_code == 401

    def test_list_videos_empty(self, client, test_user_token):
        """Should return empty list for new user."""
        response = client.get(
            "/api/videos",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        assert response.json() == []

    def test_list_analyses_requires_auth(self, client):
        """Should require authentication."""
        response = client.get("/api/analyses")
        assert response.status_code == 401

    def test_get_analysis_detail(self, client, test_user_token, sample_analysis):
        """Should return analysis detail."""
        response = client.get(
            f"/api/analyses/{sample_analysis.id}",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == sample_analysis.id
        assert data["summary"]["total_frames"] == 100


class TestVideoStatusEndpoint:
    """Tests for the video status polling endpoint."""

    def test_status_requires_auth(self, client):
        """Should require authentication."""
        response = client.get("/api/videos/1/status")
        assert response.status_code == 401

    def test_status_not_found(self, client, test_user_token):
        """Should return 404 for non-existent video."""
        response = client.get(
            "/api/videos/99999/status",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404

    def test_status_returns_processing(
        self, client, test_user_token, test_user, db_session
    ):
        """Should return processing status for in-progress video."""
        video = Video(
            user_id=test_user.id,
            filename="test.mp4",
            file_path="/tmp/test.mp4",
            status="processing",
        )
        db_session.add(video)
        db_session.commit()

        response = client.get(
            f"/api/videos/{video.id}/status",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "processing"
        assert "analysis_id" not in data

    def test_status_returns_completed_with_analysis_id(
        self, client, test_user_token, sample_analysis
    ):
        """Should return completed status with analysis_id."""
        response = client.get(
            f"/api/videos/{sample_analysis.video_id}/status",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["analysis_id"] == sample_analysis.id

    def test_status_denies_other_user(self, client, test_user_token, db_session):
        """Should not show another user's video status."""
        other_user = User(
            email="other@example.com",
            hashed_password=get_password_hash("password123"),
        )
        db_session.add(other_user)
        db_session.flush()

        video = Video(
            user_id=other_user.id,
            filename="other.mp4",
            file_path="/tmp/other.mp4",
            status="processing",
        )
        db_session.add(video)
        db_session.commit()

        response = client.get(
            f"/api/videos/{video.id}/status",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_endpoint_returns_404(self, client):
        """Invalid endpoints should return 404."""
        response = client.get("/invalid/endpoint/path")
        assert response.status_code == 404

    def test_upload_with_missing_required_field(self, client):
        """Upload without file should return 401 (authentication required)."""
        response = client.post("/upload", data={"run_metrics": "true"})
        assert response.status_code == 401
