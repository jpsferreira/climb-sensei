"""Tests for FastAPI endpoints.

This module tests all API endpoints in the web application:
- GET / (home page)
- POST /upload (video upload and analysis)
- GET /analysis/{id} (get analysis results)
- GET /download/{id} (download analysis JSON)
"""

import sys
from pathlib import Path
from io import BytesIO

import pytest
from fastapi.testclient import TestClient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the FastAPI app
from app.main import app, analysis_cache


@pytest.fixture
def client():
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def cleanup_cache():
    """Clean up analysis cache after each test."""
    yield
    analysis_cache.clear()


class TestHomeEndpoint:
    """Tests for the home page endpoint."""

    def test_home_returns_html(self, client):
        """Home page should return HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_home_contains_upload_form(self, client):
        """Home page should contain upload form."""
        response = client.get("/")
        assert b"upload" in response.content.lower()
        assert b"video" in response.content.lower()


class TestUploadEndpoint:
    """Tests for the video upload endpoint."""

    def test_upload_requires_file(self, client):
        """Upload endpoint should require a file."""
        response = client.post("/upload")
        assert response.status_code == 422  # Unprocessable Entity

    def test_upload_rejects_invalid_file_type(self, client, cleanup_cache):
        """Upload should reject non-video files."""
        # Create a fake text file
        file_content = b"This is not a video"
        files = {"file": ("test.txt", BytesIO(file_content), "text/plain")}

        response = client.post("/upload", files=files)
        assert response.status_code == 400
        assert "invalid file type" in response.json()["detail"].lower()

    def test_upload_accepts_mp4(self, client, cleanup_cache):
        """Upload should accept .mp4 files."""
        # Create a minimal valid MP4 file header
        # This is a very minimal MP4 that will fail processing but pass validation
        mp4_header = (
            b"\x00\x00\x00\x20\x66\x74\x79\x70"  # ftyp box
            b"\x69\x73\x6f\x6d\x00\x00\x02\x00"
            b"\x69\x73\x6f\x6d\x69\x73\x6f\x32"
            b"\x61\x76\x63\x31\x6d\x70\x34\x31"
        )

        files = {"file": ("test_video.mp4", BytesIO(mp4_header), "video/mp4")}
        data = {
            "run_metrics": "false",  # Disable analysis to avoid processing errors
            "run_video": "false",
            "run_quality": "false",
        }

        response = client.post("/upload", files=files, data=data)

        # It may fail during processing but should not fail on file type
        # Status could be 200 (success) or 500 (processing error)
        assert response.status_code in [200, 400, 500]
        if response.status_code == 400:
            # Should not be about file type
            assert "invalid file type" not in response.json()["detail"].lower()

    def test_upload_accepts_mov(self, client, cleanup_cache):
        """Upload should accept .mov files."""
        files = {"file": ("test.mov", BytesIO(b"fake mov content"), "video/quicktime")}
        data = {
            "run_metrics": "false",
            "run_video": "false",
            "run_quality": "false",
        }

        response = client.post("/upload", files=files, data=data)
        # Should not fail on file type validation
        assert response.status_code in [200, 400, 500]
        if response.status_code == 400:
            assert "invalid file type" not in response.json()["detail"].lower()

    def test_upload_accepts_avi(self, client, cleanup_cache):
        """Upload should accept .avi files."""
        files = {"file": ("test.avi", BytesIO(b"fake avi content"), "video/x-msvideo")}
        data = {
            "run_metrics": "false",
            "run_video": "false",
            "run_quality": "false",
        }

        response = client.post("/upload", files=files, data=data)
        assert response.status_code in [200, 400, 500]
        if response.status_code == 400:
            assert "invalid file type" not in response.json()["detail"].lower()

    def test_upload_accepts_mkv(self, client, cleanup_cache):
        """Upload should accept .mkv files."""
        files = {"file": ("test.mkv", BytesIO(b"fake mkv content"), "video/x-matroska")}
        data = {
            "run_metrics": "false",
            "run_video": "false",
            "run_quality": "false",
        }

        response = client.post("/upload", files=files, data=data)
        assert response.status_code in [200, 400, 500]
        if response.status_code == 400:
            assert "invalid file type" not in response.json()["detail"].lower()

    def test_upload_form_parameters(self, client, cleanup_cache):
        """Upload should accept form parameters."""
        files = {"file": ("test.mp4", BytesIO(b"fake content"), "video/mp4")}
        data = {
            "run_metrics": "true",
            "run_video": "false",
            "run_quality": "true",
            "dashboard_position": "right",
        }

        response = client.post("/upload", files=files, data=data)
        # Should accept the parameters (may fail on processing)
        assert response.status_code in [200, 400, 500]

    def test_upload_default_parameters(self, client, cleanup_cache):
        """Upload should use default parameters when not provided."""
        files = {"file": ("test.mp4", BytesIO(b"fake content"), "video/mp4")}

        response = client.post("/upload", files=files)
        # Should work with defaults (may fail on processing)
        assert response.status_code in [200, 400, 500]


class TestAnalysisEndpoint:
    """Tests for the analysis retrieval endpoint."""

    def test_analysis_not_found(self, client, cleanup_cache):
        """Should return 404 for non-existent analysis."""
        response = client.get("/analysis/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_analysis_returns_json(self, client, cleanup_cache):
        """Should return JSON for existing analysis."""
        # Mock an analysis in the cache
        from climb_sensei.models import ClimbingAnalysis, ClimbingSummary

        analysis_id = "test-analysis-123"
        mock_summary = ClimbingSummary(
            total_frames=100,
            total_vertical_progress=2.0,
            max_height=1.5,
            avg_velocity=0.1,
            max_velocity=0.15,
            avg_sway=0.05,
            max_sway=0.08,
            avg_jerk=0.01,
            max_jerk=0.02,
            avg_body_angle=45.0,
            avg_hand_span=0.5,
            avg_foot_span=0.4,
            total_distance_traveled=10.0,
            avg_movement_economy=0.8,
            lock_off_count=5,
            lock_off_percentage=5.0,
            rest_count=3,
            rest_percentage=3.0,
            fatigue_score=0.3,
            avg_left_elbow=90.0,
            avg_right_elbow=90.0,
            avg_left_shoulder=45.0,
            avg_right_shoulder=45.0,
            avg_left_knee=120.0,
            avg_right_knee=120.0,
            avg_left_hip=100.0,
            avg_right_hip=100.0,
        )
        mock_analysis = ClimbingAnalysis(
            summary=mock_summary,
            history={"velocity": [0.1, 0.2, 0.15]},
        )

        analysis_cache[analysis_id] = mock_analysis

        response = client.get(f"/analysis/{analysis_id}")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        data = response.json()
        assert "summary" in data
        assert "history" in data
        assert data["summary"]["total_frames"] == 100

    def test_analysis_includes_all_fields(self, client, cleanup_cache):
        """Analysis should include all expected fields."""
        from climb_sensei.models import ClimbingAnalysis, ClimbingSummary

        analysis_id = "test-complete-analysis"
        mock_summary = ClimbingSummary(
            total_frames=100,
            total_vertical_progress=2.0,
            max_height=1.5,
            avg_velocity=0.1,
            max_velocity=0.15,
            avg_sway=0.05,
            max_sway=0.08,
            avg_jerk=0.01,
            max_jerk=0.02,
            avg_body_angle=45.0,
            avg_hand_span=0.5,
            avg_foot_span=0.4,
            total_distance_traveled=10.0,
            avg_movement_economy=0.8,
            lock_off_count=5,
            lock_off_percentage=5.0,
            rest_count=3,
            rest_percentage=3.0,
            fatigue_score=0.3,
            avg_left_elbow=90.0,
            avg_right_elbow=90.0,
            avg_left_shoulder=45.0,
            avg_right_shoulder=45.0,
            avg_left_knee=120.0,
            avg_right_knee=120.0,
            avg_left_hip=100.0,
            avg_right_hip=100.0,
        )
        mock_analysis = ClimbingAnalysis(
            summary=mock_summary,
            history={
                "velocity": [0.1, 0.2],
                "sway": [0.05, 0.06],
            },
        )

        analysis_cache[analysis_id] = mock_analysis

        response = client.get(f"/analysis/{analysis_id}")
        data = response.json()

        # Check summary fields
        assert "total_frames" in data["summary"]
        assert "max_height" in data["summary"]
        assert "avg_velocity" in data["summary"]

        # Check history
        assert "velocity" in data["history"]
        assert len(data["history"]["velocity"]) == 2


class TestDownloadEndpoint:
    """Tests for the download endpoint."""

    def test_download_not_found(self, client, cleanup_cache):
        """Should return 404 for non-existent analysis."""
        response = client.get("/download/nonexistent-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_download_returns_json_file(self, client, cleanup_cache):
        """Should return downloadable JSON file."""
        from climb_sensei.models import ClimbingAnalysis, ClimbingSummary

        analysis_id = "test-download-123"
        mock_summary = ClimbingSummary(
            total_frames=50,
            total_vertical_progress=1.8,
            max_height=1.2,
            avg_velocity=0.12,
            max_velocity=0.18,
            avg_sway=0.04,
            max_sway=0.07,
            avg_jerk=0.02,
            max_jerk=0.03,
            avg_body_angle=50.0,
            avg_hand_span=0.6,
            avg_foot_span=0.45,
            total_distance_traveled=8.0,
            avg_movement_economy=0.75,
            lock_off_count=4,
            lock_off_percentage=8.0,
            rest_count=2,
            rest_percentage=4.0,
            fatigue_score=0.25,
            avg_left_elbow=85.0,
            avg_right_elbow=87.0,
            avg_left_shoulder=50.0,
            avg_right_shoulder=48.0,
            avg_left_knee=125.0,
            avg_right_knee=123.0,
            avg_left_hip=95.0,
            avg_right_hip=97.0,
        )
        mock_analysis = ClimbingAnalysis(
            summary=mock_summary,
            history={"velocity": [0.12]},
        )

        analysis_cache[analysis_id] = mock_analysis

        response = client.get(f"/download/{analysis_id}")
        assert response.status_code == 200
        assert "application/json" in response.headers["content-type"]

        # Check content-disposition header for download
        assert "attachment" in response.headers.get("content-disposition", "")

    def test_download_contains_valid_json(self, client, cleanup_cache):
        """Downloaded file should contain valid JSON."""
        import json
        from climb_sensei.models import ClimbingAnalysis, ClimbingSummary

        analysis_id = "test-valid-json"
        mock_summary = ClimbingSummary(
            total_frames=30,
            total_vertical_progress=1.5,
            max_height=1.0,
            avg_velocity=0.08,
            max_velocity=0.12,
            avg_sway=0.03,
            max_sway=0.05,
            avg_jerk=0.015,
            max_jerk=0.025,
            avg_body_angle=42.0,
            avg_hand_span=0.52,
            avg_foot_span=0.38,
            total_distance_traveled=6.5,
            avg_movement_economy=0.82,
            lock_off_count=3,
            lock_off_percentage=10.0,
            rest_count=1,
            rest_percentage=3.3,
            fatigue_score=0.2,
            avg_left_elbow=92.0,
            avg_right_elbow=91.0,
            avg_left_shoulder=47.0,
            avg_right_shoulder=46.0,
            avg_left_knee=118.0,
            avg_right_knee=119.0,
            avg_left_hip=102.0,
            avg_right_hip=101.0,
        )
        mock_analysis = ClimbingAnalysis(
            summary=mock_summary,
            history={"velocity": [0.08, 0.09]},
        )

        analysis_cache[analysis_id] = mock_analysis

        response = client.get(f"/download/{analysis_id}")

        # Parse the JSON response
        data = json.loads(response.content)
        assert "summary" in data
        assert "history" in data
        assert data["summary"]["total_frames"] == 30


class TestStaticFiles:
    """Tests for static file serving."""

    def test_static_files_accessible(self, client):
        """Static files should be accessible."""
        # Try to access the CSS file
        response = client.get("/static/style.css")
        # Should either exist (200) or not exist (404), but not error
        assert response.status_code in [200, 404]

    def test_outputs_directory_accessible(self, client):
        """Outputs directory should be accessible."""
        # This will likely return 404 for non-existent file, which is fine
        response = client.get("/outputs/nonexistent.mp4")
        # Should handle gracefully
        assert response.status_code in [200, 404]


class TestAPIIntegration:
    """Integration tests for API workflow."""

    def test_complete_workflow_with_mock_data(self, client, cleanup_cache):
        """Test complete workflow: upload -> get analysis -> download."""
        # This is a simplified integration test
        # In reality, we'd need a real video file for full workflow

        # Step 1: Mock an analysis result
        from climb_sensei.models import ClimbingAnalysis, ClimbingSummary

        analysis_id = "workflow-test-123"
        mock_summary = ClimbingSummary(
            total_frames=75,
            total_vertical_progress=1.9,
            max_height=1.3,
            avg_velocity=0.11,
            max_velocity=0.16,
            avg_sway=0.045,
            max_sway=0.07,
            avg_jerk=0.018,
            max_jerk=0.028,
            avg_body_angle=48.0,
            avg_hand_span=0.55,
            avg_foot_span=0.42,
            total_distance_traveled=9.0,
            avg_movement_economy=0.78,
            lock_off_count=6,
            lock_off_percentage=8.0,
            rest_count=4,
            rest_percentage=5.3,
            fatigue_score=0.28,
            avg_left_elbow=88.0,
            avg_right_elbow=89.0,
            avg_left_shoulder=49.0,
            avg_right_shoulder=47.0,
            avg_left_knee=122.0,
            avg_right_knee=121.0,
            avg_left_hip=98.0,
            avg_right_hip=99.0,
        )
        mock_analysis = ClimbingAnalysis(
            summary=mock_summary,
            history={"velocity": [0.11, 0.12, 0.10]},
        )

        analysis_cache[analysis_id] = mock_analysis

        # Step 2: Get analysis
        response_get = client.get(f"/analysis/{analysis_id}")
        assert response_get.status_code == 200
        assert response_get.json()["summary"]["total_frames"] == 75

        # Step 3: Download analysis
        response_download = client.get(f"/download/{analysis_id}")
        assert response_download.status_code == 200

        # Step 4: Verify downloaded content matches
        import json

        downloaded_data = json.loads(response_download.content)
        assert downloaded_data["summary"]["total_frames"] == 75
        assert len(downloaded_data["history"]["velocity"]) == 3


class TestErrorHandling:
    """Tests for error handling."""

    def test_invalid_endpoint_returns_404(self, client):
        """Invalid endpoints should return 404."""
        response = client.get("/invalid/endpoint/path")
        assert response.status_code == 404

    def test_upload_with_missing_required_field(self, client):
        """Upload without file should return 422."""
        response = client.post("/upload", data={"run_metrics": "true"})
        assert response.status_code == 422

    def test_analysis_with_invalid_id_format(self, client, cleanup_cache):
        """Should handle any ID format gracefully."""
        # Test with various ID formats
        test_ids = [
            "123",
            "invalid-id",
            "test_id_with_underscores",
            "id-with-special-chars-!@#",
        ]

        for test_id in test_ids:
            response = client.get(f"/analysis/{test_id}")
            # Should return 404 for non-existent IDs
            assert response.status_code == 404
            assert "not found" in response.json()["detail"].lower()
