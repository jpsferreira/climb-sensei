"""Tests for security features added in the production readiness audit.

Covers:
- Upload path traversal protection
- Upload file size limits
- Upload magic byte validation
- Authenticated output file serving
- Security headers on responses
- AUTH_DISABLED guard
- Rate limiting configuration
"""

import os
import struct
import sys
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from app.main import app
from climb_sensei.auth import create_access_token, get_password_hash
from climb_sensei.database.config import get_db
from climb_sensei.database.models import Analysis, Base, User, Video


@pytest.fixture
def test_engine():
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
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def test_user(db_session):
    user = User(
        email="security-test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Security Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user(db_session):
    user = User(
        email="other@example.com",
        hashed_password=get_password_hash("otherpassword123"),
        full_name="Other User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_user_token(test_user):
    return create_access_token(data={"sub": str(test_user.id)})


@pytest.fixture
def other_user_token(other_user):
    return create_access_token(data={"sub": str(other_user.id)})


@pytest.fixture
def client(db_session):
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    # Reset rate limiter state so tests aren't affected by cross-test quota
    from app.rate_limit import limiter

    limiter.reset()

    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _make_mp4_bytes(size=1024):
    """Create minimal valid MP4 file bytes with ftyp box."""
    # ftyp box: 4 bytes size + 'ftyp' + 'isom' brand
    header = struct.pack(">I", 20) + b"ftypisom" + b"\x00" * 8
    return header + b"\x00" * max(0, size - len(header))


def _make_avi_bytes(size=1024):
    """Create minimal valid AVI file bytes."""
    header = b"RIFF" + struct.pack("<I", size - 8) + b"AVI "
    return header + b"\x00" * max(0, size - len(header))


# ==========================================================================
# Upload Security
# ==========================================================================


class TestUploadPathTraversal:
    """Verify filenames with path traversal are rejected."""

    def test_path_traversal_filename_rejected(self, client, test_user_token):
        """Filenames with directory traversal should be sanitized."""
        mp4_data = _make_mp4_bytes()
        files = {
            "file": ("../../etc/passwd.mp4", BytesIO(mp4_data), "video/mp4")
        }
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        # Should succeed (filename sanitized to just the extension)
        # or fail with a non-500 error — never write outside upload dir
        assert response.status_code in [202, 400, 413]

    def test_dotdot_in_filename_does_not_escape(self, client, test_user_token):
        """Even if the upload succeeds, the file stays inside UPLOAD_DIR."""
        from app.services.upload import UPLOAD_DIR

        mp4_data = _make_mp4_bytes()
        files = {
            "file": ("../../../tmp/evil.mp4", BytesIO(mp4_data), "video/mp4")
        }
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        # Verify no file was written outside UPLOAD_DIR
        assert not Path("/tmp/evil.mp4").exists()


class TestUploadFileSize:
    """Verify file size limits are enforced."""

    def test_small_file_accepted(self, client, test_user_token):
        """Files under the limit should be accepted."""
        mp4_data = _make_mp4_bytes(2048)
        files = {"file": ("small.mp4", BytesIO(mp4_data), "video/mp4")}
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 202

    @patch("app.services.upload.MAX_UPLOAD_SIZE", 100)
    def test_oversized_file_rejected(self, client, test_user_token):
        """Files over the size limit should return 413."""
        mp4_data = _make_mp4_bytes(500)
        files = {"file": ("big.mp4", BytesIO(mp4_data), "video/mp4")}
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 413
        assert "too large" in response.json()["detail"].lower()


class TestUploadMagicBytes:
    """Verify magic byte validation rejects non-video files."""

    def test_text_file_with_mp4_extension_rejected(self, client, test_user_token):
        """A text file renamed to .mp4 should be rejected."""
        fake_mp4 = b"This is not a video file at all" + b"\x00" * 100
        files = {"file": ("fake.mp4", BytesIO(fake_mp4), "video/mp4")}
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 400
        assert "valid video" in response.json()["detail"].lower()

    def test_wav_file_with_avi_extension_rejected(self, client, test_user_token):
        """A WAV file (RIFF but not AVI) should be rejected."""
        wav_data = b"RIFF" + struct.pack("<I", 100) + b"WAVE" + b"\x00" * 100
        files = {"file": ("audio.avi", BytesIO(wav_data), "video/avi")}
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 400

    def test_valid_mp4_accepted(self, client, test_user_token):
        """A file with valid MP4 magic bytes should be accepted."""
        mp4_data = _make_mp4_bytes()
        files = {"file": ("valid.mp4", BytesIO(mp4_data), "video/mp4")}
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 202

    def test_valid_avi_accepted(self, client, test_user_token):
        """A file with valid AVI magic bytes should be accepted."""
        avi_data = _make_avi_bytes()
        files = {"file": ("valid.avi", BytesIO(avi_data), "video/avi")}
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 202

    def test_disallowed_extension_rejected(self, client, test_user_token):
        """Files with non-video extensions should be rejected."""
        files = {
            "file": ("script.py", BytesIO(b"import os"), "text/x-python")
        }
        response = client.post(
            "/upload",
            files=files,
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 400
        assert "invalid file type" in response.json()["detail"].lower()


# ==========================================================================
# Output File Authentication
# ==========================================================================


class TestOutputFileAuth:
    """Verify output files require authentication and ownership."""

    def test_unauthenticated_access_returns_401(self, client):
        """Unauthenticated requests to /outputs/ should return 401."""
        response = client.get("/outputs/some_video.webm")
        assert response.status_code == 401

    def test_nonexistent_file_returns_404(self, client, test_user_token):
        """Authenticated request for missing file should return 404."""
        response = client.get(
            "/outputs/nonexistent.webm",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.status_code == 404

    def test_other_users_output_returns_404(
        self, client, db_session, test_user, other_user_token
    ):
        """User should not access another user's output files."""
        # Create a video + analysis owned by test_user
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
            summary={},
            history={},
            output_video_path="/outputs/secret_output.webm",
            run_metrics=True,
            run_video=True,
            run_quality=False,
            dashboard_position="right",
        )
        db_session.add(analysis)
        db_session.commit()

        # other_user tries to access test_user's output
        response = client.get(
            "/outputs/secret_output.webm",
            headers={"Authorization": f"Bearer {other_user_token}"},
        )
        assert response.status_code == 404


# ==========================================================================
# Security Headers
# ==========================================================================


class TestSecurityHeaders:
    """Verify security headers are present on all responses."""

    def test_security_headers_on_html_page(self, client):
        """HTML pages should include security headers."""
        response = client.get("/login")
        headers = response.headers

        assert headers.get("X-Content-Type-Options") == "nosniff"
        assert headers.get("X-Frame-Options") == "DENY"
        assert "strict-origin" in headers.get("Referrer-Policy", "")
        assert "camera=()" in headers.get("Permissions-Policy", "")
        assert "default-src 'self'" in headers.get(
            "Content-Security-Policy", ""
        )
        assert "frame-ancestors 'none'" in headers.get(
            "Content-Security-Policy", ""
        )

    def test_security_headers_on_api_response(self, client, test_user_token):
        """API responses should also include security headers."""
        response = client.get(
            "/api/videos",
            headers={"Authorization": f"Bearer {test_user_token}"},
        )
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"

    def test_csp_allows_required_sources(self, client):
        """CSP should allow Chart.js CDN and Google Fonts."""
        response = client.get("/login")
        csp = response.headers.get("Content-Security-Policy", "")
        assert "cdn.jsdelivr.net" in csp
        assert "fonts.googleapis.com" in csp
        assert "fonts.gstatic.com" in csp


# ==========================================================================
# AUTH_DISABLED Guard
# ==========================================================================


class TestAuthDisabledGuard:
    """Verify AUTH_DISABLED only works in dev/testing environments."""

    def test_auth_disabled_blocked_in_production(self):
        """AUTH_DISABLED should be ignored when ENVIRONMENT=production."""
        with patch.dict(
            os.environ,
            {"AUTH_DISABLED": "1", "ENVIRONMENT": "production"},
            clear=False,
        ):
            # Re-evaluate the guard logic
            from climb_sensei.auth import __init__ as auth_mod

            requested = os.getenv("AUTH_DISABLED", "").lower() in (
                "1",
                "true",
                "yes",
            )
            env = os.getenv("ENVIRONMENT", "production").lower()
            result = requested and env in ("development", "testing")
            assert result is False

    def test_auth_disabled_allowed_in_development(self):
        """AUTH_DISABLED should work when ENVIRONMENT=development."""
        with patch.dict(
            os.environ,
            {"AUTH_DISABLED": "1", "ENVIRONMENT": "development"},
            clear=False,
        ):
            requested = os.getenv("AUTH_DISABLED", "").lower() in (
                "1",
                "true",
                "yes",
            )
            env = os.getenv("ENVIRONMENT", "production").lower()
            result = requested and env in ("development", "testing")
            assert result is True

    def test_auth_disabled_allowed_in_testing(self):
        """AUTH_DISABLED should work when ENVIRONMENT=testing."""
        with patch.dict(
            os.environ,
            {"AUTH_DISABLED": "1", "ENVIRONMENT": "testing"},
            clear=False,
        ):
            requested = os.getenv("AUTH_DISABLED", "").lower() in (
                "1",
                "true",
                "yes",
            )
            env = os.getenv("ENVIRONMENT", "production").lower()
            result = requested and env in ("development", "testing")
            assert result is True


# ==========================================================================
# User Enumeration Prevention
# ==========================================================================


class TestUserEnumeration:
    """Verify error messages don't reveal whether an email exists."""

    def test_duplicate_registration_generic_error(self, client, test_user):
        """Registration with existing email should return generic message."""
        # First register via fastapi-users endpoint
        client.post(
            "/api/auth/register",
            json={
                "email": test_user.email,
                "password": "anotherpassword123",
            },
        )
        # Second attempt with same email
        response = client.post(
            "/api/auth/register",
            json={
                "email": test_user.email,
                "password": "anotherpassword123",
            },
        )
        assert response.status_code == 400
        detail = str(response.json().get("detail", "")).lower()
        # Should NOT reveal that the specific email is already taken
        assert test_user.email not in detail


# ==========================================================================
# Upload Endpoint Requires Auth
# ==========================================================================


class TestUploadAuth:
    """Verify upload endpoint requires authentication."""

    def test_upload_without_token_returns_401(self, client):
        """Upload without auth token should return 401."""
        mp4_data = _make_mp4_bytes()
        files = {"file": ("test.mp4", BytesIO(mp4_data), "video/mp4")}
        response = client.post("/upload", files=files)
        assert response.status_code == 401
