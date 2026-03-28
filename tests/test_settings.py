"""Tests for centralized Settings class."""

import os
from unittest.mock import patch

from app.settings import Settings


class TestSettings:
    """Tests for Settings class."""

    def test_default_values(self):
        """Settings should have sensible defaults."""
        s = Settings(secret_key="x" * 32)
        assert s.max_upload_mb == 500
        assert s.rate_limit_default == "60/minute"
        assert s.rate_limit_auth == 5
        assert ".mp4" in s.allowed_extensions

    def test_max_upload_bytes(self):
        """max_upload_bytes should convert MB to bytes."""
        s = Settings(secret_key="x" * 32, max_upload_mb=100)
        assert s.max_upload_bytes == 100 * 1024 * 1024

    def test_cors_origin_list_empty(self):
        """Empty CORS string should return empty list."""
        s = Settings(secret_key="x" * 32, cors_origins="")
        assert s.cors_origin_list == []

    def test_cors_origin_list_parsing(self):
        """Comma-separated CORS origins should be parsed into list."""
        s = Settings(
            secret_key="x" * 32,
            cors_origins="http://localhost:3000, https://example.com",
        )
        assert s.cors_origin_list == [
            "http://localhost:3000",
            "https://example.com",
        ]

    def test_env_override(self):
        """Environment variables should override defaults."""
        with patch.dict(os.environ, {"MAX_UPLOAD_MB": "200"}):
            s = Settings(secret_key="x" * 32)
            assert s.max_upload_mb == 200

    def test_allowed_extensions_typed(self):
        """allowed_extensions should be a tuple of strings."""
        s = Settings(secret_key="x" * 32)
        assert isinstance(s.allowed_extensions, tuple)
        for ext in s.allowed_extensions:
            assert isinstance(ext, str)
            assert ext.startswith(".")
