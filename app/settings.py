"""Centralized application settings.

Uses Pydantic BaseSettings so values can come from environment variables,
.env files, or defaults. Import `settings` from this module wherever config
is needed instead of calling os.getenv() directly.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ── Environment ─────────────────────────────────────────
    environment: str = ""
    debug: bool = False
    log_level: str = "INFO"

    # ── Database ────────────────────────────────────────────
    database_url: str = "sqlite:///./climbsensei.db"

    # ── Auth ────────────────────────────────────────────────
    secret_key: str = ""
    access_token_expire_minutes: int = 30
    auth_disabled: bool = False

    # ── OAuth ───────────────────────────────────────────────
    google_client_id: str = ""
    google_client_secret: str = ""
    oauth_state_secret: str = ""
    oauth_redirect_url: str = "http://localhost:8000/api/auth/google/callback"

    # ── Upload ──────────────────────────────────────────────
    max_upload_mb: int = 500
    allowed_extensions: tuple = (".mp4", ".avi", ".mov", ".mkv", ".webm")

    # ── Rate Limiting ───────────────────────────────────────
    rate_limit_default: str = "60/minute"
    rate_limit_upload: str = "10/minute"
    rate_limit_auth: int = 5  # requests per minute per IP

    # ── CORS ────────────────────────────────────────────────
    cors_origins: str = ""  # Comma-separated origins

    # ── Docs ────────────────────────────────────────────────
    base_doc_url: str = "https://jpsferreira.github.io/climb-sensei/metrics/"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @property
    def cors_origin_list(self) -> list[str]:
        """Parse comma-separated CORS origins into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        """Upload size limit in bytes."""
        return self.max_upload_mb * 1024 * 1024


settings = Settings()
