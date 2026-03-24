"""FastAPI web application for ClimbingSensei video analysis.

Slim app factory: creates the FastAPI instance, mounts routers,
configures static files and middleware.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

from climb_sensei.logging_config import configure_logging
from climb_sensei.auth.routes_new import router as auth_router
from climb_sensei.database.config import init_db
from climb_sensei.progress.routes import router as progress_router
from climb_sensei.progress.route_routes import router as route_router
from climb_sensei.progress.attempt_routes import router as attempt_router

from app.rate_limit import limiter
from app.routers.api import router as api_router
from app.routers.pages import router as pages_router
from app.services.upload import OUTPUT_DIR, UPLOAD_DIR

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to every response."""

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        permissions = "camera=(), microphone=(), geolocation=()"
        response.headers["Permissions-Policy"] = permissions
        csp = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src https://fonts.gstatic.com; "
            "img-src 'self' data:; "
            "media-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'"
        )
        response.headers["Content-Security-Policy"] = csp
        return response


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log method, path, status, and duration for every request."""

    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s %d %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response


def _validate_config():
    """Log warnings for questionable production configuration."""
    env = os.getenv("ENVIRONMENT", "production").lower()
    if env == "production":
        db_url = os.getenv("DATABASE_URL", "")
        if db_url.startswith("sqlite") or not db_url:
            logger.warning(
                "Using SQLite in production is not recommended for concurrent access"
            )
        gid = os.getenv("GOOGLE_CLIENT_ID", "")
        if not gid or gid.startswith("YOUR_"):
            logger.warning("Google OAuth is not configured (GOOGLE_CLIENT_ID missing)")


@asynccontextmanager
async def lifespan(app):
    """Application lifespan: run startup validation."""
    _validate_config()
    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance.

    Returns:
        FastAPI: Configured application with routers, static files,
            and database initialised.
    """
    configure_logging()

    application = FastAPI(
        title="ClimbingSensei Web App",
        description="Upload climbing videos and analyze performance",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Rate limiting
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    application.add_middleware(SlowAPIMiddleware)

    # CORS — only if explicit origins are configured (must be outermost middleware)
    cors_origins = [
        o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
    ]
    if cors_origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # Security headers (runs after CORS so preflight responses also get headers)
    application.add_middleware(SecurityHeadersMiddleware)

    # Request logging
    application.add_middleware(RequestLoggingMiddleware)

    # Initialize database
    init_db()

    # Routers
    application.include_router(auth_router, prefix="/api")
    application.include_router(progress_router)
    application.include_router(route_router)
    application.include_router(attempt_router)
    application.include_router(api_router)
    application.include_router(pages_router)

    # Static files — ensure directories exist (CI may not have them)
    for dir_path in [BASE_DIR / "static", OUTPUT_DIR, UPLOAD_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

    application.mount(
        "/static",
        StaticFiles(directory=str(BASE_DIR / "static")),
        name="static",
    )
    # Note: /uploads is NOT mounted as static — files are temporary and deleted
    # after analysis. /outputs served via authenticated route in api.py.

    return application


# Module-level instance for backward compatibility with run_app.py and uvicorn
app = create_app()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ClimbingSensei Web App")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("Press CTRL+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
