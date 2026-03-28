"""FastAPI web application for ClimbingSensei video analysis.

Slim app factory: creates the FastAPI instance, mounts routers,
configures static files and middleware.
"""

import logging
import os
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.middleware import SlowAPIMiddleware
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
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
            "connect-src 'self' https://cdn.jsdelivr.net https://fonts.googleapis.com https://fonts.gstatic.com; "
            "frame-ancestors 'none'"
        )
        response.headers["Content-Security-Policy"] = csp
        # Allow service worker to control the root scope
        if request.url.path.endswith("/sw.js"):
            response.headers["Service-Worker-Allowed"] = "/"
        return response


class AuthRateLimitMiddleware(BaseHTTPMiddleware):
    """Stricter rate limiting for authentication endpoints.

    Limits login and register to 5 requests/minute per IP
    to prevent brute-force and registration spam.

    Note: Uses in-process storage — effective for single-worker
    deployments. For multi-worker setups, use SlowAPI with a
    Redis backend instead.
    """

    AUTH_PATHS = {
        "/api/v1/auth/jwt/login",
        "/api/v1/auth/register",
        "/api/auth/jwt/login",  # Legacy paths (before redirect)
        "/api/auth/register",
    }
    MAX_REQUESTS = 5
    WINDOW_SECONDS = 60

    def __init__(self, app):
        super().__init__(app)
        self._counts: dict[str, list[float]] = {}

    def reset(self):
        """Reset all counters (used by tests)."""
        self._counts.clear()

    async def dispatch(self, request, call_next):
        if request.method == "POST" and request.url.path in self.AUTH_PATHS:
            import time as _time

            ip = get_remote_address(request)
            key = f"{ip}:{request.url.path}"
            now = _time.time()

            # Clean old entries and evict empty keys
            timestamps = self._counts.get(key, [])
            timestamps = [t for t in timestamps if now - t < self.WINDOW_SECONDS]
            if not timestamps:
                self._counts.pop(key, None)

            if len(timestamps) >= self.MAX_REQUESTS:
                from starlette.responses import JSONResponse

                logger.warning(
                    "Auth rate limit exceeded: %s from %s", request.url.path, ip
                )
                return JSONResponse(
                    status_code=429,
                    content={"detail": "Too many attempts. Please try again later."},
                )

            timestamps.append(now)
            self._counts[key] = timestamps

        return await call_next(request)


class ApiV1RedirectMiddleware(BaseHTTPMiddleware):
    """Redirect legacy /api/ paths to /api/v1/ for backward compatibility."""

    async def dispatch(self, request, call_next):
        path = request.url.path
        # Redirect /api/... to /api/v1/... (but not /api/v1/ itself)
        if path.startswith("/api/") and not path.startswith("/api/v1/"):
            from starlette.responses import RedirectResponse

            new_path = "/api/v1/" + path[5:]  # len("/api/") == 5
            query = request.url.query
            new_url = new_path + (f"?{query}" if query else "")
            return RedirectResponse(url=new_url, status_code=307)
        return await call_next(request)


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
    )

    # Rate limiting
    application.state.limiter = limiter
    application.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    application.add_middleware(SlowAPIMiddleware)

    # CORS — only if explicit origins are configured (must be outermost middleware)
    from app.settings import settings as app_settings

    if app_settings.cors_origin_list:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=app_settings.cors_origin_list,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )

    # Security headers (runs after CORS so preflight responses also get headers)
    application.add_middleware(SecurityHeadersMiddleware)

    # Request logging
    application.add_middleware(RequestLoggingMiddleware)

    # Stricter rate limiting on auth endpoints (5/min per IP)
    # Must be before redirect middleware so legacy /api/auth/ paths are also limited
    application.add_middleware(AuthRateLimitMiddleware)

    # Backward-compat: redirect /api/... → /api/v1/...
    application.add_middleware(ApiV1RedirectMiddleware)

    # Initialize database
    init_db()

    # Startup validation
    @application.on_event("startup")
    def validate_config():
        env = os.getenv("ENVIRONMENT", "production").lower()
        if env == "production":
            db_url = os.getenv("DATABASE_URL", "")
            if db_url.startswith("sqlite") or not db_url:
                logger.warning(
                    "Using SQLite in production is not recommended for concurrent access"
                )
            gid = os.getenv("GOOGLE_CLIENT_ID", "")
            if not gid or gid.startswith("YOUR_"):
                logger.warning(
                    "Google OAuth is not configured (GOOGLE_CLIENT_ID missing)"
                )

    # Background analysis thread pool (bounded concurrency, graceful shutdown)
    from concurrent.futures import ThreadPoolExecutor

    application.state.analysis_executor = ThreadPoolExecutor(
        max_workers=2, thread_name_prefix="analysis"
    )

    @application.on_event("shutdown")
    def shutdown_executor():
        executor = getattr(application.state, "analysis_executor", None)
        if executor:
            executor.shutdown(wait=False, cancel_futures=True)
            # Replace with fresh executor in case app is reused (e.g. tests)
            application.state.analysis_executor = ThreadPoolExecutor(
                max_workers=2, thread_name_prefix="analysis"
            )

    # Routers
    application.include_router(auth_router, prefix="/api/v1")
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
