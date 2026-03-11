"""FastAPI web application for ClimbingSensei video analysis.

Slim app factory: creates the FastAPI instance, mounts routers,
configures static files and middleware.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn

from climb_sensei.auth.routes_new import router as auth_router
from climb_sensei.database.config import init_db
from climb_sensei.progress.routes import router as progress_router

from app.routers.api import router as api_router
from app.routers.pages import router as pages_router
from app.services.upload import OUTPUT_DIR

BASE_DIR = Path(__file__).parent

app = FastAPI(
    title="ClimbingSensei Web App",
    description="Upload climbing videos and analyze performance",
    version="1.0.0",
)

# Initialize database
init_db()

# Routers
app.include_router(auth_router, prefix="/api")
app.include_router(progress_router)
app.include_router(api_router)
app.include_router(pages_router)

# Static files
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("ClimbingSensei Web App")
    print("=" * 60)
    print("\nStarting server at http://localhost:8000")
    print("Press CTRL+C to stop\n")

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
