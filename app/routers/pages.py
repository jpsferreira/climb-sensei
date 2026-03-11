"""Page routes serving HTML templates."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from app.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse("upload.html", {"request": request})


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the login page."""
    return templates.TemplateResponse("login.html", {"request": request})


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Render the registration page."""
    return templates.TemplateResponse("register.html", {"request": request})


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Render the dashboard page."""
    return templates.TemplateResponse("dashboard.html", {"request": request})


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request):
    """Render the sessions page."""
    return templates.TemplateResponse("sessions.html", {"request": request})


@router.get("/goals", response_class=HTMLResponse)
async def goals_page(request: Request):
    """Render the goals page."""
    return templates.TemplateResponse("goals.html", {"request": request})


@router.get("/progress", response_class=HTMLResponse)
async def progress_page(request: Request):
    """Render the progress tracking page."""
    return templates.TemplateResponse("progress.html", {"request": request})
