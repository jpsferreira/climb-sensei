"""Page routes serving HTML templates."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from starlette.requests import Request

from app.templating import templates

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the routes landing page."""
    return templates.TemplateResponse(
        "routes.html", {"request": request, "active_tab": "routes"}
    )


@router.get("/routes/{route_id}", response_class=HTMLResponse)
async def route_detail(request: Request, route_id: int):
    """Render the route detail page."""
    return templates.TemplateResponse(
        "route_detail.html",
        {"request": request, "active_tab": "routes", "route_id": route_id},
    )


@router.get("/routes/{route_id}/attempts/{attempt_id}", response_class=HTMLResponse)
async def attempt_detail(request: Request, route_id: int, attempt_id: int):
    """Render the attempt detail page."""
    return templates.TemplateResponse(
        "attempt_detail.html",
        {
            "request": request,
            "active_tab": "routes",
            "route_id": route_id,
            "attempt_id": attempt_id,
        },
    )


@router.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Render the upload page."""
    return templates.TemplateResponse(
        "upload.html", {"request": request, "active_tab": "upload"}
    )


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render the login page."""
    return templates.TemplateResponse(
        "login.html", {"request": request, "hide_nav": True}
    )


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    """Render the registration page."""
    return templates.TemplateResponse(
        "register.html", {"request": request, "hide_nav": True}
    )


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    """Render the dashboard page."""
    return templates.TemplateResponse(
        "dashboard.html", {"request": request, "active_tab": "routes"}
    )


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request):
    """Render the sessions page."""
    return templates.TemplateResponse(
        "sessions.html", {"request": request, "active_tab": "sessions"}
    )


@router.get("/goals", response_class=HTMLResponse)
async def goals_page(request: Request):
    """Render the goals page."""
    return templates.TemplateResponse(
        "goals.html", {"request": request, "active_tab": "routes"}
    )


@router.get("/progress", response_class=HTMLResponse)
async def progress_page(request: Request):
    """Render the progress tracking page."""
    return templates.TemplateResponse(
        "progress.html", {"request": request, "active_tab": "routes"}
    )


@router.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    """Render the profile page."""
    return templates.TemplateResponse(
        "profile.html", {"request": request, "active_tab": "profile"}
    )
