"""Page routes serving HTML templates."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, RedirectResponse
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
    from app.services.upload import MAX_UPLOAD_SIZE

    max_upload_mb = MAX_UPLOAD_SIZE // (1024 * 1024)
    return templates.TemplateResponse(
        "upload.html",
        {"request": request, "active_tab": "upload", "max_upload_mb": max_upload_mb},
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


@router.get("/dashboard")
async def dashboard_redirect():
    """Redirect legacy dashboard URL to home."""
    return RedirectResponse(url="/", status_code=301)


@router.get("/sessions", response_class=HTMLResponse)
async def sessions_page(request: Request):
    """Render the sessions page."""
    return templates.TemplateResponse(
        "sessions.html", {"request": request, "active_tab": "sessions"}
    )


@router.get("/goals")
async def goals_redirect():
    """Redirect legacy goals URL to profile."""
    return RedirectResponse(url="/profile", status_code=301)


@router.get("/progress")
async def progress_redirect():
    """Redirect legacy progress URL to home."""
    return RedirectResponse(url="/", status_code=301)


@router.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    """Render the profile page."""
    return templates.TemplateResponse(
        "profile.html", {"request": request, "active_tab": "profile"}
    )
