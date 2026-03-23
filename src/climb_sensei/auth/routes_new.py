"""Authentication API routes using fastapi-users.

Provides endpoints for registration, login, and Google OAuth.
"""

import os

from fastapi import APIRouter, Depends
from fastapi_users import schemas

from climb_sensei.auth.users import (
    SECRET_KEY,
    auth_backend,
    fastapi_users,
    google_oauth_client,
)

# Create router
router = APIRouter(prefix="/auth", tags=["Authentication"])


# User schemas
class UserRead(schemas.BaseUser[int]):
    """User response schema."""

    full_name: str | None = None


class UserCreate(schemas.BaseUserCreate):
    """User creation schema."""

    full_name: str | None = None


class UserUpdate(schemas.BaseUserUpdate):
    """User update schema."""

    full_name: str | None = None


# Include fastapi-users routers
router.include_router(
    fastapi_users.get_auth_router(auth_backend),
    prefix="/jwt",
)

router.include_router(
    fastapi_users.get_register_router(UserRead, UserCreate),
)

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
)

# Google OAuth router
router.include_router(
    fastapi_users.get_oauth_router(
        google_oauth_client,
        auth_backend,
        os.getenv("OAUTH_STATE_SECRET", SECRET_KEY),
        redirect_url=os.getenv(
            "OAUTH_REDIRECT_URL",
            "http://localhost:8000/api/auth/google/callback",
        ),
    ),
    prefix="/google",
    tags=["OAuth"],
)


# Keep backward compatibility endpoint for /me
@router.get("/me", response_model=UserRead)
async def get_current_user(user=Depends(fastapi_users.current_user(active=True))):
    """Get current authenticated user."""
    return user
