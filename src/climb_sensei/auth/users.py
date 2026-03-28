"""FastAPI-Users configuration for authentication and OAuth.

Sets up user management, authentication backend, and OAuth providers.
"""

import logging
import os
from typing import Optional
from fastapi import Depends, Request
from fastapi_users import BaseUserManager, FastAPIUsers, IntegerIDMixin
from fastapi_users.authentication import (
    AuthenticationBackend,
    BearerTransport,
    JWTStrategy,
)
from fastapi_users_db_sqlalchemy import SQLAlchemyUserDatabase
from httpx_oauth.clients.google import GoogleOAuth2
from sqlalchemy.ext.asyncio import AsyncSession

from climb_sensei.database.models import User
from climb_sensei.database.config import get_async_db

logger = logging.getLogger(__name__)

# Environment variables for OAuth
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
SECRET_KEY = os.getenv("SECRET_KEY", "").strip()
if not SECRET_KEY or len(SECRET_KEY) < 32:
    raise RuntimeError(
        "SECRET_KEY environment variable is missing or too short (min 32 chars). "
        'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
    )

# Google OAuth client
google_oauth_client = GoogleOAuth2(
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
)


class UserManager(IntegerIDMixin, BaseUserManager[User, int]):
    """User manager for fastapi-users."""

    reset_password_token_secret = SECRET_KEY
    verification_token_secret = SECRET_KEY

    async def on_after_register(self, user: User, request: Optional[Request] = None):
        """Hook called after user registration."""
        logger.info("User %s has registered.", user.id)

    async def on_after_forgot_password(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Hook called after password reset request."""
        logger.info("User %s requested a password reset.", user.id)

    async def on_after_request_verify(
        self, user: User, token: str, request: Optional[Request] = None
    ):
        """Hook called after email verification request."""
        logger.info("Verification requested for user %s.", user.id)


async def get_user_db(session: AsyncSession = Depends(get_async_db)):
    """Get user database adapter."""
    yield SQLAlchemyUserDatabase(session, User)


async def get_user_manager(user_db: SQLAlchemyUserDatabase = Depends(get_user_db)):
    """Get user manager instance."""
    yield UserManager(user_db)


# JWT authentication
bearer_transport = BearerTransport(tokenUrl="api/auth/jwt/login")


def get_jwt_strategy() -> JWTStrategy:
    """Get JWT strategy for token generation.

    Uses the same SECRET_KEY as auth/__init__.py's create_access_token
    so tokens from both sources are interchangeable.

    Derives lifetime from ACCESS_TOKEN_EXPIRE_MINUTES for consistency
    with create_access_token's default expiry.
    """
    from climb_sensei.auth import ACCESS_TOKEN_EXPIRE_MINUTES

    return JWTStrategy(
        secret=SECRET_KEY, lifetime_seconds=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


auth_backend = AuthenticationBackend(
    name="jwt",
    transport=bearer_transport,
    get_strategy=get_jwt_strategy,
)

# FastAPI Users instance
fastapi_users = FastAPIUsers[User, int](
    get_user_manager,
    [auth_backend],
)

# Dependency to get current active user
current_active_user = fastapi_users.current_user(active=True)
