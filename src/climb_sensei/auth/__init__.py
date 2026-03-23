"""Authentication utilities for user management.

This module provides user authentication using fastapi-users.
Also provides sync wrappers for compatibility with sync database code.
"""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from climb_sensei.database.config import get_db
from climb_sensei.database.models import User

# Import new fastapi-users dependencies (re-exported for use by other modules)
from climb_sensei.auth.users import (  # noqa: F401
    current_active_user as async_current_active_user,
    fastapi_users,
)

logger = logging.getLogger(__name__)

# Configuration from environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
if SECRET_KEY is None:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set. "
        'Generate one with: python -c "import secrets; print(secrets.token_urlsafe(32))"'
    )
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))

# Bearer token security scheme for sync endpoints
bearer_scheme = HTTPBearer(auto_error=False)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The hashed password to verify against

    Returns:
        True if the password matches, False otherwise
    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"), hashed_password.encode("utf-8")
    )


def get_password_hash(password: str) -> str:
    """Hash a plain text password.

    Args:
        password: The plain text password to hash

    Returns:
        The hashed password
    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token.

    Args:
        data: The data to encode in the token (should include "sub" with user email)
        expires_delta: Optional custom expiration time

    Returns:
        The encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


_auth_disabled_requested = os.getenv("AUTH_DISABLED", "").lower() in (
    "1",
    "true",
    "yes",
)
_environment = os.getenv("ENVIRONMENT", "production").lower()
AUTH_DISABLED = _auth_disabled_requested and _environment in ("development", "testing")
if _auth_disabled_requested and not AUTH_DISABLED:
    logger.critical(
        "AUTH_DISABLED ignored because ENVIRONMENT=%s. "
        "AUTH_DISABLED only works in development/testing environments.",
        _environment,
    )


def _get_or_create_dev_user(db: Session) -> User:
    """Get or create a dev user for AUTH_DISABLED mode."""
    user = db.query(User).filter(User.email == "dev@localhost").first()
    if user is None:
        user = User(
            email="dev@localhost",
            hashed_password=get_password_hash("dev"),
            full_name="Dev User",
            is_active=True,
            is_superuser=True,
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        logger.info("Created dev user (id=%s)", user.id)
    return user


def get_current_active_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """Get the current active user (SYNC version for compatibility).

    This validates JWT tokens issued by fastapi-users but works with
    sync database sessions for compatibility with existing endpoints.

    When AUTH_DISABLED=1 env var is set, returns a dev user without
    requiring authentication (dev/testing only).

    Args:
        credentials: Bearer token credentials
        db: Sync database session

    Returns:
        The authenticated User object

    Raises:
        HTTPException: 401 if credentials are invalid
    """
    if AUTH_DISABLED:
        return _get_or_create_dev_user(db)

    if not credentials:
        logger.debug("No credentials provided")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    logger.debug("Received token (length=%d)", len(token))

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Decode JWT token (same SECRET_KEY as fastapi-users)
        # Don't verify audience claim (fastapi-users sets it but we don't need it)
        payload = jwt.decode(
            token,
            SECRET_KEY,
            algorithms=[ALGORITHM],
            audience=None,  # Don't verify audience
            options={"verify_aud": False},  # Also disable in options
        )
        logger.debug("JWT payload decoded successfully")

        user_id: str = payload.get("sub")
        if user_id is None:
            logger.debug("No 'sub' claim in payload")
            raise credentials_exception

        logger.debug("User ID from token: %s", user_id)

        # Convert to int (fastapi-users uses string user IDs in JWT)
        user_id = int(user_id)
    except (JWTError, ValueError) as e:
        logger.debug("Token decode error: %s", e)
        raise credentials_exception

    # Query user from sync database
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        logger.debug("User %s not found in database", user_id)
        raise credentials_exception

    logger.debug("Found user id=%s, is_active=%s", user_id, user.is_active)

    # Check if user is active
    if not user.is_active:
        logger.debug("User id=%s is not active", user_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account",
        )

    logger.debug("Successfully authenticated user id=%s", user_id)
    return user


def get_current_superuser(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """Get the current superuser.

    This dependency ensures the user is a superuser.

    Args:
        current_user: The current user from get_current_active_user

    Returns:
        The superuser User object

    Raises:
        HTTPException: 403 if user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions",
        )
    return current_user


def authenticate_user(db: Session, email: str, password: str) -> Optional[User]:
    """Authenticate a user by email and password.

    Args:
        db: Database session
        email: User's email address
        password: Plain text password

    Returns:
        User object if authentication successful, None otherwise
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    # Use hashed_password (fastapi-users convention)
    if not verify_password(password, user.hashed_password):
        return None
    return user
