"""Tests for authentication system.

This module tests user registration, login, JWT tokens,
and authentication dependencies.
"""

import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from climb_sensei.database.models import Base, User
from climb_sensei.database.config import get_db
from climb_sensei.auth import (
    get_password_hash,
    verify_password,
    create_access_token,
    authenticate_user,
)
from climb_sensei.auth.routes import router as auth_router


@pytest.fixture
def test_engine():
    """Create a test database engine."""
    # Enable thread-safe mode for SQLite
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def db_session(test_engine):
    """Create a test database session."""
    Session = sessionmaker(bind=test_engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture
def app(db_session):
    """Create a test FastAPI application."""
    app = FastAPI()
    app.include_router(auth_router)

    # Override get_db dependency
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    return TestClient(app)


@pytest.fixture
def test_user(db_session):
    """Create a test user."""
    user = User(
        email="test@example.com",
        hashed_password=get_password_hash("testpassword123"),
        full_name="Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


class TestPasswordHashing:
    """Tests for password hashing utilities."""

    def test_hash_password(self):
        """Should hash a password."""
        password = "mysecretpassword"
        hashed = get_password_hash(password)

        assert hashed != password
        assert len(hashed) > 0

    def test_verify_correct_password(self):
        """Should verify a correct password."""
        password = "mysecretpassword"
        hashed = get_password_hash(password)

        assert verify_password(password, hashed) is True

    def test_verify_incorrect_password(self):
        """Should reject an incorrect password."""
        password = "mysecretpassword"
        wrong_password = "wrongpassword"
        hashed = get_password_hash(password)

        assert verify_password(wrong_password, hashed) is False

    def test_different_hashes_for_same_password(self):
        """Should create different hashes for the same password (salt)."""
        password = "mysecretpassword"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)

        assert hash1 != hash2
        assert verify_password(password, hash1) is True
        assert verify_password(password, hash2) is True


class TestJWTTokens:
    """Tests for JWT token creation and validation."""

    def test_create_access_token(self):
        """Should create a JWT token."""
        data = {"sub": "user@example.com"}
        token = create_access_token(data)

        assert isinstance(token, str)
        assert len(token) > 0

    def test_token_contains_user_email(self):
        """Token should contain user email in payload."""
        from jose import jwt
        from climb_sensei.auth import SECRET_KEY, ALGORITHM

        email = "user@example.com"
        token = create_access_token({"sub": email})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert payload["sub"] == email

    def test_token_has_expiration(self):
        """Token should have an expiration time."""
        from jose import jwt
        from climb_sensei.auth import SECRET_KEY, ALGORITHM

        token = create_access_token({"sub": "user@example.com"})
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])

        assert "exp" in payload
        assert payload["exp"] > 0


class TestUserAuthentication:
    """Tests for user authentication."""

    def test_authenticate_valid_user(self, db_session, test_user):
        """Should authenticate user with correct credentials."""
        user = authenticate_user(db_session, "test@example.com", "testpassword123")

        assert user is not None
        assert user.email == "test@example.com"

    def test_authenticate_wrong_password(self, db_session, test_user):
        """Should reject user with wrong password."""
        user = authenticate_user(db_session, "test@example.com", "wrongpassword")

        assert user is None

    def test_authenticate_nonexistent_user(self, db_session):
        """Should reject nonexistent user."""
        user = authenticate_user(db_session, "nonexistent@example.com", "password")

        assert user is None


class TestRegistrationEndpoint:
    """Tests for user registration endpoint."""

    def test_register_new_user(self, client):
        """Should register a new user."""
        response = client.post(
            "/auth/register",
            json={
                "email": "newuser@example.com",
                "password": "securepassword123",
                "full_name": "New User",
            },
        )

        assert response.status_code == 201
        data = response.json()
        assert data["email"] == "newuser@example.com"
        assert data["full_name"] == "New User"
        assert "password" not in data
        assert "password_hash" not in data
        assert data["is_active"] is True

    def test_register_duplicate_email(self, client, test_user):
        """Should reject registration with duplicate email."""
        response = client.post(
            "/auth/register",
            json={
                "email": "test@example.com",
                "password": "anotherpassword123",
                "full_name": "Another User",
            },
        )

        assert response.status_code == 400
        assert "registration failed" in response.json()["detail"].lower()

    def test_register_invalid_email(self, client):
        """Should reject invalid email format."""
        response = client.post(
            "/auth/register",
            json={
                "email": "not-an-email",
                "password": "password123",
                "full_name": "Test",
            },
        )

        assert response.status_code == 422

    def test_register_short_password(self, client):
        """Should reject password shorter than 8 characters."""
        response = client.post(
            "/auth/register",
            json={
                "email": "user@example.com",
                "password": "short",
                "full_name": "Test",
            },
        )

        assert response.status_code == 422


class TestLoginEndpoint:
    """Tests for user login endpoint."""

    def test_login_valid_credentials(self, client, test_user):
        """Should login with valid credentials."""
        response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "testpassword123"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert isinstance(data["access_token"], str)

    def test_login_wrong_password(self, client, test_user):
        """Should reject login with wrong password."""
        response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "wrongpassword"},
        )

        assert response.status_code == 401
        assert "incorrect" in response.json()["detail"].lower()

    def test_login_nonexistent_user(self, client):
        """Should reject login for nonexistent user."""
        response = client.post(
            "/auth/login",
            json={"email": "nonexistent@example.com", "password": "password123"},
        )

        assert response.status_code == 401

    def test_login_inactive_user(self, client, db_session):
        """Should reject login for inactive user."""
        inactive_user = User(
            email="inactive@example.com",
            hashed_password=get_password_hash("password123"),
            is_active=False,
        )
        db_session.add(inactive_user)
        db_session.commit()

        response = client.post(
            "/auth/login",
            json={"email": "inactive@example.com", "password": "password123"},
        )

        assert response.status_code == 403
        assert "inactive" in response.json()["detail"].lower()


class TestGetCurrentUserEndpoint:
    """Tests for getting current user information."""

    def test_get_current_user_with_valid_token(self, client, test_user):
        """Should get user info with valid token."""
        # Login to get token
        login_response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        # Get current user
        response = client.get("/auth/me", headers={"Authorization": f"Bearer {token}"})

        assert response.status_code == 200
        data = response.json()
        assert data["email"] == "test@example.com"
        assert data["full_name"] == "Test User"

    def test_get_current_user_without_token(self, client):
        """Should reject request without token."""
        response = client.get("/auth/me")

        assert response.status_code == 401

    def test_get_current_user_with_invalid_token(self, client):
        """Should reject request with invalid token."""
        response = client.get(
            "/auth/me", headers={"Authorization": "Bearer invalid_token"}
        )

        assert response.status_code == 401


class TestUpdateUserEndpoint:
    """Tests for updating user information."""

    def test_update_full_name(self, client, test_user):
        """Should update user's full name."""
        # Login
        login_response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        # Update
        response = client.patch(
            "/auth/me",
            json={"full_name": "Updated Name"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["full_name"] == "Updated Name"

    def test_update_email(self, client, test_user):
        """Should update user's email."""
        # Login
        login_response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        # Update
        response = client.patch(
            "/auth/me",
            json={"email": "newemail@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 200
        assert response.json()["email"] == "newemail@example.com"

    def test_update_email_to_existing(self, client, test_user, db_session):
        """Should reject email update to existing email."""
        # Create another user
        another_user = User(
            email="another@example.com",
            hashed_password=get_password_hash("password123"),
        )
        db_session.add(another_user)
        db_session.commit()

        # Login as test_user
        login_response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        # Try to update to existing email
        response = client.patch(
            "/auth/me",
            json={"email": "another@example.com"},
            headers={"Authorization": f"Bearer {token}"},
        )

        assert response.status_code == 400


class TestSuperuserEndpoints:
    """Tests for superuser-only endpoints."""

    def test_list_users_as_superuser(self, client, db_session):
        """Superuser should be able to list all users."""
        # Create superuser
        superuser = User(
            email="admin@example.com",
            hashed_password=get_password_hash("adminpassword123"),
            is_superuser=True,
        )
        db_session.add(superuser)
        db_session.commit()

        # Login as superuser
        login_response = client.post(
            "/auth/login",
            json={"email": "admin@example.com", "password": "adminpassword123"},
        )
        token = login_response.json()["access_token"]

        # List users
        response = client.get(
            "/auth/users", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 200
        users = response.json()
        assert isinstance(users, list)
        assert len(users) > 0

    def test_list_users_as_regular_user(self, client, test_user):
        """Regular user should not be able to list users."""
        # Login as regular user
        login_response = client.post(
            "/auth/login",
            json={"email": "test@example.com", "password": "testpassword123"},
        )
        token = login_response.json()["access_token"]

        # Try to list users
        response = client.get(
            "/auth/users", headers={"Authorization": f"Bearer {token}"}
        )

        assert response.status_code == 403
