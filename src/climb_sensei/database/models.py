"""Database models for ClimbSensei platform.

This module defines SQLAlchemy ORM models for users, videos,
analyses, and climbing sessions.
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    JSON,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from fastapi_users.db import SQLAlchemyBaseUserTable

from climb_sensei.types import VideoStatus


class Base(DeclarativeBase):
    """Base class for all models."""

    pass


def utcnow():
    """Get current UTC time as timezone-aware datetime."""
    return datetime.now(timezone.utc)


class User(SQLAlchemyBaseUserTable[int], Base):
    """User account model.

    Represents a registered user with authentication credentials
    and metadata. Compatible with fastapi-users.
    """

    __tablename__ = "users"

    # Primary key (required for fastapi-users with int ID)
    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Additional fields beyond fastapi-users base
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    # Relationships
    videos: Mapped[list["Video"]] = relationship(
        "Video", back_populates="user", cascade="all, delete-orphan"
    )
    sessions: Mapped[list["ClimbSession"]] = relationship(
        "ClimbSession", back_populates="user", cascade="all, delete-orphan"
    )
    progress_metrics: Mapped[list["ProgressMetric"]] = relationship(
        "ProgressMetric", back_populates="user", cascade="all, delete-orphan"
    )
    goals: Mapped[list["Goal"]] = relationship(
        "Goal", back_populates="user", cascade="all, delete-orphan"
    )
    routes: Mapped[list["Route"]] = relationship(
        "Route", back_populates="user", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}')>"


class Video(Base):
    """Video file model.

    Represents an uploaded climbing video with metadata and processing status.
    """

    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    uploaded_at = Column(DateTime, default=utcnow, nullable=False)

    # Video properties
    duration_seconds = Column(Float)
    fps = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    file_size_bytes = Column(Integer)

    # Processing status
    status = Column(
        String(50), default=VideoStatus.UPLOADED, nullable=False
    )  # uploaded, processing, completed, failed

    # Relationships
    user = relationship("User", back_populates="videos")
    analyses = relationship(
        "Analysis", back_populates="video", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return (
            f"<Video(id={self.id}, filename='{self.filename}', status='{self.status}')>"
        )


class Route(Base):
    """Climbing route model."""

    __tablename__ = "routes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    grade = Column(String(20), nullable=False)
    grade_system = Column(String(20), nullable=False)  # hueco, font, yds, french
    type = Column(String(20), nullable=False)  # boulder, sport, trad
    location = Column(String(255))
    status = Column(
        String(20), default="projecting", nullable=False
    )  # projecting, sent
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="routes")
    attempts = relationship(
        "Attempt", back_populates="route", cascade="all, delete-orphan"
    )
    goals = relationship("Goal", back_populates="route")

    def __repr__(self) -> str:
        return f"<Route(id={self.id}, name='{self.name}', grade='{self.grade}', status='{self.status}')>"


class Attempt(Base):
    """Climbing attempt model."""

    __tablename__ = "attempts"

    id = Column(Integer, primary_key=True, index=True)
    route_id = Column(Integer, ForeignKey("routes.id"), nullable=False, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("climb_sessions.id"), index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), index=True)
    notes = Column(Text)
    date = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    # Relationships
    route = relationship("Route", back_populates="attempts")
    video = relationship("Video")
    session = relationship("ClimbSession", back_populates="attempts")
    analysis = relationship("Analysis")

    def __repr__(self) -> str:
        return f"<Attempt(id={self.id}, route_id={self.route_id}, date={self.date})>"


class Analysis(Base):
    """Analysis result model.

    Stores the complete analysis results for a video, including
    summary statistics, history data, and quality reports.
    """

    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False, index=True)
    session_id = Column(Integer, ForeignKey("climb_sessions.id"), index=True)
    created_at = Column(DateTime, default=utcnow, nullable=False)

    # Analysis configuration
    run_metrics = Column(Boolean, default=True, nullable=False)
    run_video = Column(Boolean, default=False, nullable=False)
    run_quality = Column(Boolean, default=True, nullable=False)
    dashboard_position = Column(String(20), default="right")

    # Store analysis results as JSON
    summary = Column(JSON)  # ClimbingSummary.to_dict()
    history = Column(JSON)  # Full metrics history
    video_quality = Column(JSON)  # VideoQualityReport
    tracking_quality = Column(JSON)  # TrackingQualityReport

    # Denormalized key metrics for efficient querying
    total_frames = Column(Integer)
    avg_velocity = Column(Float)
    max_velocity = Column(Float)
    max_height = Column(Float)
    total_vertical_progress = Column(Float)
    avg_sway = Column(Float)
    avg_movement_economy = Column(Float)
    lock_off_count = Column(Integer)
    rest_count = Column(Integer)
    fatigue_score = Column(Float)

    # Output files (if generated)
    output_video_path = Column(String(500))
    output_json_path = Column(String(500))

    # Relationships
    video = relationship("Video", back_populates="analyses")
    session = relationship("ClimbSession", back_populates="analyses")
    progress_metrics = relationship(
        "ProgressMetric", back_populates="analysis", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Analysis(id={self.id}, video_id={self.video_id}, frames={self.total_frames})>"


class ClimbSession(Base):
    """Climbing session model.

    Represents a climbing session where users can group multiple
    videos and track their performance.
    """

    __tablename__ = "climb_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=True)
    date = Column(DateTime, nullable=False)
    location = Column(String(255))
    notes = Column(Text)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Session statistics (optional, can be computed from videos)
    total_videos = Column(Integer, default=0)
    avg_performance_score = Column(Float)

    # Relationships
    user = relationship("User", back_populates="sessions")
    analyses = relationship("Analysis", back_populates="session")
    attempts = relationship("Attempt", back_populates="session")

    def __repr__(self) -> str:
        return f"<ClimbSession(id={self.id}, name='{self.name}', date={self.date})>"


class ProgressMetric(Base):
    """Progress metric model.

    Records individual metrics from analyses to enable progress
    tracking over time.
    """

    __tablename__ = "progress_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"), nullable=False, index=True)
    metric_name = Column(String(100), nullable=False, index=True)
    value = Column(Float, nullable=False)
    recorded_at = Column(DateTime, default=utcnow, nullable=False, index=True)

    # Relationships
    user = relationship("User", back_populates="progress_metrics")
    analysis = relationship("Analysis", back_populates="progress_metrics")

    def __repr__(self) -> str:
        return f"<ProgressMetric(id={self.id}, metric='{self.metric_name}', value={self.value})>"


class Goal(Base):
    """Goal model.

    Represents a user's training goal with target values and deadlines.
    """

    __tablename__ = "goals"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    route_id = Column(
        Integer, ForeignKey("routes.id"), index=True
    )  # nullable for migration
    metric_name = Column(String(100), nullable=False)
    target_value = Column(Float, nullable=False)
    current_value = Column(Float)
    deadline = Column(DateTime)
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)
    achieved = Column(Boolean, default=False, nullable=False)
    achieved_at = Column(DateTime)
    notes = Column(Text)

    # Relationships
    user = relationship("User", back_populates="goals")
    route = relationship("Route", back_populates="goals")

    def __repr__(self) -> str:
        return f"<Goal(id={self.id}, metric='{self.metric_name}', target={self.target_value}, achieved={self.achieved})>"
