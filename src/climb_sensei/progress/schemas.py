"""Pydantic schemas for progress tracking and goals.

This module defines request/response schemas for progress metrics,
goals, and session management.
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ========== ProgressMetric Schemas ==========


class ProgressMetricBase(BaseModel):
    """Base schema for progress metrics."""

    metric_name: str = Field(..., max_length=100)
    value: float


class ProgressMetricCreate(ProgressMetricBase):
    """Schema for creating a progress metric."""

    analysis_id: int


class ProgressMetricResponse(ProgressMetricBase):
    """Schema for progress metric responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    analysis_id: int
    recorded_at: datetime


class ProgressDataPoint(BaseModel):
    """Schema for a single data point in progress history."""

    date: datetime
    value: float
    analysis_id: int


class ProgressHistory(BaseModel):
    """Schema for metric progress over time."""

    metric: str
    data: list[ProgressDataPoint]
    count: int
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    avg_value: Optional[float] = None


# ========== Goal Schemas ==========


class GoalBase(BaseModel):
    """Base schema for goals."""

    metric_name: str = Field(..., max_length=100)
    target_value: float
    deadline: Optional[datetime] = None
    notes: Optional[str] = None


class GoalCreate(GoalBase):
    """Schema for creating a goal."""

    route_id: Optional[int] = None


class GoalUpdate(BaseModel):
    """Schema for updating a goal."""

    target_value: Optional[float] = None
    current_value: Optional[float] = None
    deadline: Optional[datetime] = None
    notes: Optional[str] = None
    achieved: Optional[bool] = None


class GoalResponse(GoalBase):
    """Schema for goal responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    route_id: Optional[int] = None
    current_value: Optional[float] = None
    created_at: datetime
    updated_at: datetime
    achieved: bool
    achieved_at: Optional[datetime] = None
    progress_percentage: Optional[float] = None  # Computed field


# ========== ClimbSession Schemas ==========


class ClimbSessionBase(BaseModel):
    """Base schema for climbing sessions."""

    name: Optional[str] = Field(None, max_length=255)
    date: datetime
    location: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None


class ClimbSessionCreate(ClimbSessionBase):
    """Schema for creating a climbing session."""

    pass


class ClimbSessionUpdate(BaseModel):
    """Schema for updating a climbing session."""

    name: Optional[str] = Field(None, max_length=255)
    date: Optional[datetime] = None
    location: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None


class ClimbSessionResponse(ClimbSessionBase):
    """Schema for climbing session responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    user_id: int
    created_at: datetime
    updated_at: datetime
    total_videos: int
    avg_performance_score: Optional[float] = None


# ========== Comparison Schemas ==========


class AnalysisComparison(BaseModel):
    """Schema for comparing multiple analyses."""

    id: int
    date: datetime
    session_id: Optional[int] = None
    session_name: Optional[str] = None

    # Key metrics for comparison
    total_frames: Optional[int] = None
    avg_velocity: Optional[float] = None
    max_velocity: Optional[float] = None
    max_height: Optional[float] = None
    total_vertical_progress: Optional[float] = None
    avg_sway: Optional[float] = None
    avg_movement_economy: Optional[float] = None
    lock_off_count: Optional[int] = None
    rest_count: Optional[int] = None
    fatigue_score: Optional[float] = None


class ComparisonRequest(BaseModel):
    """Request schema for comparing analyses."""

    analysis_ids: list[int] = Field(..., min_length=2, max_length=10)


class ComparisonResponse(BaseModel):
    """Response schema for analysis comparisons."""

    comparisons: list[AnalysisComparison]
    count: int
