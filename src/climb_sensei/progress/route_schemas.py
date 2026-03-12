"""Pydantic schemas for Route and Attempt management."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class RouteCreate(BaseModel):
    name: str = Field(..., max_length=255)
    grade: str = Field(..., max_length=20)
    grade_system: str = Field(..., pattern="^(hueco|font|yds|french)$")
    type: str = Field(..., pattern="^(boulder|sport|trad)$")
    location: Optional[str] = Field(None, max_length=255)


class RouteUpdate(BaseModel):
    name: Optional[str] = Field(None, max_length=255)
    grade: Optional[str] = Field(None, max_length=20)
    grade_system: Optional[str] = Field(None, pattern="^(hueco|font|yds|french)$")
    type: Optional[str] = Field(None, pattern="^(boulder|sport|trad)$")
    location: Optional[str] = Field(None, max_length=255)
    status: Optional[str] = Field(None, pattern="^(projecting|sent)$")


class RouteResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    user_id: int
    name: str
    grade: str
    grade_system: str
    type: str
    location: Optional[str] = None
    status: str
    created_at: datetime
    updated_at: datetime
    attempt_count: int = 0
    last_attempt_date: Optional[datetime] = None


class RouteListResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    grade: str
    grade_system: str
    type: str
    location: Optional[str] = None
    status: str
    attempt_count: int = 0
    last_attempt_date: Optional[datetime] = None
    sparkline: list[float] = []


class AttemptCreate(BaseModel):
    route_id: int
    session_id: Optional[int] = None
    notes: Optional[str] = None
    date: Optional[datetime] = None


class AttemptResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    route_id: int
    video_id: int
    session_id: Optional[int] = None
    analysis_id: Optional[int] = None
    notes: Optional[str] = None
    date: datetime
    created_at: datetime
    avg_velocity: Optional[float] = None
    avg_sway: Optional[float] = None
    avg_movement_economy: Optional[float] = None
    has_video: bool = False
    video_filename: Optional[str] = None


class AttemptDetailResponse(AttemptResponse):
    max_velocity: Optional[float] = None
    max_height: Optional[float] = None
    total_vertical_progress: Optional[float] = None
    lock_off_count: Optional[int] = None
    rest_count: Optional[int] = None
    fatigue_score: Optional[float] = None
    summary: Optional[dict] = None
    history: Optional[dict] = None
    video_quality: Optional[dict] = None
    tracking_quality: Optional[dict] = None
    output_video_path: Optional[str] = None
    original_video_url: Optional[str] = None
    prev_attempt_id: Optional[int] = None
    deltas: Optional[dict] = None
