"""Attempt endpoints for route-specific attempt tracking.

Provides:
- GET /api/routes/{route_id}/attempts          — list attempts (most recent first)
- GET /api/routes/{route_id}/attempts/{id}     — full detail with comparison deltas
- PATCH /api/routes/{route_id}/attempts/{id}   — update attempt notes
- DELETE /api/routes/{route_id}/attempts/{id}  — delete attempt record only
- GET /api/routes/{route_id}/progress/{metric} — metric trend for a route
"""

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..database.config import get_db
from ..database.models import User, Route, Attempt, Analysis
from ..auth import get_current_active_user
from .route_schemas import AttemptResponse, AttemptDetailResponse


class AttemptUpdate(BaseModel):
    """Schema for updating attempt fields."""

    notes: str | None = None


router = APIRouter(prefix="/api/v1/routes", tags=["attempts"])

ALLOWED_METRICS = [
    "avg_velocity",
    "max_velocity",
    "max_height",
    "total_vertical_progress",
    "avg_sway",
    "avg_movement_economy",
    "lock_off_count",
    "rest_count",
    "fatigue_score",
]


def _get_owned_route(route_id: int, user_id: int, db: Session) -> Route:
    """Return the route if it belongs to the user, else raise 404."""
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == user_id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")
    return route


def _build_attempt_response(attempt: Attempt) -> AttemptResponse:
    """Build an AttemptResponse from an Attempt ORM object."""
    analysis = attempt.analysis
    return AttemptResponse(
        id=attempt.id,
        route_id=attempt.route_id,
        video_id=attempt.video_id,
        session_id=attempt.session_id,
        analysis_id=attempt.analysis_id,
        notes=attempt.notes,
        date=attempt.date,
        created_at=attempt.created_at,
        avg_velocity=analysis.avg_velocity if analysis else None,
        avg_sway=analysis.avg_sway if analysis else None,
        avg_movement_economy=analysis.avg_movement_economy if analysis else None,
        has_video=attempt.video is not None,
        video_filename=attempt.video.filename if attempt.video else None,
    )


def _extract_metric(analysis: Analysis, metric: str) -> float | None:
    """Extract a scalar metric value from an Analysis record."""
    return getattr(analysis, metric, None)


# ========== List Attempts ==========


@router.get("/{route_id}/attempts", response_model=list[AttemptResponse])
async def list_attempts(
    route_id: int,
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all attempts for a route, most recent first.

    Args:
        route_id: Route ID
        skip: Pagination offset
        limit: Maximum number of attempts to return
        current_user: Authenticated user
        db: Database session

    Returns:
        List of AttemptResponse with inline metrics
    """
    _get_owned_route(route_id, current_user.id, db)

    attempts = (
        db.query(Attempt)
        .filter(Attempt.route_id == route_id)
        .order_by(Attempt.date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [_build_attempt_response(a) for a in attempts]


# ========== Get Attempt Detail ==========


@router.get("/{route_id}/attempts/{attempt_id}", response_model=AttemptDetailResponse)
async def get_attempt(
    route_id: int,
    attempt_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get full attempt detail including analysis and comparison deltas.

    Comparison deltas are computed against the previous attempt for the same
    route, ordered by date ascending.

    Args:
        route_id: Route ID
        attempt_id: Attempt ID
        current_user: Authenticated user
        db: Database session

    Returns:
        AttemptDetailResponse with optional deltas vs. previous attempt
    """
    _get_owned_route(route_id, current_user.id, db)

    attempt = (
        db.query(Attempt)
        .filter(and_(Attempt.id == attempt_id, Attempt.route_id == route_id))
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    # Find the previous attempt (the one just before this date)
    prev_attempt = (
        db.query(Attempt)
        .filter(
            and_(
                Attempt.route_id == route_id,
                Attempt.id != attempt_id,
                Attempt.date < attempt.date,
            )
        )
        .order_by(Attempt.date.desc())
        .first()
    )

    # Build comparison deltas if both attempts have analyses
    deltas: dict | None = None
    prev_attempt_id: int | None = None
    if prev_attempt:
        prev_attempt_id = prev_attempt.id
        if attempt.analysis and prev_attempt.analysis:
            deltas = {}
            for metric in ALLOWED_METRICS:
                current_val = _extract_metric(attempt.analysis, metric)
                prev_val = _extract_metric(prev_attempt.analysis, metric)
                if current_val is not None and prev_val is not None:
                    deltas[metric] = round(current_val - prev_val, 6)

    analysis = attempt.analysis
    return AttemptDetailResponse(
        id=attempt.id,
        route_id=attempt.route_id,
        video_id=attempt.video_id,
        session_id=attempt.session_id,
        analysis_id=attempt.analysis_id,
        notes=attempt.notes,
        date=attempt.date,
        created_at=attempt.created_at,
        avg_velocity=analysis.avg_velocity if analysis else None,
        max_velocity=analysis.max_velocity if analysis else None,
        max_height=analysis.max_height if analysis else None,
        total_vertical_progress=analysis.total_vertical_progress if analysis else None,
        avg_sway=analysis.avg_sway if analysis else None,
        avg_movement_economy=analysis.avg_movement_economy if analysis else None,
        lock_off_count=analysis.lock_off_count if analysis else None,
        rest_count=analysis.rest_count if analysis else None,
        fatigue_score=analysis.fatigue_score if analysis else None,
        has_video=attempt.video is not None,
        video_filename=attempt.video.filename if attempt.video else None,
        summary=analysis.summary if analysis else None,
        history=analysis.history if analysis else None,
        video_quality=analysis.video_quality if analysis else None,
        tracking_quality=analysis.tracking_quality if analysis else None,
        output_video_path=analysis.output_video_path if analysis else None,
        original_video_url=(
            f"/uploads/{Path(attempt.video.file_path).name}"
            if attempt.video and attempt.video.file_path
            else None
        ),
        prev_attempt_id=prev_attempt_id,
        deltas=deltas,
    )


# ========== Update Attempt Notes ==========


@router.patch("/{route_id}/attempts/{attempt_id}", response_model=AttemptResponse)
async def update_attempt(
    route_id: int,
    attempt_id: int,
    payload: AttemptUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update attempt notes.

    Args:
        route_id: Route ID
        attempt_id: Attempt ID
        payload: JSON body with optional ``notes`` field
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated AttemptResponse
    """
    _get_owned_route(route_id, current_user.id, db)

    attempt = (
        db.query(Attempt)
        .filter(and_(Attempt.id == attempt_id, Attempt.route_id == route_id))
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    if payload.notes is not None:
        attempt.notes = payload.notes
        db.commit()
        db.refresh(attempt)

    return _build_attempt_response(attempt)


# ========== Delete Attempt ==========


@router.delete("/{route_id}/attempts/{attempt_id}", status_code=204)
async def delete_attempt(
    route_id: int,
    attempt_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete an attempt record.

    Deletes only the Attempt row — the associated Video and Analysis records
    are preserved.

    Args:
        route_id: Route ID
        attempt_id: Attempt ID
        current_user: Authenticated user
        db: Database session
    """
    _get_owned_route(route_id, current_user.id, db)

    attempt = (
        db.query(Attempt)
        .filter(and_(Attempt.id == attempt_id, Attempt.route_id == route_id))
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    db.delete(attempt)
    db.commit()


# ========== Metric Trend ==========


@router.get("/{route_id}/progress/{metric}")
async def get_route_metric_trend(
    route_id: int,
    metric: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Return the trend of a metric across all attempts on a route.

    Args:
        route_id: Route ID
        metric: Metric name — must be one of ALLOWED_METRICS
        current_user: Authenticated user
        db: Database session

    Returns:
        JSON with route_id, metric, and a list of {attempt_id, date, value} points
    """
    if metric not in ALLOWED_METRICS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid metric '{metric}'. Allowed: {ALLOWED_METRICS}",
        )

    _get_owned_route(route_id, current_user.id, db)

    attempts = (
        db.query(Attempt)
        .filter(Attempt.route_id == route_id)
        .order_by(Attempt.date.asc())
        .all()
    )

    data = []
    for attempt in attempts:
        if attempt.analysis:
            value = _extract_metric(attempt.analysis, metric)
            if value is not None:
                data.append(
                    {
                        "attempt_id": attempt.id,
                        "date": attempt.date.isoformat(),
                        "value": value,
                    }
                )

    return {
        "route_id": route_id,
        "metric": metric,
        "data": data,
        "count": len(data),
    }
