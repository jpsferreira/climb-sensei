"""Progress tracking and goal management routes.

This module provides endpoints for:
- Progress metric tracking over time
- Goal setting and tracking
- Analysis comparisons
- Climbing session management
"""

from datetime import datetime, timedelta
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..database.config import get_db
from ..database.models import User, Analysis, ProgressMetric, Goal, ClimbSession, Video
from ..auth import get_current_active_user
from .schemas import (
    ProgressHistory,
    ProgressDataPoint,
    GoalCreate,
    GoalUpdate,
    GoalResponse,
    ClimbSessionCreate,
    ClimbSessionUpdate,
    ClimbSessionResponse,
    ComparisonRequest,
    ComparisonResponse,
    AnalysisComparison,
)

router = APIRouter(prefix="/api", tags=["progress"])


# ========== Progress Metrics Endpoints ==========


@router.get("/progress/{metric_name}", response_model=ProgressHistory)
async def get_metric_progress(
    metric_name: str,
    days: int = Query(30, ge=1, le=365, description="Number of days to retrieve"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get progress for a specific metric over time.

    Args:
        metric_name: Name of the metric (e.g., avg_velocity, lock_off_count)
        days: Number of days to look back (default: 30)
        current_user: Authenticated user
        db: Database session

    Returns:
        ProgressHistory with data points, statistics
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    metrics = (
        db.query(ProgressMetric)
        .filter(
            and_(
                ProgressMetric.user_id == current_user.id,
                ProgressMetric.metric_name == metric_name,
                ProgressMetric.recorded_at >= cutoff,
            )
        )
        .order_by(ProgressMetric.recorded_at)
        .all()
    )

    if not metrics:
        return ProgressHistory(metric=metric_name, data=[], count=0)

    data_points = [
        ProgressDataPoint(date=m.recorded_at, value=m.value, analysis_id=m.analysis_id)
        for m in metrics
    ]

    values = [m.value for m in metrics]

    return ProgressHistory(
        metric=metric_name,
        data=data_points,
        count=len(metrics),
        min_value=min(values),
        max_value=max(values),
        avg_value=sum(values) / len(values),
    )


@router.get("/progress", response_model=dict)
async def get_all_metrics_progress(
    days: int = Query(30, ge=1, le=365),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get progress for all tracked metrics.

    Returns a dictionary with metric names as keys and their progress data.
    """
    cutoff = datetime.utcnow() - timedelta(days=days)

    metrics = (
        db.query(ProgressMetric)
        .filter(
            and_(
                ProgressMetric.user_id == current_user.id,
                ProgressMetric.recorded_at >= cutoff,
            )
        )
        .order_by(ProgressMetric.recorded_at)
        .all()
    )

    # Group by metric name
    grouped = {}
    for metric in metrics:
        if metric.metric_name not in grouped:
            grouped[metric.metric_name] = []
        grouped[metric.metric_name].append(metric)

    # Build response
    result = {}
    for metric_name, metric_list in grouped.items():
        data_points = [
            ProgressDataPoint(
                date=m.recorded_at, value=m.value, analysis_id=m.analysis_id
            )
            for m in metric_list
        ]
        values = [m.value for m in metric_list]

        result[metric_name] = {
            "data": [dp.model_dump() for dp in data_points],
            "count": len(metric_list),
            "min_value": min(values),
            "max_value": max(values),
            "avg_value": sum(values) / len(values),
        }

    return result


# ========== Analysis Comparison Endpoints ==========


@router.post("/compare", response_model=ComparisonResponse)
async def compare_analyses(
    request: ComparisonRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Compare multiple analyses side by side.

    Args:
        request: List of analysis IDs to compare (2-10)
        current_user: Authenticated user
        db: Database session

    Returns:
        Comparison data for all analyses
    """
    # Fetch analyses with user verification
    analyses = (
        db.query(Analysis)
        .join(Video)
        .filter(
            and_(
                Analysis.id.in_(request.analysis_ids), Video.user_id == current_user.id
            )
        )
        .all()
    )

    if len(analyses) != len(request.analysis_ids):
        raise HTTPException(
            status_code=404, detail="Some analyses not found or don't belong to you"
        )

    comparisons = []
    for analysis in analyses:
        # Get session name if exists
        session_name = None
        if analysis.session_id:
            session = (
                db.query(ClimbSession)
                .filter(ClimbSession.id == analysis.session_id)
                .first()
            )
            if session:
                session_name = session.name

        comparisons.append(
            AnalysisComparison(
                id=analysis.id,
                date=analysis.created_at,
                session_id=analysis.session_id,
                session_name=session_name,
                total_frames=analysis.total_frames,
                avg_velocity=analysis.avg_velocity,
                max_velocity=analysis.max_velocity,
                max_height=analysis.max_height,
                total_vertical_progress=analysis.total_vertical_progress,
                avg_sway=analysis.avg_sway,
                avg_movement_economy=analysis.avg_movement_economy,
                lock_off_count=analysis.lock_off_count,
                rest_count=analysis.rest_count,
                fatigue_score=analysis.fatigue_score,
            )
        )

    return ComparisonResponse(comparisons=comparisons, count=len(comparisons))


# ========== Goal Management Endpoints ==========


@router.post("/goals", response_model=GoalResponse, status_code=201)
async def create_goal(
    goal: GoalCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Create a new training goal.

    Args:
        goal: Goal creation data
        current_user: Authenticated user
        db: Database session

    Returns:
        Created goal
    """
    # Get current value from latest metric
    latest_metric = (
        db.query(ProgressMetric)
        .filter(
            and_(
                ProgressMetric.user_id == current_user.id,
                ProgressMetric.metric_name == goal.metric_name,
            )
        )
        .order_by(ProgressMetric.recorded_at.desc())
        .first()
    )

    current_value = latest_metric.value if latest_metric else None

    db_goal = Goal(
        user_id=current_user.id,
        metric_name=goal.metric_name,
        target_value=goal.target_value,
        current_value=current_value,
        deadline=goal.deadline,
        notes=goal.notes,
        route_id=goal.route_id,
    )

    db.add(db_goal)
    db.commit()
    db.refresh(db_goal)

    # Calculate progress percentage
    progress_percentage = None
    if current_value is not None and goal.target_value != 0:
        progress_percentage = (current_value / goal.target_value) * 100

    response = GoalResponse.model_validate(db_goal)
    response.progress_percentage = progress_percentage

    return response


@router.get("/goals", response_model=list[GoalResponse])
async def list_goals(
    active_only: bool = Query(False, description="Show only active goals"),
    route_id: int = Query(None, description="Filter goals by route ID"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all goals for the current user.

    Args:
        active_only: Filter to show only active (not achieved) goals
        route_id: Optional route ID to filter goals for a specific route
        current_user: Authenticated user
        db: Database session

    Returns:
        List of goals
    """
    query = db.query(Goal).filter(Goal.user_id == current_user.id)

    if active_only:
        query = query.filter(Goal.achieved == 0)  # SQLite stores False as 0

    if route_id:
        query = query.filter(Goal.route_id == route_id)

    goals = query.order_by(Goal.created_at.desc()).all()

    # Add progress percentage to each goal
    result = []
    for goal in goals:
        response = GoalResponse.model_validate(goal)
        if goal.current_value is not None and goal.target_value != 0:
            response.progress_percentage = (
                goal.current_value / goal.target_value
            ) * 100
        result.append(response)

    return result


@router.get("/goals/{goal_id}", response_model=GoalResponse)
async def get_goal(
    goal_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a specific goal by ID.

    Args:
        goal_id: Goal ID
        current_user: Authenticated user
        db: Database session

    Returns:
        Goal data
    """
    goal = (
        db.query(Goal)
        .filter(and_(Goal.id == goal_id, Goal.user_id == current_user.id))
        .first()
    )

    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")

    response = GoalResponse.model_validate(goal)
    if goal.current_value is not None and goal.target_value != 0:
        response.progress_percentage = (goal.current_value / goal.target_value) * 100

    return response


@router.patch("/goals/{goal_id}", response_model=GoalResponse)
async def update_goal(
    goal_id: int,
    goal_update: GoalUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update a goal.

    Args:
        goal_id: Goal ID
        goal_update: Fields to update
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated goal
    """
    goal = (
        db.query(Goal)
        .filter(and_(Goal.id == goal_id, Goal.user_id == current_user.id))
        .first()
    )

    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")

    # Update fields
    update_data = goal_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(goal, field, value)

    # Auto-set achieved_at if marking as achieved
    if "achieved" in update_data and update_data["achieved"] and not goal.achieved_at:
        goal.achieved_at = datetime.utcnow()

    db.commit()
    db.refresh(goal)

    response = GoalResponse.model_validate(goal)
    if goal.current_value is not None and goal.target_value != 0:
        response.progress_percentage = (goal.current_value / goal.target_value) * 100

    return response


@router.delete("/goals/{goal_id}", status_code=204)
async def delete_goal(
    goal_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a goal.

    Args:
        goal_id: Goal ID
        current_user: Authenticated user
        db: Database session
    """
    goal = (
        db.query(Goal)
        .filter(and_(Goal.id == goal_id, Goal.user_id == current_user.id))
        .first()
    )

    if not goal:
        raise HTTPException(status_code=404, detail="Goal not found")

    db.delete(goal)
    db.commit()


# ========== ClimbSession Management Endpoints ==========


@router.post("/sessions", response_model=ClimbSessionResponse, status_code=201)
async def create_session(
    session: ClimbSessionCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Create a new climbing session.

    Args:
        session: Session creation data
        current_user: Authenticated user
        db: Database session

    Returns:
        Created session
    """
    name = session.name or session.date.strftime("%b %d, %Y")
    db_session = ClimbSession(
        user_id=current_user.id,
        name=name,
        date=session.date,
        location=session.location,
        notes=session.notes,
    )

    db.add(db_session)
    db.commit()
    db.refresh(db_session)

    return ClimbSessionResponse.model_validate(db_session)


@router.get("/sessions")
async def list_sessions(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all climbing sessions with attempts grouped by route.

    Args:
        skip: Number of sessions to skip (pagination)
        limit: Maximum number of sessions to return
        current_user: Authenticated user
        db: Database session

    Returns:
        List of sessions with route summaries
    """
    sessions = (
        db.query(ClimbSession)
        .filter(ClimbSession.user_id == current_user.id)
        .order_by(ClimbSession.date.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    result = []
    for s in sessions:
        # Group attempts by route
        route_map: dict = {}
        for attempt in s.attempts:
            route = attempt.route
            if route and route.id not in route_map:
                route_map[route.id] = {
                    "route_id": route.id,
                    "name": route.name,
                    "grade": route.grade,
                    "type": route.type,
                    "attempt_count": 0,
                }
            if route:
                route_map[route.id]["attempt_count"] += 1

        result.append(
            {
                "id": s.id,
                "name": s.name,
                "date": s.date.isoformat() if s.date else None,
                "location": s.location,
                "notes": s.notes,
                "total_videos": s.total_videos or len(s.attempts),
                "routes": list(route_map.values()),
            }
        )

    return result


@router.get("/sessions/{session_id}", response_model=ClimbSessionResponse)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a specific climbing session by ID.

    Args:
        session_id: Session ID
        current_user: Authenticated user
        db: Database session

    Returns:
        Session data
    """
    session = (
        db.query(ClimbSession)
        .filter(
            and_(ClimbSession.id == session_id, ClimbSession.user_id == current_user.id)
        )
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return ClimbSessionResponse.model_validate(session)


@router.patch("/sessions/{session_id}", response_model=ClimbSessionResponse)
async def update_session(
    session_id: int,
    session_update: ClimbSessionUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update a climbing session.

    Args:
        session_id: Session ID
        session_update: Fields to update
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated session
    """
    session = (
        db.query(ClimbSession)
        .filter(
            and_(ClimbSession.id == session_id, ClimbSession.user_id == current_user.id)
        )
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    update_data = session_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(session, field, value)

    db.commit()
    db.refresh(session)

    return ClimbSessionResponse.model_validate(session)


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a climbing session.

    Note: This will NOT delete associated analyses, just unlink them.

    Args:
        session_id: Session ID
        current_user: Authenticated user
        db: Database session
    """
    session = (
        db.query(ClimbSession)
        .filter(
            and_(ClimbSession.id == session_id, ClimbSession.user_id == current_user.id)
        )
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Unlink analyses
    db.query(Analysis).filter(Analysis.session_id == session_id).update(
        {"session_id": None}
    )

    db.delete(session)
    db.commit()


@router.get("/sessions/{session_id}/analyses", response_model=list[int])
async def get_session_analyses(
    session_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get all analysis IDs associated with a session.

    Args:
        session_id: Session ID
        current_user: Authenticated user
        db: Database session

    Returns:
        List of analysis IDs
    """
    session = (
        db.query(ClimbSession)
        .filter(
            and_(ClimbSession.id == session_id, ClimbSession.user_id == current_user.id)
        )
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    analyses = db.query(Analysis.id).filter(Analysis.session_id == session_id).all()

    return [a[0] for a in analyses]
