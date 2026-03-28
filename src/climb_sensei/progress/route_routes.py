"""Route CRUD endpoints for climbing route management.

This module provides endpoints for:
- Listing routes with filters (type, search, sort)
- Creating, reading, updating and deleting routes
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..database.config import get_db
from ..database.models import User, Route, Attempt, Analysis
from ..auth import get_current_active_user
from ..grades import grade_sort_key
from .route_schemas import RouteCreate, RouteUpdate, RouteListResponse, RouteResponse

router = APIRouter(prefix="/api/v1/routes", tags=["routes"])


def _route_to_list_response(
    route: Route, attempt_count: int, last_attempt_date, sparkline: list[float]
) -> RouteListResponse:
    """Build a RouteListResponse from a Route ORM object and computed fields.

    Args:
        route: Route ORM instance
        attempt_count: Total number of attempts for this route
        last_attempt_date: Date of the most recent attempt (or None)
        sparkline: List of avg_velocity values for the last 10 attempts

    Returns:
        RouteListResponse with all fields populated
    """
    return RouteListResponse(
        id=route.id,
        name=route.name,
        grade=route.grade,
        grade_system=route.grade_system,
        type=route.type,
        location=route.location,
        status=route.status,
        attempt_count=attempt_count,
        last_attempt_date=last_attempt_date,
        sparkline=sparkline,
    )


def _build_sparkline(db: Session, route_id: int) -> list[float]:
    """Fetch last 10 avg_velocity values from analyses linked to route attempts.

    Args:
        db: Database session
        route_id: Route ID to query attempts for

    Returns:
        List of avg_velocity floats (most recent last), empty list if none
    """
    rows = (
        db.query(Analysis.avg_velocity)
        .join(Attempt, Attempt.analysis_id == Analysis.id)
        .filter(
            and_(
                Attempt.route_id == route_id,
                Analysis.avg_velocity.isnot(None),
            )
        )
        .order_by(Attempt.date.asc())
        .limit(10)
        .all()
    )
    return [row[0] for row in rows]


# ========== Route Endpoints ==========


@router.get("", response_model=list[RouteListResponse])
async def list_routes(
    type: str = Query(None, description="Filter by route type (boulder, sport, trad)"),
    search: str = Query(None, description="Search by name (case-insensitive)"),
    sort: str = Query("recent", description="Sort order: recent, grade, attempts"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List routes for the current user with optional filters.

    Args:
        type: Optional filter by route type
        search: Optional name search substring (case-insensitive)
        sort: Sort order — recent (by last attempt), grade, or attempts count
        current_user: Authenticated user
        db: Database session

    Returns:
        List of routes with attempt counts and sparkline data
    """
    query = db.query(Route).filter(Route.user_id == current_user.id)

    if type is not None:
        query = query.filter(Route.type == type)

    if search is not None:
        query = query.filter(Route.name.ilike(f"%{search}%"))

    routes = query.all()

    # Compute attempt_count and last_attempt_date for each route in one query
    attempt_stats = (
        db.query(
            Attempt.route_id,
            func.count(Attempt.id).label("attempt_count"),
            func.max(Attempt.date).label("last_attempt_date"),
        )
        .filter(Attempt.route_id.in_([r.id for r in routes]))
        .group_by(Attempt.route_id)
        .all()
    )
    stats_by_route = {row.route_id: row for row in attempt_stats}

    results = []
    for route in routes:
        stats = stats_by_route.get(route.id)
        attempt_count = stats.attempt_count if stats else 0
        last_attempt_date = stats.last_attempt_date if stats else None
        sparkline = _build_sparkline(db, route.id)
        results.append(
            _route_to_list_response(route, attempt_count, last_attempt_date, sparkline)
        )

    # Sort
    if sort == "recent":
        results.sort(
            key=lambda r: r.last_attempt_date or route.created_at, reverse=True
        )
    elif sort == "grade":
        results.sort(key=lambda r: grade_sort_key(r.grade, r.grade_system))
    elif sort == "attempts":
        results.sort(key=lambda r: r.attempt_count, reverse=True)

    return results


@router.post("", response_model=RouteResponse, status_code=status.HTTP_201_CREATED)
async def create_route(
    route_in: RouteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Create a new climbing route.

    Args:
        route_in: Route creation data
        current_user: Authenticated user
        db: Database session

    Returns:
        Created route
    """
    db_route = Route(
        user_id=current_user.id,
        name=route_in.name,
        grade=route_in.grade,
        grade_system=route_in.grade_system,
        type=route_in.type,
        location=route_in.location,
    )
    db.add(db_route)
    db.commit()
    db.refresh(db_route)

    return RouteResponse(
        id=db_route.id,
        user_id=db_route.user_id,
        name=db_route.name,
        grade=db_route.grade,
        grade_system=db_route.grade_system,
        type=db_route.type,
        location=db_route.location,
        status=db_route.status,
        created_at=db_route.created_at,
        updated_at=db_route.updated_at,
        attempt_count=0,
        last_attempt_date=None,
    )


@router.get("/{route_id}", response_model=RouteResponse)
async def get_route(
    route_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a specific route by ID.

    Args:
        route_id: Route ID
        current_user: Authenticated user
        db: Database session

    Returns:
        Route detail with attempt statistics
    """
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    attempt_count = (
        db.query(func.count(Attempt.id)).filter(Attempt.route_id == route_id).scalar()
        or 0
    )
    last_attempt_date = (
        db.query(func.max(Attempt.date)).filter(Attempt.route_id == route_id).scalar()
    )

    return RouteResponse(
        id=route.id,
        user_id=route.user_id,
        name=route.name,
        grade=route.grade,
        grade_system=route.grade_system,
        type=route.type,
        location=route.location,
        status=route.status,
        created_at=route.created_at,
        updated_at=route.updated_at,
        attempt_count=attempt_count,
        last_attempt_date=last_attempt_date,
    )


@router.patch("/{route_id}", response_model=RouteResponse)
async def update_route(
    route_id: int,
    route_update: RouteUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update a climbing route.

    Args:
        route_id: Route ID
        route_update: Fields to update (all optional)
        current_user: Authenticated user
        db: Database session

    Returns:
        Updated route
    """
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    update_data = route_update.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(route, field, value)

    db.commit()
    db.refresh(route)

    attempt_count = (
        db.query(func.count(Attempt.id)).filter(Attempt.route_id == route_id).scalar()
        or 0
    )
    last_attempt_date = (
        db.query(func.max(Attempt.date)).filter(Attempt.route_id == route_id).scalar()
    )

    return RouteResponse(
        id=route.id,
        user_id=route.user_id,
        name=route.name,
        grade=route.grade,
        grade_system=route.grade_system,
        type=route.type,
        location=route.location,
        status=route.status,
        created_at=route.created_at,
        updated_at=route.updated_at,
        attempt_count=attempt_count,
        last_attempt_date=last_attempt_date,
    )


@router.delete("/{route_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_route(
    route_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a climbing route and all associated attempts.

    Args:
        route_id: Route ID
        current_user: Authenticated user
        db: Database session
    """
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    db.delete(route)
    db.commit()
