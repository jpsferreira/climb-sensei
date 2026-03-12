# UI/UX Redesign Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign Climb Sensei from a session/analysis-centric app to a route-centric "Route Journal" with mobile-first UI.

**Architecture:** Add Route and Attempt models that wrap existing Video/Analysis entities. New REST endpoints for route CRUD and attempt management. Rebuild all frontend templates with bottom tab navigation, compact route list, timeline-based route detail, calendar sessions, and context-aware upload.

**Tech Stack:** FastAPI, SQLAlchemy, Alembic, Jinja2, Chart.js, vanilla CSS/JS (no framework).

**Spec:** `docs/superpowers/specs/2026-03-12-ui-ux-redesign-design.md`

---

## Prerequisites & Implementation Notes

### Task 0: Introduce create_app factory (required for testing)

**Files:**
- Modify: `app/main.py`

The current `app/main.py` creates a module-level `app = FastAPI(...)`. Tests in Tasks 8-10 need a `create_app()` factory to override dependencies cleanly.

- [ ] **Step 1: Refactor app/main.py to use factory pattern**

Wrap the existing app setup in a `create_app()` function. Keep the module-level `app = create_app()` for backward compat with `run_app.py`.

```python
def create_app() -> FastAPI:
    app = FastAPI(title="ClimbSensei")
    # ... existing include_router calls, static mounts, etc.
    return app

app = create_app()
```

- [ ] **Step 2: Verify app still starts**

Run: `python run_app.py`
Expected: App starts normally.

- [ ] **Step 3: Commit**

```bash
git add app/main.py
git commit -m "refactor: introduce create_app factory for testability"
```

### Critical Notes for Implementers

**Model style:** The existing codebase mixes `Column(...)` (Video, Analysis, etc.) and `Mapped[T] = mapped_column(...)` (User). New models (Route, Attempt) should use `Column(...)` to match the majority pattern. This is consistent with the existing codebase.

**persist_results return value:** In Task 10, the call to `persist_results(...)` in `_run_analysis_pipeline` (api.py line ~107) must be changed to capture the return value: `db_analysis_id = persist_results(...)`. The function already returns the analysis ID but the call site doesn't capture it.

**Metric allow-list:** The `get_route_progress` endpoint (Task 9) uses `getattr(analysis, metric)`. Validate `metric` against a known list:
```python
ALLOWED_METRICS = [
    "avg_velocity", "max_velocity", "max_height", "total_vertical_progress",
    "avg_sway", "avg_movement_economy", "lock_off_count", "rest_count", "fatigue_score",
]
if metric not in ALLOWED_METRICS:
    raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")
```

**Frontend dependencies:**
- Chart.js 4.4.0 is already loaded in the current base.html — keep it in the rewrite (Task 12).
- The `/upload` page route must be added in Task 12 since the current `/` maps to upload.html and will be changed to routes.html in Task 13. Add: `@router.get("/upload", ...)` returning `upload.html` with `active_tab: "upload"`.

**Missing PATCH endpoint for attempt notes:** Task 19 references editing notes on blur. Add a PATCH endpoint to `attempt_routes.py` (Task 9):
```python
@router.patch("/routes/{route_id}/attempts/{attempt_id}")
async def update_attempt(route_id, attempt_id, notes: str = None, ...):
    # Update attempt.notes
```

**Goals page redirect:** In Task 20, also redirect `/goals` → `/profile`:
```python
@router.get("/goals")
async def goals_redirect():
    return RedirectResponse(url="/profile", status_code=301)
```

**Session auto-creation in upload flow (Task 15):** The JS should: (1) fetch `/api/sessions` for today's date, (2) if none exists, POST to create one with `date: today` and no name (auto-generated), (3) use the session_id in the upload form.

**Desktop responsive nav (Task 12):** At `min-width: 768px`:
```css
@media (min-width: 768px) {
    .bottom-nav {
        position: fixed;
        left: 0; top: 0; bottom: 0;
        width: 200px;
        flex-direction: column;
        border-top: none;
        border-right: 1px solid var(--border);
        padding-top: 20px;
    }
    .main-content { margin-left: 200px; padding-bottom: 0; }
}
```

**Grade color helper (Task 13):** Use this JS function for grade badge colors:
```javascript
function gradeColor(grade) {
    const num = parseInt(grade.replace(/[^0-9]/g, ''));
    if (isNaN(num)) return 'gray';
    if (num <= 3) return 'green';
    if (num <= 6) return 'amber';
    if (num <= 9) return 'orange';
    return 'red';
}
```

With CSS classes: `.grade-green`, `.grade-amber`, `.grade-orange`, `.grade-red`, `.grade-gray`.

**Sparkline SVG helper (Task 13):**
```javascript
function renderSparkline(values) {
    if (!values.length) return '';
    const h = 20, w = 50;
    const max = Math.max(...values), min = Math.min(...values);
    const range = max - min || 1;
    const points = values.map((v, i) =>
        `${(i / (values.length - 1)) * w},${h - ((v - min) / range) * h}`
    ).join(' ');
    return `<svg width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">
        <polyline points="${points}" fill="none" stroke="var(--accent)" stroke-width="1.5"/>
    </svg>`;
}
```

**New route creation (Task 13):** Add a "+" button at the bottom of the route list that opens a simple modal with fields: name, grade, grade_system (dropdown), type (dropdown), location. POST to `/api/routes`. Same modal reusable in upload flow (Task 15).

**Profile stats (Task 17):** Compute client-side: fetch `/api/routes` (count), `/api/sessions` (count + frequency calc), count total attempts from routes. No new endpoint needed.

---

## Chunk 1: Backend — Models & Migration

### Task 1: Add Route model

**Files:**
- Modify: `src/climb_sensei/database/models.py`
- Create: `tests/test_route_model.py`

- [ ] **Step 1: Write failing test for Route model**

```python
# tests/test_route_model.py
"""Tests for Route model."""
import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from climb_sensei.database.models import Base, User, Route


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    # Create a test user (minimal fields for fastapi-users)
    user = User(
        id=1,
        email="test@example.com",
        hashed_password="fakehash",
        full_name="Test User",
    )
    session.add(user)
    session.commit()
    yield session
    session.close()


def test_create_route(db_session):
    route = Route(
        user_id=1,
        name="Crimpy Arete",
        grade="V4",
        grade_system="hueco",
        type="boulder",
        location="Magic Wood",
        status="projecting",
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    assert route.id is not None
    assert route.name == "Crimpy Arete"
    assert route.grade == "V4"
    assert route.grade_system == "hueco"
    assert route.type == "boulder"
    assert route.location == "Magic Wood"
    assert route.status == "projecting"
    assert route.created_at is not None
    assert route.updated_at is not None


def test_route_user_relationship(db_session):
    route = Route(
        user_id=1, name="Test Route", grade="6a", grade_system="french", type="sport"
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    assert route.user.email == "test@example.com"


def test_route_defaults(db_session):
    route = Route(
        user_id=1, name="Minimal", grade="V0", grade_system="hueco", type="boulder"
    )
    db_session.add(route)
    db_session.commit()
    db_session.refresh(route)

    assert route.status == "projecting"
    assert route.location is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_route_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'Route'`

- [ ] **Step 3: Add Route model to models.py**

Add after the `Video` class in `src/climb_sensei/database/models.py`:

```python
class Route(Base):
    """Climbing route model.

    Represents a specific climbing route that a user is tracking.
    Routes are the primary organizational unit — users track
    progression on routes over time through multiple attempts.
    """

    __tablename__ = "routes"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    name = Column(String(255), nullable=False)
    grade = Column(String(20), nullable=False)
    grade_system = Column(String(20), nullable=False)  # hueco, font, yds, french
    type = Column(String(20), nullable=False)  # boulder, sport, trad
    location = Column(String(255))
    status = Column(String(20), default="projecting", nullable=False)  # projecting, sent
    created_at = Column(DateTime, default=utcnow, nullable=False)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="routes")
    attempts = relationship(
        "Attempt", back_populates="route", cascade="all, delete-orphan"
    )
    goals = relationship("Goal", back_populates="route")

    def __repr__(self) -> str:
        return f"<Route(id={self.id}, name='{self.name}', grade='{self.grade}')>"
```

Also add `routes` relationship to the `User` class:

```python
routes: Mapped[list["Route"]] = relationship(
    "Route", back_populates="user", cascade="all, delete-orphan"
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_route_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/climb_sensei/database/models.py tests/test_route_model.py
git commit -m "feat: add Route model"
```

### Task 2: Add Attempt model

**Files:**
- Modify: `src/climb_sensei/database/models.py`
- Create: `tests/test_attempt_model.py`

- [ ] **Step 1: Write failing test for Attempt model**

```python
# tests/test_attempt_model.py
"""Tests for Attempt model."""
import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from climb_sensei.database.models import Base, User, Video, Route, Attempt, Analysis
from climb_sensei.types import VideoStatus


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(
        id=1,
        email="test@example.com",
        hashed_password="fakehash",
        full_name="Test User",
    )
    session.add(user)
    route = Route(
        user_id=1, name="Test Route", grade="V4", grade_system="hueco", type="boulder"
    )
    session.add(route)
    video = Video(
        user_id=1,
        filename="test.mp4",
        file_path="/tmp/test.mp4",
        status=VideoStatus.COMPLETED,
    )
    session.add(video)
    session.commit()
    yield session
    session.close()


def test_create_attempt(db_session):
    route = db_session.query(Route).first()
    video = db_session.query(Video).first()

    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime.now(timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.id is not None
    assert attempt.route_id == route.id
    assert attempt.video_id == video.id
    assert attempt.session_id is None
    assert attempt.analysis_id is None
    assert attempt.notes is None
    assert attempt.created_at is not None


def test_attempt_route_relationship(db_session):
    route = db_session.query(Route).first()
    video = db_session.query(Video).first()

    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        date=datetime.now(timezone.utc),
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.route.name == "Test Route"
    assert len(route.attempts) == 1


def test_attempt_with_analysis(db_session):
    route = db_session.query(Route).first()
    video = db_session.query(Video).first()

    analysis = Analysis(video_id=video.id, run_metrics=True)
    db_session.add(analysis)
    db_session.commit()

    attempt = Attempt(
        route_id=route.id,
        video_id=video.id,
        analysis_id=analysis.id,
        date=datetime.now(timezone.utc),
        notes="Good send attempt",
    )
    db_session.add(attempt)
    db_session.commit()
    db_session.refresh(attempt)

    assert attempt.analysis.id == analysis.id
    assert attempt.notes == "Good send attempt"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attempt_model.py -v`
Expected: FAIL — `ImportError: cannot import name 'Attempt'`

- [ ] **Step 3: Add Attempt model to models.py**

Add after the `Route` class:

```python
class Attempt(Base):
    """Climbing attempt model.

    Represents a single attempt on a route, linking a video and its
    analysis to a specific route and optional session. This is the
    bridge between the route-centric UI and the existing Video/Analysis
    data layer.
    """

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
```

Add `attempts` relationship to `ClimbSession`:

```python
attempts = relationship("Attempt", back_populates="session")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_attempt_model.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/climb_sensei/database/models.py tests/test_attempt_model.py
git commit -m "feat: add Attempt model"
```

### Task 3: Update Goal model with route_id

**Files:**
- Modify: `src/climb_sensei/database/models.py`
- Create: `tests/test_goal_route.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_goal_route.py
"""Tests for Goal-Route relationship."""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from climb_sensei.database.models import Base, User, Route, Goal


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, email="test@example.com", hashed_password="fakehash")
    route = Route(
        user_id=1, name="Test Route", grade="V4", grade_system="hueco", type="boulder"
    )
    session.add_all([user, route])
    session.commit()
    yield session
    session.close()


def test_goal_with_route(db_session):
    route = db_session.query(Route).first()
    goal = Goal(
        user_id=1,
        route_id=route.id,
        metric_name="avg_velocity",
        target_value=0.5,
    )
    db_session.add(goal)
    db_session.commit()
    db_session.refresh(goal)

    assert goal.route_id == route.id
    assert goal.route.name == "Test Route"
    assert len(route.goals) == 1


def test_goal_without_route_for_migration(db_session):
    """Goals can have null route_id for backward compatibility."""
    goal = Goal(
        user_id=1,
        metric_name="avg_velocity",
        target_value=0.5,
    )
    db_session.add(goal)
    db_session.commit()
    db_session.refresh(goal)

    assert goal.route_id is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_goal_route.py -v`
Expected: FAIL — Goal has no `route_id` column / `route` relationship

- [ ] **Step 3: Add route_id to Goal model**

In `src/climb_sensei/database/models.py`, update the `Goal` class — add after `user_id`:

```python
route_id = Column(Integer, ForeignKey("routes.id"), index=True)  # nullable for migration
```

Add relationship:

```python
route = relationship("Route", back_populates="goals")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_goal_route.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/climb_sensei/database/models.py tests/test_goal_route.py
git commit -m "feat: add route_id to Goal model"
```

### Task 4: Update ClimbSession — make name nullable

**Files:**
- Modify: `src/climb_sensei/database/models.py`
- Create: `tests/test_session_nullable_name.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_session_nullable_name.py
"""Tests for ClimbSession nullable name."""
import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from climb_sensei.database.models import Base, User, ClimbSession


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, email="test@example.com", hashed_password="fakehash")
    session.add(user)
    session.commit()
    yield session
    session.close()


def test_session_without_name(db_session):
    """Session can be created without a name (auto-created sessions)."""
    session = ClimbSession(
        user_id=1,
        date=datetime.now(timezone.utc),
        location="Local Gym",
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)

    assert session.id is not None
    assert session.name is None


def test_session_with_name(db_session):
    """Session can still be created with an explicit name."""
    session = ClimbSession(
        user_id=1,
        name="Morning bouldering",
        date=datetime.now(timezone.utc),
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)

    assert session.name == "Morning bouldering"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_session_nullable_name.py -v`
Expected: FAIL — `NOT NULL constraint failed: climb_sessions.name`

- [ ] **Step 3: Make name nullable**

In `src/climb_sensei/database/models.py`, change the `ClimbSession.name` line:

```python
# Before:
name = Column(String(255), nullable=False)
# After:
name = Column(String(255), nullable=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_session_nullable_name.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/climb_sensei/database/models.py tests/test_session_nullable_name.py
git commit -m "feat: make ClimbSession.name nullable for auto-created sessions"
```

### Task 5: Generate Alembic migration

**Files:**
- Create: `alembic/versions/<auto>_add_route_and_attempt_models.py`

- [ ] **Step 1: Generate migration**

Run: `uv run alembic revision --autogenerate -m "add route and attempt models"`

- [ ] **Step 2: Review the generated migration file**

Open the generated file in `alembic/versions/`. Verify it includes:
- Create `routes` table
- Create `attempts` table
- Add `route_id` column to `goals` table
- Make `climb_sessions.name` nullable

- [ ] **Step 3: Run migration**

Run: `uv run alembic upgrade head`

- [ ] **Step 4: Verify migration applied**

Run: `uv run alembic current`
Expected: Shows the new migration as current head.

- [ ] **Step 5: Commit**

```bash
git add alembic/versions/
git commit -m "feat: add migration for route and attempt models"
```

### Task 6: Add grade sorting utility

**Files:**
- Create: `src/climb_sensei/grades.py`
- Create: `tests/test_grades.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_grades.py
"""Tests for grade sorting utility."""
import pytest
from climb_sensei.grades import grade_sort_key


def test_hueco_ordering():
    grades = ["V4", "V0", "V10", "V2", "V7"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "hueco"))
    assert sorted_grades == ["V0", "V2", "V4", "V7", "V10"]


def test_french_ordering():
    grades = ["6b+", "5a", "7a", "6a", "8a+"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "french"))
    assert sorted_grades == ["5a", "6a", "6b+", "7a", "8a+"]


def test_yds_ordering():
    grades = ["5.11a", "5.9", "5.10d", "5.12b"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "yds"))
    assert sorted_grades == ["5.9", "5.10d", "5.11a", "5.12b"]


def test_font_ordering():
    grades = ["6A+", "4", "7A", "5+", "8A"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "font"))
    assert sorted_grades == ["4", "5+", "6A+", "7A", "8A"]


def test_unknown_grade_sorts_to_end():
    grades = ["V4", "???", "V2"]
    sorted_grades = sorted(grades, key=lambda g: grade_sort_key(g, "hueco"))
    assert sorted_grades == ["V2", "V4", "???"]


def test_unknown_system_sorts_to_end():
    assert grade_sort_key("V4", "unknown") == (999, 999)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_grades.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'climb_sensei.grades'`

- [ ] **Step 3: Implement grade sorting**

```python
# src/climb_sensei/grades.py
"""Grade sorting utilities.

Maps climbing grades from different systems to numeric sort keys.
Supports Hueco (V-scale), French, YDS, and Font systems.
"""

import re
from typing import Tuple

# Hueco V-scale: V0, V1, ... V17
_HUECO_RE = re.compile(r"^V(\d+)$")

# French sport: 4a, 5a, 5b, 5c, 6a, 6a+, 6b, 6b+, ... 9c+
_FRENCH_LETTER_MAP = {"a": 0, "b": 1, "c": 2}
_FRENCH_RE = re.compile(r"^(\d)([abc])(\+?)$")

# YDS: 5.0, 5.1, ... 5.9, 5.10a, 5.10b, ... 5.15d
_YDS_RE = re.compile(r"^5\.(\d+)([a-d]?)$")
_YDS_LETTER_MAP = {"": 0, "a": 0, "b": 1, "c": 2, "d": 3}

# Font (bouldering): 4, 4+, 5, 5+, 6A, 6A+, 6B, 6B+, 6C, 6C+, 7A, ... 8C+
_FONT_LETTER_MAP = {"": 0, "a": 0, "b": 1, "c": 2}
_FONT_RE = re.compile(r"^(\d)([ABC]?)(\+?)$", re.IGNORECASE)

_FALLBACK = (999, 999)


def grade_sort_key(grade: str, system: str) -> Tuple[int, int]:
    """Return a (major, minor) sort key for a grade string.

    Args:
        grade: The grade string (e.g., "V4", "6b+", "5.11a", "7A").
        system: The grading system ("hueco", "french", "yds", "font").

    Returns:
        Tuple of (major, minor) for sorting. Unknown grades return (999, 999).
    """
    try:
        if system == "hueco":
            m = _HUECO_RE.match(grade)
            if m:
                return (int(m.group(1)), 0)

        elif system == "french":
            m = _FRENCH_RE.match(grade)
            if m:
                num, letter, plus = m.groups()
                minor = _FRENCH_LETTER_MAP[letter] * 2 + (1 if plus else 0)
                return (int(num), minor)

        elif system == "yds":
            m = _YDS_RE.match(grade)
            if m:
                num, letter = m.groups()
                minor = _YDS_LETTER_MAP.get(letter, 0)
                return (int(num), minor)

        elif system == "font":
            m = _FONT_RE.match(grade)
            if m:
                num, letter, plus = m.groups()
                letter_val = _FONT_LETTER_MAP.get(letter.lower(), 0)
                minor = letter_val * 2 + (1 if plus else 0)
                return (int(num), minor)

    except (ValueError, KeyError):
        pass

    return _FALLBACK
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_grades.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/climb_sensei/grades.py tests/test_grades.py
git commit -m "feat: add grade sorting utility"
```

---

## Chunk 2: Backend — Route & Attempt API Endpoints

### Task 7: Add Route schemas

**Files:**
- Create: `src/climb_sensei/progress/route_schemas.py`

- [ ] **Step 1: Create route schemas**

```python
# src/climb_sensei/progress/route_schemas.py
"""Pydantic schemas for Route and Attempt management."""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


# ========== Route Schemas ==========


class RouteCreate(BaseModel):
    """Schema for creating a route."""

    name: str = Field(..., max_length=255)
    grade: str = Field(..., max_length=20)
    grade_system: str = Field(..., pattern="^(hueco|font|yds|french)$")
    type: str = Field(..., pattern="^(boulder|sport|trad)$")
    location: Optional[str] = Field(None, max_length=255)


class RouteUpdate(BaseModel):
    """Schema for updating a route."""

    name: Optional[str] = Field(None, max_length=255)
    grade: Optional[str] = Field(None, max_length=20)
    grade_system: Optional[str] = Field(None, pattern="^(hueco|font|yds|french)$")
    type: Optional[str] = Field(None, pattern="^(boulder|sport|trad)$")
    location: Optional[str] = Field(None, max_length=255)
    status: Optional[str] = Field(None, pattern="^(projecting|sent)$")


class RouteResponse(BaseModel):
    """Schema for route responses."""

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
    """Schema for route list item with sparkline data."""

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
    sparkline: list[float] = []  # Recent metric values for mini chart


# ========== Attempt Schemas ==========


class AttemptCreate(BaseModel):
    """Schema for creating an attempt (used internally after upload)."""

    route_id: int
    session_id: Optional[int] = None
    notes: Optional[str] = None
    date: Optional[datetime] = None  # Defaults to now


class AttemptResponse(BaseModel):
    """Schema for attempt responses."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    route_id: int
    video_id: int
    session_id: Optional[int] = None
    analysis_id: Optional[int] = None
    notes: Optional[str] = None
    date: datetime
    created_at: datetime
    # Inline metrics from analysis (denormalized for timeline display)
    avg_velocity: Optional[float] = None
    avg_sway: Optional[float] = None
    avg_movement_economy: Optional[float] = None
    has_video: bool = False
    video_filename: Optional[str] = None


class AttemptDetailResponse(AttemptResponse):
    """Schema for attempt detail with full metrics."""

    summary: Optional[dict] = None
    history: Optional[dict] = None
    video_quality: Optional[dict] = None
    tracking_quality: Optional[dict] = None
    output_video_path: Optional[str] = None
    # Delta vs previous attempt
    prev_attempt_id: Optional[int] = None
    deltas: Optional[dict] = None  # {metric_name: float_delta}
```

- [ ] **Step 2: Commit**

```bash
git add src/climb_sensei/progress/route_schemas.py
git commit -m "feat: add Route and Attempt Pydantic schemas"
```

### Task 8: Add Route CRUD endpoints

**Files:**
- Create: `src/climb_sensei/progress/route_routes.py`
- Create: `tests/test_route_api.py`

- [ ] **Step 1: Write failing test for route list endpoint**

```python
# tests/test_route_api.py
"""Tests for Route API endpoints."""
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climb_sensei.database.models import Base, User, Route
from climb_sensei.database.config import get_db
from climb_sensei.auth import get_current_active_user
from app.main import create_app


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, email="test@example.com", hashed_password="fakehash")
    session.add(user)
    session.commit()
    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def client(db_session):
    app = create_app()

    def override_get_db():
        yield db_session

    def override_get_user():
        return db_session.query(User).first()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_active_user] = override_get_user

    return TestClient(app)


def test_list_routes_empty(client):
    response = client.get("/api/routes")
    assert response.status_code == 200
    assert response.json() == []


def test_create_route(client):
    response = client.post(
        "/api/routes",
        json={
            "name": "Crimpy Arete",
            "grade": "V4",
            "grade_system": "hueco",
            "type": "boulder",
            "location": "Magic Wood",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Crimpy Arete"
    assert data["grade"] == "V4"
    assert data["status"] == "projecting"


def test_get_route(client):
    # Create a route first
    client.post(
        "/api/routes",
        json={
            "name": "Test",
            "grade": "V0",
            "grade_system": "hueco",
            "type": "boulder",
        },
    )
    response = client.get("/api/routes/1")
    assert response.status_code == 200
    assert response.json()["name"] == "Test"


def test_update_route(client):
    client.post(
        "/api/routes",
        json={
            "name": "Test",
            "grade": "V0",
            "grade_system": "hueco",
            "type": "boulder",
        },
    )
    response = client.patch("/api/routes/1", json={"status": "sent"})
    assert response.status_code == 200
    assert response.json()["status"] == "sent"


def test_delete_route(client):
    client.post(
        "/api/routes",
        json={
            "name": "Test",
            "grade": "V0",
            "grade_system": "hueco",
            "type": "boulder",
        },
    )
    response = client.delete("/api/routes/1")
    assert response.status_code == 204
    assert client.get("/api/routes/1").status_code == 404


def test_list_routes_with_filters(client):
    client.post(
        "/api/routes",
        json={
            "name": "Boulder",
            "grade": "V4",
            "grade_system": "hueco",
            "type": "boulder",
        },
    )
    client.post(
        "/api/routes",
        json={
            "name": "Sport",
            "grade": "6a",
            "grade_system": "french",
            "type": "sport",
        },
    )

    # Filter by type
    response = client.get("/api/routes?type=boulder")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["name"] == "Boulder"

    # Search by name
    response = client.get("/api/routes?search=sport")
    assert response.status_code == 200
    assert len(response.json()) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_route_api.py -v`
Expected: FAIL — route endpoints not found (404)

- [ ] **Step 3: Implement route endpoints**

```python
# src/climb_sensei/progress/route_routes.py
"""Route management API endpoints."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import and_, func

from ..database.config import get_db
from ..database.models import User, Route, Attempt, Analysis, ProgressMetric
from ..auth import get_current_active_user
from ..grades import grade_sort_key
from .route_schemas import (
    RouteCreate,
    RouteUpdate,
    RouteResponse,
    RouteListResponse,
)

router = APIRouter(prefix="/api", tags=["routes"])


@router.get("/routes", response_model=list[RouteListResponse])
async def list_routes(
    type: str = Query(None, pattern="^(boulder|sport|trad)$"),
    search: str = Query(None, max_length=100),
    sort: str = Query("recent", pattern="^(recent|grade|attempts)$"),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all routes for the current user."""
    query = db.query(Route).filter(Route.user_id == current_user.id)

    if type:
        query = query.filter(Route.type == type)
    if search:
        query = query.filter(Route.name.ilike(f"%{search}%"))

    routes = query.all()

    result = []
    for route in routes:
        attempt_count = (
            db.query(func.count(Attempt.id))
            .filter(Attempt.route_id == route.id)
            .scalar()
        )
        last_attempt = (
            db.query(Attempt.date)
            .filter(Attempt.route_id == route.id)
            .order_by(Attempt.date.desc())
            .first()
        )

        # Get sparkline data (last 10 velocity values for this route)
        sparkline_data = (
            db.query(Analysis.avg_velocity)
            .join(Attempt, Attempt.analysis_id == Analysis.id)
            .filter(
                and_(
                    Attempt.route_id == route.id,
                    Analysis.avg_velocity.isnot(None),
                )
            )
            .order_by(Attempt.date.asc())
            .limit(10)
            .all()
        )
        sparkline = [v[0] for v in sparkline_data if v[0] is not None]

        result.append(
            RouteListResponse(
                id=route.id,
                name=route.name,
                grade=route.grade,
                grade_system=route.grade_system,
                type=route.type,
                location=route.location,
                status=route.status,
                attempt_count=attempt_count,
                last_attempt_date=last_attempt[0] if last_attempt else None,
                sparkline=sparkline,
            )
        )

    # Sort results
    if sort == "recent":
        result.sort(
            key=lambda r: r.last_attempt_date or datetime.min.replace(
                tzinfo=timezone.utc
            ),
            reverse=True,
        )
    elif sort == "grade":
        result.sort(key=lambda r: grade_sort_key(r.grade, r.grade_system))
    elif sort == "attempts":
        result.sort(key=lambda r: r.attempt_count, reverse=True)

    return result


@router.post("/routes", response_model=RouteResponse, status_code=201)
async def create_route(
    route: RouteCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Create a new climbing route."""
    db_route = Route(
        user_id=current_user.id,
        name=route.name,
        grade=route.grade,
        grade_system=route.grade_system,
        type=route.type,
        location=route.location,
    )
    db.add(db_route)
    db.commit()
    db.refresh(db_route)

    response = RouteResponse.model_validate(db_route)
    return response


@router.get("/routes/{route_id}", response_model=RouteResponse)
async def get_route(
    route_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get a specific route by ID."""
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    attempt_count = (
        db.query(func.count(Attempt.id))
        .filter(Attempt.route_id == route.id)
        .scalar()
    )
    last_attempt = (
        db.query(Attempt.date)
        .filter(Attempt.route_id == route.id)
        .order_by(Attempt.date.desc())
        .first()
    )

    response = RouteResponse.model_validate(route)
    response.attempt_count = attempt_count
    response.last_attempt_date = last_attempt[0] if last_attempt else None
    return response


@router.patch("/routes/{route_id}", response_model=RouteResponse)
async def update_route(
    route_id: int,
    route_update: RouteUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update a route."""
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

    return RouteResponse.model_validate(route)


@router.delete("/routes/{route_id}", status_code=204)
async def delete_route(
    route_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete a route and all its attempts."""
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    db.delete(route)
    db.commit()
```

- [ ] **Step 4: Register route_routes router in app/main.py**

In `app/main.py`, add:

```python
from climb_sensei.progress.route_routes import router as route_router
# ... in create_app():
app.include_router(route_router)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_route_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/climb_sensei/progress/route_routes.py src/climb_sensei/progress/route_schemas.py tests/test_route_api.py app/main.py
git commit -m "feat: add Route CRUD API endpoints"
```

### Task 9: Add Attempt endpoints

**Files:**
- Create: `src/climb_sensei/progress/attempt_routes.py`
- Create: `tests/test_attempt_api.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_attempt_api.py
"""Tests for Attempt API endpoints."""
import pytest
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climb_sensei.database.models import (
    Base, User, Route, Video, Analysis, Attempt,
)
from climb_sensei.database.config import get_db
from climb_sensei.auth import get_current_active_user
from climb_sensei.types import VideoStatus
from app.main import create_app


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, email="test@example.com", hashed_password="fakehash")
    route = Route(
        user_id=1, name="Test Route", grade="V4",
        grade_system="hueco", type="boulder",
    )
    video = Video(
        user_id=1, filename="test.mp4",
        file_path="/tmp/test.mp4", status=VideoStatus.COMPLETED,
    )
    session.add_all([user, route, video])
    session.commit()

    # Add analysis linked to video
    analysis = Analysis(
        video_id=video.id, run_metrics=True,
        avg_velocity=0.42, avg_sway=0.15, avg_movement_economy=0.85,
        summary={"test": True},
    )
    session.add(analysis)
    session.commit()

    # Add attempt linking route, video, analysis
    attempt = Attempt(
        route_id=route.id, video_id=video.id,
        analysis_id=analysis.id,
        date=datetime(2026, 3, 10, tzinfo=timezone.utc),
    )
    session.add(attempt)
    session.commit()

    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def client(db_session):
    app = create_app()

    def override_get_db():
        yield db_session

    def override_get_user():
        return db_session.query(User).first()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_active_user] = override_get_user
    return TestClient(app)


def test_list_attempts(client):
    response = client.get("/api/routes/1/attempts")
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 1
    assert data[0]["avg_velocity"] == 0.42


def test_get_attempt_detail(client):
    response = client.get("/api/routes/1/attempts/1")
    assert response.status_code == 200
    data = response.json()
    assert data["summary"] == {"test": True}
    assert data["avg_velocity"] == 0.42


def test_delete_attempt(client):
    response = client.delete("/api/routes/1/attempts/1")
    assert response.status_code == 204
    # Attempt deleted but video/analysis still exist
    assert client.get("/api/routes/1/attempts").json() == []


def test_route_progress(client):
    response = client.get("/api/routes/1/progress/avg_velocity")
    assert response.status_code == 200
    data = response.json()
    assert data["metric"] == "avg_velocity"
    assert len(data["data"]) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_attempt_api.py -v`
Expected: FAIL — endpoints not found

- [ ] **Step 3: Implement attempt endpoints**

```python
# src/climb_sensei/progress/attempt_routes.py
"""Attempt management API endpoints."""

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import and_

from ..database.config import get_db
from ..database.models import User, Route, Attempt, Analysis, Video
from ..auth import get_current_active_user
from .route_schemas import AttemptResponse, AttemptDetailResponse

router = APIRouter(prefix="/api", tags=["attempts"])

_DELTA_METRICS = [
    "avg_velocity", "max_velocity", "max_height",
    "total_vertical_progress", "avg_sway",
    "avg_movement_economy", "lock_off_count",
    "rest_count", "fatigue_score",
]


@router.get(
    "/routes/{route_id}/attempts", response_model=list[AttemptResponse]
)
async def list_attempts(
    route_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """List all attempts for a route, most recent first."""
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    attempts = (
        db.query(Attempt)
        .filter(Attempt.route_id == route_id)
        .order_by(Attempt.date.desc())
        .all()
    )

    result = []
    for attempt in attempts:
        analysis = (
            db.query(Analysis).filter(Analysis.id == attempt.analysis_id).first()
            if attempt.analysis_id
            else None
        )
        video = db.query(Video).filter(Video.id == attempt.video_id).first()

        result.append(
            AttemptResponse(
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
                avg_movement_economy=(
                    analysis.avg_movement_economy if analysis else None
                ),
                has_video=analysis.output_video_path is not None if analysis else False,
                video_filename=video.filename if video else None,
            )
        )

    return result


@router.get(
    "/routes/{route_id}/attempts/{attempt_id}",
    response_model=AttemptDetailResponse,
)
async def get_attempt_detail(
    route_id: int,
    attempt_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get full attempt detail with metrics and comparison to previous."""
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    attempt = (
        db.query(Attempt)
        .filter(and_(Attempt.id == attempt_id, Attempt.route_id == route_id))
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    analysis = (
        db.query(Analysis).filter(Analysis.id == attempt.analysis_id).first()
        if attempt.analysis_id
        else None
    )
    video = db.query(Video).filter(Video.id == attempt.video_id).first()

    # Find previous attempt for deltas
    prev_attempt = (
        db.query(Attempt)
        .filter(
            and_(
                Attempt.route_id == route_id,
                Attempt.date < attempt.date,
            )
        )
        .order_by(Attempt.date.desc())
        .first()
    )

    deltas = None
    prev_attempt_id = None
    if prev_attempt and prev_attempt.analysis_id and analysis:
        prev_analysis = (
            db.query(Analysis)
            .filter(Analysis.id == prev_attempt.analysis_id)
            .first()
        )
        if prev_analysis:
            prev_attempt_id = prev_attempt.id
            deltas = {}
            for metric in _DELTA_METRICS:
                curr = getattr(analysis, metric, None)
                prev = getattr(prev_analysis, metric, None)
                if curr is not None and prev is not None:
                    deltas[metric] = curr - prev

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
        avg_sway=analysis.avg_sway if analysis else None,
        avg_movement_economy=analysis.avg_movement_economy if analysis else None,
        has_video=analysis.output_video_path is not None if analysis else False,
        video_filename=video.filename if video else None,
        summary=analysis.summary if analysis else None,
        history=analysis.history if analysis else None,
        video_quality=analysis.video_quality if analysis else None,
        tracking_quality=analysis.tracking_quality if analysis else None,
        output_video_path=analysis.output_video_path if analysis else None,
        prev_attempt_id=prev_attempt_id,
        deltas=deltas,
    )


@router.delete("/routes/{route_id}/attempts/{attempt_id}", status_code=204)
async def delete_attempt(
    route_id: int,
    attempt_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Delete an attempt. Does not delete the underlying Video or Analysis."""
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    attempt = (
        db.query(Attempt)
        .filter(and_(Attempt.id == attempt_id, Attempt.route_id == route_id))
        .first()
    )
    if not attempt:
        raise HTTPException(status_code=404, detail="Attempt not found")

    db.delete(attempt)
    db.commit()


@router.get("/routes/{route_id}/progress/{metric}")
async def get_route_progress(
    route_id: int,
    metric: str,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get metric trend data for a specific route."""
    route = (
        db.query(Route)
        .filter(and_(Route.id == route_id, Route.user_id == current_user.id))
        .first()
    )
    if not route:
        raise HTTPException(status_code=404, detail="Route not found")

    # Get metric values from analyses linked through attempts
    attempts = (
        db.query(Attempt)
        .filter(Attempt.route_id == route_id)
        .order_by(Attempt.date.asc())
        .all()
    )

    data = []
    for attempt in attempts:
        if not attempt.analysis_id:
            continue
        analysis = (
            db.query(Analysis).filter(Analysis.id == attempt.analysis_id).first()
        )
        if analysis:
            value = getattr(analysis, metric, None)
            if value is not None:
                data.append({
                    "date": attempt.date.isoformat(),
                    "value": value,
                    "attempt_id": attempt.id,
                    "analysis_id": analysis.id,
                })

    values = [d["value"] for d in data]

    return {
        "metric": metric,
        "route_id": route_id,
        "data": data,
        "count": len(data),
        "min_value": min(values) if values else None,
        "max_value": max(values) if values else None,
        "avg_value": sum(values) / len(values) if values else None,
    }
```

- [ ] **Step 4: Register attempt router in app/main.py**

```python
from climb_sensei.progress.attempt_routes import router as attempt_router
# in create_app():
app.include_router(attempt_router)
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_attempt_api.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/climb_sensei/progress/attempt_routes.py tests/test_attempt_api.py app/main.py
git commit -m "feat: add Attempt API endpoints with route progress"
```

### Task 10: Update upload endpoint to support route_id

**Files:**
- Modify: `app/routers/api.py`
- Create: `tests/test_upload_route.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_upload_route.py
"""Test upload endpoint with route_id parameter."""
import pytest
from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from io import BytesIO

from climb_sensei.database.models import Base, User, Route, Attempt
from climb_sensei.database.config import get_db
from climb_sensei.auth import get_current_active_user
from app.main import create_app


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    user = User(id=1, email="test@example.com", hashed_password="fakehash")
    route = Route(
        user_id=1, name="Test", grade="V4",
        grade_system="hueco", type="boulder",
    )
    session.add_all([user, route])
    session.commit()
    yield session
    session.close()


@pytest.fixture
def client(db_session):
    app = create_app()

    def override_get_db():
        yield db_session

    def override_get_user():
        return db_session.query(User).first()

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_current_active_user] = override_get_user
    return TestClient(app)


@patch("app.routers.api.save_upload")
@patch("app.routers.api.create_video_record")
def test_upload_with_route_id(mock_create_video, mock_save, client, db_session):
    """Upload with route_id should include it in the response."""
    mock_save.return_value = MagicMock()
    mock_video = MagicMock()
    mock_video.id = 1
    mock_create_video.return_value = mock_video

    response = client.post(
        "/upload",
        data={"route_id": "1"},
        files={"file": ("test.mp4", BytesIO(b"fake"), "video/mp4")},
    )
    assert response.status_code == 202
    data = response.json()
    assert data["route_id"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_upload_route.py -v`
Expected: FAIL — `route_id` not in response

- [ ] **Step 3: Add route_id parameter to upload endpoint**

In `app/routers/api.py`, modify the `upload_video` function signature — add parameter:

```python
route_id: int = Form(None),
```

Pass `route_id` to the background thread kwargs. In the response, add:

```python
"route_id": route_id,
```

In `_run_analysis_pipeline`, add `route_id` parameter. After `persist_results`, if `route_id` is set, create an `Attempt` record:

```python
if route_id:
    from climb_sensei.database.models import Attempt
    attempt = Attempt(
        route_id=route_id,
        video_id=video_id,
        session_id=session_id,
        analysis_id=db_analysis_id,  # returned from persist_results
        date=datetime.now(timezone.utc),
    )
    db.add(attempt)
    db.commit()
```

Note: `persist_results` needs to return the analysis ID. Check `app/services/upload.py:persist_results` — it currently returns the analysis record ID. Store it and use it for the Attempt.

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_upload_route.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add app/routers/api.py tests/test_upload_route.py
git commit -m "feat: upload endpoint accepts route_id, creates Attempt record"
```

### Task 11: Update session endpoints and goal endpoints

**Files:**
- Modify: `src/climb_sensei/progress/routes.py`
- Modify: `src/climb_sensei/progress/schemas.py`

- [ ] **Step 1: Update session schema — make name optional**

In `src/climb_sensei/progress/schemas.py`, change `ClimbSessionBase`:

```python
class ClimbSessionBase(BaseModel):
    """Base schema for climbing sessions."""

    name: Optional[str] = Field(None, max_length=255)  # Changed: was required
    date: datetime
    location: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None
```

- [ ] **Step 2: Update session create to auto-generate name**

In `src/climb_sensei/progress/routes.py`, update `create_session`:

```python
# Auto-generate name from date if not provided
name = session.name or session.date.strftime("%b %d, %Y")

db_session = ClimbSession(
    user_id=current_user.id,
    name=name,
    # ... rest unchanged
)
```

- [ ] **Step 3: Update goal schema — add route_id**

In `src/climb_sensei/progress/schemas.py`, update `GoalCreate`:

```python
class GoalCreate(GoalBase):
    """Schema for creating a goal."""

    route_id: Optional[int] = None
```

Update `GoalResponse`:

```python
class GoalResponse(GoalBase):
    # ... existing fields ...
    route_id: Optional[int] = None
```

- [ ] **Step 4: Update goal create endpoint — pass route_id**

In `src/climb_sensei/progress/routes.py`, update `create_goal`:

```python
db_goal = Goal(
    user_id=current_user.id,
    route_id=goal.route_id,  # Add this line
    # ... rest unchanged
)
```

- [ ] **Step 5: Add route_id filter to list_goals**

In `src/climb_sensei/progress/routes.py`, update `list_goals` — add parameter:

```python
route_id: int = Query(None, description="Filter by route"),
```

Add filter:

```python
if route_id:
    query = query.filter(Goal.route_id == route_id)
```

- [ ] **Step 6: Run existing tests to verify no regressions**

Run: `uv run pytest tests/ -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add src/climb_sensei/progress/routes.py src/climb_sensei/progress/schemas.py
git commit -m "feat: session name optional, goal route_id support"
```

---

## Chunk 3: Frontend — Base Layout & Navigation

### Task 12: Rebuild base.html with bottom tab navigation

**Files:**
- Modify: `app/templates/base.html`

- [ ] **Step 1: Read current base.html to understand the full structure**

Read: `app/templates/base.html`

- [ ] **Step 2: Rewrite base.html**

Keep the existing CSS custom properties and design tokens. Replace the top navbar with a bottom tab bar. Key changes:

- Remove top sticky navbar
- Add fixed bottom tab bar with 4 tabs (Routes, Upload, Sessions, Profile)
- Upload button is elevated center FAB
- Keep existing alert system, auth JS
- Add `<meta name="viewport">` for mobile
- Ensure min 44px touch targets
- Page content gets `padding-bottom: 70px` to clear the tab bar

The bottom nav HTML structure:

```html
<nav class="bottom-nav">
    <a href="/" class="nav-tab {% if active_tab == 'routes' %}active{% endif %}">
        <svg><!-- climbing icon --></svg>
        <span>Routes</span>
    </a>
    <a href="/upload" class="nav-tab nav-tab-fab">
        <div class="fab">+</div>
        <span>Upload</span>
    </a>
    <a href="/sessions" class="nav-tab {% if active_tab == 'sessions' %}active{% endif %}">
        <svg><!-- calendar icon --></svg>
        <span>Sessions</span>
    </a>
    <a href="/profile" class="nav-tab {% if active_tab == 'profile' %}active{% endif %}">
        <svg><!-- user icon --></svg>
        <span>Profile</span>
    </a>
</nav>
```

CSS for bottom nav:

```css
.bottom-nav {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background: var(--bg-raised);
    border-top: 1px solid var(--border);
    display: flex;
    justify-content: space-around;
    align-items: center;
    padding: 8px 0 env(safe-area-inset-bottom, 8px);
    z-index: 100;
}
.nav-tab {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    font-size: 0.65rem;
    color: var(--text-muted);
    text-decoration: none;
    min-width: 64px;
    min-height: 44px;
    justify-content: center;
}
.nav-tab.active { color: var(--accent); }
.nav-tab-fab .fab {
    width: 48px;
    height: 48px;
    background: linear-gradient(135deg, var(--accent), #ef4444);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 800;
    color: #000;
    margin-top: -16px;
    box-shadow: var(--shadow-lg);
}
```

Desktop responsive: at `min-width: 768px`, move nav to a left sidebar or top bar.

- [ ] **Step 3: Update pages.py to pass active_tab context**

In `app/routers/pages.py`, add `active_tab` to each template render:

```python
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("routes.html", {"request": request, "active_tab": "routes"})
```

- [ ] **Step 4: Test manually**

Run: `python run_app.py`
Open in browser. Verify:
- Bottom tab bar appears
- Tabs navigate between pages
- FAB button is centered and elevated
- Active tab is highlighted
- Content doesn't overlap with nav

- [ ] **Step 5: Commit**

```bash
git add app/templates/base.html app/routers/pages.py
git commit -m "feat: rebuild base layout with bottom tab navigation"
```

### Task 13: Create Routes page (landing)

**Files:**
- Create: `app/templates/routes.html`
- Modify: `app/routers/pages.py`

- [ ] **Step 1: Create routes.html template**

Extends `base.html`. Contains:
- Search bar input
- Type filter chips (All / Boulder / Sport / Trad)
- Sort control dropdown (Recent / Grade / Attempts)
- Route list container (populated by JS)
- Empty state
- JS: fetch `/api/routes`, render compact list items with grade badges, sparklines, attempt count

Each route list item HTML:

```html
<a href="/routes/${route.id}" class="route-item ${route.status === 'sent' ? 'route-sent' : ''}">
    <div class="grade-badge grade-${gradeColor(route.grade)}">${route.grade}</div>
    <div class="route-info">
        <div class="route-name">${route.name}</div>
        <div class="route-meta">${route.location || 'No location'} · ${route.attempt_count} attempts</div>
    </div>
    <div class="sparkline">${renderSparkline(route.sparkline)}</div>
</a>
```

Sparkline: render as inline SVG polyline from the `sparkline` array values.

- [ ] **Step 2: Add page route**

In `app/routers/pages.py`, change `/` to render `routes.html`:

```python
@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("routes.html", {"request": request, "active_tab": "routes"})
```

Add route detail page:

```python
@router.get("/routes/{route_id}", response_class=HTMLResponse)
async def route_detail(request: Request, route_id: int):
    return templates.TemplateResponse("route_detail.html", {"request": request, "active_tab": "routes", "route_id": route_id})
```

- [ ] **Step 3: Test manually**

Verify routes page loads, filters work, empty state shows for new users.

- [ ] **Step 4: Commit**

```bash
git add app/templates/routes.html app/routers/pages.py
git commit -m "feat: add Routes landing page with compact list"
```

### Task 14: Create Route Detail page

**Files:**
- Create: `app/templates/route_detail.html`

- [ ] **Step 1: Create route_detail.html**

Extends `base.html`. Contains:
- Back link ("← Routes")
- Route header (name, grade badge, location, type)
- Quick stats row (attempts, first try, last try, status)
- Metric chart section (selectable dropdown, Chart.js bar chart)
- Attempt timeline (vertical timeline, most recent first)
- "Add attempt" button → links to `/upload?route_id=${id}`
- Edit/delete route actions (in a ... menu)
- Goal section (set/view per-route goals)

JS flow:
1. Fetch `/api/routes/${routeId}` for route data
2. Fetch `/api/routes/${routeId}/attempts` for timeline
3. Fetch `/api/routes/${routeId}/progress/avg_velocity` for default chart
4. Render timeline and chart

Chart.js config for metric trend:

```javascript
new Chart(ctx, {
    type: 'bar',
    data: {
        labels: data.data.map(d => new Date(d.date).toLocaleDateString()),
        datasets: [{
            data: data.data.map(d => d.value),
            backgroundColor: 'rgba(245, 158, 11, 0.6)',
            borderRadius: 4,
        }]
    },
    options: {
        responsive: true,
        plugins: { legend: { display: false } },
        scales: {
            x: { grid: { display: false }, ticks: { color: '#64748b' } },
            y: { grid: { color: '#1a2332' }, ticks: { color: '#64748b' } },
        }
    }
});
```

- [ ] **Step 2: Test manually**

Create a route via API, add an attempt. Navigate to route detail. Verify:
- Header shows correct info
- Timeline renders attempts
- Chart shows metric trend
- "Add attempt" links to upload with route pre-filled

- [ ] **Step 3: Commit**

```bash
git add app/templates/route_detail.html
git commit -m "feat: add Route Detail page with timeline and metric chart"
```

---

## Chunk 4: Frontend — Upload, Sessions, Profile, Auth

### Task 15: Rebuild Upload page (context-aware)

**Files:**
- Modify: `app/templates/upload.html`

- [ ] **Step 1: Rewrite upload.html**

Keep the existing upload/analysis JS logic but restructure the form:
- If URL has `?route_id=X`, fetch route info and show pre-filled route card with "Change" button
- If no route_id, show route selector dropdown (fetches from `/api/routes`, includes "Create new route" option)
- Session defaults to today (auto-creates on submit)
- Analysis toggle chips (Metrics on by default, Annotated Video, Quality Check)
- "Analyze Attempt" submit button
- Progress indicator (reuse existing polling logic)
- After completion: link to `/routes/${routeId}/attempts/${attemptId}` or show inline

- [ ] **Step 2: Test manually**

- Visit `/upload` directly → route selector shown
- Visit `/upload?route_id=1` → route pre-filled
- Upload a video → verify attempt is created and linked

- [ ] **Step 3: Commit**

```bash
git add app/templates/upload.html
git commit -m "feat: rebuild Upload page with context-aware route selection"
```

### Task 16: Rebuild Sessions page (calendar)

**Files:**
- Modify: `app/templates/sessions.html`

- [ ] **Step 1: Rewrite sessions.html**

Replace session list with:
- Mini calendar (month view, CSS grid 7 columns)
- Navigation arrows for months
- Dots on days with sessions (fetch `/api/sessions`, group by date)
- Clicking a day shows day detail below:
  - Date heading
  - List of attempts for that day's session (fetch session detail, show routes with grade badges)
  - Location, notes (editable)
- Today highlighted with accent background

Calendar rendering: pure JS, no library. Build a 7×6 grid of day cells, mark days with sessions.

- [ ] **Step 2: Test manually**

Create a session with some analyses/attempts. Verify calendar shows dots, clicking a day reveals details.

- [ ] **Step 3: Commit**

```bash
git add app/templates/sessions.html
git commit -m "feat: rebuild Sessions page with calendar view"
```

### Task 17: Create Profile page

**Files:**
- Create: `app/templates/profile.html`
- Modify: `app/routers/pages.py`

- [ ] **Step 1: Create profile.html**

Extends `base.html`. Sections:
- **Stats section:** Total routes, sessions, attempts (fetch counts from API). Climbing frequency (sessions per week). Member since date.
- **Grade pyramid:** Fetch all routes, group by type + grade, render horizontal bar chart showing count per grade. Separate sections for boulder and sport grades.
- **Goals overview:** Fetch `/api/goals`, show all active goals with route name, target, current value, progress bar, deadline. Tapping → navigate to route. Toggle for Active/Achieved.
- **Settings:** Name, email (read-only), password change link, Google OAuth status, logout button.

- [ ] **Step 2: Add page route**

```python
@router.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    return templates.TemplateResponse("profile.html", {"request": request, "active_tab": "profile"})
```

- [ ] **Step 3: Test manually**

Verify stats load, grade pyramid renders, goals show, settings section appears.

- [ ] **Step 4: Commit**

```bash
git add app/templates/profile.html app/routers/pages.py
git commit -m "feat: add Profile page with stats, grade pyramid, and goals overview"
```

### Task 18: Visual refresh for Login & Register pages

**Files:**
- Modify: `app/templates/login.html`
- Modify: `app/templates/register.html`

- [ ] **Step 1: Update login.html**

Keep existing form logic. Visual changes:
- Center the form vertically
- App logo/name at top
- Cleaner card styling matching new design system
- Larger touch targets for inputs (min 44px height)
- Hide bottom nav on auth pages (add `hide-nav` class to body)

- [ ] **Step 2: Update register.html**

Same visual treatment as login.

- [ ] **Step 3: Test manually**

Verify login and register work, look consistent with new design.

- [ ] **Step 4: Commit**

```bash
git add app/templates/login.html app/templates/register.html
git commit -m "feat: visual refresh for auth pages"
```

### Task 19: Create Attempt Detail page

**Files:**
- Create: `app/templates/attempt_detail.html`
- Modify: `app/routers/pages.py`

- [ ] **Step 1: Create attempt_detail.html**

Extends `base.html`. Components:
- Back link ("← Route Name")
- Video player (if annotated video available, use `<video>` tag with `/outputs/` path)
- Metrics grid organized by category (reuse category structure from existing upload results):
  - Movement: velocity, height, distance
  - Stability: sway, jerk
  - Efficiency: movement economy, lock-offs, rests
  - Biomechanics: joint angles
- Comparison bar: for each metric, show current value + delta from previous attempt (↑ green / ↓ red / → gray)
- Previous/Next attempt navigation arrows
- Editable notes field (PATCH on blur)

JS flow:
1. Fetch `/api/routes/${routeId}/attempts/${attemptId}`
2. Render video player, metrics with deltas
3. Next/prev arrows: link to adjacent attempt IDs

- [ ] **Step 2: Add page route**

```python
@router.get("/routes/{route_id}/attempts/{attempt_id}", response_class=HTMLResponse)
async def attempt_detail(request: Request, route_id: int, attempt_id: int):
    return templates.TemplateResponse("attempt_detail.html", {
        "request": request, "active_tab": "routes",
        "route_id": route_id, "attempt_id": attempt_id,
    })
```

- [ ] **Step 3: Test manually**

- [ ] **Step 4: Commit**

```bash
git add app/templates/attempt_detail.html app/routers/pages.py
git commit -m "feat: add Attempt Detail page with video player and metric deltas"
```

### Task 20: Clean up deprecated files

**Files:**
- Delete: `app/static/style.css` (unused light theme)
- Delete: `app/templates/dashboard.html` (replaced by routes page)
- Delete: `app/templates/progress.html` (replaced by route detail + profile)
- Delete: `app/templates/index.html` (if exists, replaced by routes.html)
- Modify: `app/routers/pages.py` — remove old routes, add redirects

- [ ] **Step 1: Add redirects for old URLs**

```python
from fastapi.responses import RedirectResponse

@router.get("/dashboard")
async def dashboard_redirect():
    return RedirectResponse(url="/", status_code=301)

@router.get("/progress")
async def progress_redirect():
    return RedirectResponse(url="/", status_code=301)
```

- [ ] **Step 2: Delete unused files**

- [ ] **Step 3: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All pass (update any tests that reference deleted pages).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore: remove deprecated templates and add redirects"
```

### Task 21: Data migration script

**Files:**
- Create: `scripts/migrate_to_routes.py`

- [ ] **Step 1: Write migration script**

```python
#!/usr/bin/env python
"""Migrate existing analyses to the route-centric model.

For each user:
1. Create an "Uncategorized" route
2. For each existing Analysis, create an Attempt linking:
   - Attempt.video_id = Analysis.video_id
   - Attempt.analysis_id = Analysis.id
   - Attempt.session_id = Analysis.session_id
   - Attempt.route_id = uncategorized_route.id
   - Attempt.date = Analysis.created_at
"""

from climb_sensei.database.config import SessionLocal
from climb_sensei.database.models import User, Analysis, Route, Attempt, Video


def migrate():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        for user in users:
            # Check if user already has an Uncategorized route
            existing = (
                db.query(Route)
                .filter(Route.user_id == user.id, Route.name == "Uncategorized")
                .first()
            )
            if existing:
                print(f"User {user.id}: already has Uncategorized route, skipping")
                continue

            # Get all analyses for this user
            analyses = (
                db.query(Analysis)
                .join(Video)
                .filter(Video.user_id == user.id)
                .all()
            )

            if not analyses:
                print(f"User {user.id}: no analyses, skipping")
                continue

            # Create uncategorized route
            route = Route(
                user_id=user.id,
                name="Uncategorized",
                grade="?",
                grade_system="hueco",
                type="boulder",
                status="projecting",
            )
            db.add(route)
            db.flush()

            # Create attempts for each analysis
            for analysis in analyses:
                attempt = Attempt(
                    route_id=route.id,
                    video_id=analysis.video_id,
                    session_id=analysis.session_id,
                    analysis_id=analysis.id,
                    date=analysis.created_at,
                )
                db.add(attempt)

            db.commit()
            print(
                f"User {user.id}: migrated {len(analyses)} analyses "
                f"to Uncategorized route"
            )

    finally:
        db.close()


if __name__ == "__main__":
    migrate()
```

- [ ] **Step 2: Test with existing database**

Run: `uv run python scripts/migrate_to_routes.py`
Verify: Uncategorized route created, attempts link to existing analyses.

- [ ] **Step 3: Commit**

```bash
git add scripts/migrate_to_routes.py
git commit -m "feat: add data migration script for route-centric model"
```

### Task 22: Final integration test

**Files:**
- Modify: `tests/test_e2e.py` (if applicable)

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest tests/ -v`
Expected: All pass.

- [ ] **Step 2: Run linter**

Run: `make check`
Expected: No issues.

- [ ] **Step 3: Manual smoke test**

1. Start app: `python run_app.py`
2. Register/login
3. Verify Routes page (landing) loads
4. Create a route via the upload flow
5. Upload a video assigned to that route
6. Verify route detail shows the attempt
7. Check Sessions calendar shows today
8. Check Profile shows stats and goals
9. Test on mobile viewport (Chrome DevTools)

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "test: integration verification for UI/UX redesign"
```
