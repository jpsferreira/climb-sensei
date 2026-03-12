# Platform Evolution Plan: User Accounts & Persistent Data

## Overview

Evolution roadmap for transforming ClimbSensei from a standalone analysis tool into a full platform with user accounts, persistent data storage, and progress tracking capabilities.

**Approach**: Stay with FastAPI + Add Database (Don't switch to Django)

**Why FastAPI + Database?**

- ✅ Keep excellent service architecture
- ✅ Keep performance for video processing
- ✅ Add user accounts & data persistence
- ✅ Keep auto-generated API docs
- ✅ Modern async stack throughout

---

## Tech Stack

```
Frontend:    React/Vue (SPA) or htmx (simpler option)
API:         FastAPI (current - keep it)
Database:    PostgreSQL (production) / SQLite (development)
ORM:         SQLAlchemy 2.0
Migrations:  Alembic
Auth:        JWT tokens (python-jose)
Storage:     Local filesystem or S3 for videos
Cache:       Redis (optional, for sessions/rate limiting)
Queue:       Celery (optional, for async video processing)
```

---

## Phase 1: Database Foundation (Week 1-2)

### Goals

- Set up database infrastructure
- Create core models
- Configure migrations
- Test database operations

### Dependencies to Add

```toml
# pyproject.toml
[project]
dependencies = [
    # ... existing dependencies ...
    "sqlalchemy >= 2.0",
    "alembic >= 1.13",
    "psycopg2-binary >= 2.9",  # PostgreSQL driver
    # or "asyncpg >= 0.29" for async PostgreSQL
]
```

### Database Models

```python
# src/climb_sensei/database/models.py

from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

    # Relationships
    videos = relationship("Video", back_populates="user")
    sessions = relationship("ClimbSession", back_populates="user")

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    duration_seconds = Column(Float)
    fps = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    status = Column(String(50), default="uploaded")  # uploaded, processing, completed, failed

    # Relationships
    user = relationship("User", back_populates="videos")
    analyses = relationship("Analysis", back_populates="video")

class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(Integer, ForeignKey("videos.id"), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Store analysis results as JSON
    summary = Column(JSON)  # ClimbingSummary.to_dict()
    history = Column(JSON)  # Full metrics history
    video_quality = Column(JSON)  # VideoQualityReport
    tracking_quality = Column(JSON)  # TrackingQualityReport

    # Optional: denormalize key metrics for querying
    total_frames = Column(Integer)
    avg_velocity = Column(Float)
    max_height = Column(Float)

    # Relationships
    video = relationship("Video", back_populates="analyses")

class ClimbSession(Base):
    __tablename__ = "climb_sessions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String(255))
    date = Column(DateTime)
    location = Column(String(255))
    notes = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="sessions")
```

### Database Configuration

```python
# src/climb_sensei/database/config.py

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://user:password@localhost/climbsensei"
)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

### Alembic Setup

```bash
# Initialize Alembic
alembic init alembic

# Create first migration
alembic revision --autogenerate -m "Initial database schema"

# Apply migrations
alembic upgrade head
```

### Tasks

- [ ] Install dependencies
- [ ] Create database models
- [ ] Configure Alembic
- [ ] Create initial migration
- [ ] Test database connection
- [ ] Set up development PostgreSQL or SQLite

---

## Phase 2: User Authentication (Week 2-3)

### Goals

- Implement user registration and login
- Add JWT token authentication
- Protect existing routes
- Create user management endpoints

### Dependencies to Add

```toml
[project]
dependencies = [
    # ... existing ...
    "python-jose[cryptography] >= 3.3",
    "passlib[bcrypt] >= 1.7",
    "python-multipart >= 0.0.6",
]
```

### Authentication Utilities

```python
# src/climb_sensei/auth/utils.py

from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

SECRET_KEY = "your-secret-key-here"  # Use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user
```

### Authentication Routes

```python
# app/api/auth.py

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from datetime import timedelta

router = APIRouter(prefix="/auth", tags=["Authentication"])

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str | None
    created_at: datetime

class Token(BaseModel):
    access_token: str
    token_type: str

@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate, db: Session = Depends(get_db)):
    # Check if user exists
    existing = db.query(User).filter(User.email == user_data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    # Create user
    user = User(
        email=user_data.email,
        password_hash=get_password_hash(user_data.password),
        full_name=user_data.full_name
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    access_token = create_access_token(
        data={"sub": user.email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    return current_user
```

### Protect Existing Routes

```python
# app/main.py - Update upload route

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    run_metrics: bool = Form(True),
    run_video: bool = Form(False),
    run_quality: bool = Form(True),
    current_user: User = Depends(get_current_user),  # ← ADD THIS
    db: Session = Depends(get_db),  # ← ADD THIS
):
    # ... existing code ...
```

### Tasks

- [ ] Implement password hashing
- [ ] Create JWT token system
- [ ] Add register endpoint
- [ ] Add login endpoint
- [ ] Add get_current_user dependency
- [ ] Protect existing routes
- [ ] Test authentication flow
- [ ] Add logout (token blacklist optional)

---

## Phase 3: Persistent Analysis Storage (Week 3-4)

### Goals

- Save analysis results to database
- Link analyses to users and videos
- Enable retrieval of past analyses
- Add analysis management endpoints

### Updated Upload Route

```python
# app/main.py

@app.post("/upload")
async def upload_video(
    file: UploadFile = File(...),
    run_metrics: bool = Form(True),
    run_video: bool = Form(False),
    run_quality: bool = Form(True),
    dashboard_position: str = Form("right"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    # Generate unique ID
    analysis_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"
    with open(upload_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create video record in database
    video_record = Video(
        user_id=current_user.id,
        filename=file.filename,
        file_path=str(upload_path),
        status="processing"
    )
    db.add(video_record)
    db.commit()
    db.refresh(video_record)

    try:
        # ... existing analysis code ...

        # Update video metadata
        video_record.fps = fps
        video_record.duration_seconds = len(landmarks_history) / fps
        video_record.status = "completed"

        # Save analysis to database
        analysis_record = Analysis(
            video_id=video_record.id,
            summary=analysis.summary.to_dict() if analysis else None,
            history=analysis.history if analysis else None,
            video_quality=video_quality_report.to_dict() if video_quality_report else None,
            tracking_quality=tracking_quality_report.to_dict() if tracking_quality_report else None,
            total_frames=analysis.summary.total_frames if analysis else 0,
            avg_velocity=analysis.summary.avg_velocity if analysis else 0,
            max_height=analysis.summary.max_height if analysis else 0,
        )
        db.add(analysis_record)
        db.commit()

        # Return with database IDs
        results["video_id"] = video_record.id
        results["analysis_id"] = analysis_record.id

    except Exception as e:
        video_record.status = "failed"
        db.commit()
        raise

    return JSONResponse(content=results)
```

### Analysis Management Routes

```python
# app/api/analyses.py

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

router = APIRouter(prefix="/analyses", tags=["Analyses"])

@router.get("/")
async def list_analyses(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all analyses for current user."""
    analyses = db.query(Analysis).join(Video).filter(
        Video.user_id == current_user.id
    ).offset(skip).limit(limit).all()

    return [{
        "id": a.id,
        "video_id": a.video_id,
        "created_at": a.created_at,
        "total_frames": a.total_frames,
        "avg_velocity": a.avg_velocity,
        "max_height": a.max_height,
    } for a in analyses]

@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific analysis with full details."""
    analysis = db.query(Analysis).join(Video).filter(
        Analysis.id == analysis_id,
        Video.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    return {
        "id": analysis.id,
        "video_id": analysis.video_id,
        "created_at": analysis.created_at,
        "summary": analysis.summary,
        "history": analysis.history,
        "video_quality": analysis.video_quality,
        "tracking_quality": analysis.tracking_quality,
    }

@router.delete("/{analysis_id}")
async def delete_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an analysis."""
    analysis = db.query(Analysis).join(Video).filter(
        Analysis.id == analysis_id,
        Video.user_id == current_user.id
    ).first()

    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    db.delete(analysis)
    db.commit()
    return {"message": "Analysis deleted"}

@router.get("/videos/{video_id}/analyses")
async def list_video_analyses(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all analyses for a specific video."""
    video = db.query(Video).filter(
        Video.id == video_id,
        Video.user_id == current_user.id
    ).first()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    analyses = db.query(Analysis).filter(Analysis.video_id == video_id).all()
    return analyses
```

### Video Management Routes

```python
# app/api/videos.py

router = APIRouter(prefix="/videos", tags=["Videos"])

@router.get("/")
async def list_videos(
    skip: int = 0,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all videos for current user."""
    videos = db.query(Video).filter(
        Video.user_id == current_user.id
    ).offset(skip).limit(limit).all()
    return videos

@router.get("/{video_id}")
async def get_video(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get specific video details."""
    video = db.query(Video).filter(
        Video.id == video_id,
        Video.user_id == current_user.id
    ).first()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")
    return video

@router.delete("/{video_id}")
async def delete_video(
    video_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete video and all associated analyses."""
    video = db.query(Video).filter(
        Video.id == video_id,
        Video.user_id == current_user.id
    ).first()

    if not video:
        raise HTTPException(status_code=404, detail="Video not found")

    # Delete file from filesystem
    if Path(video.file_path).exists():
        Path(video.file_path).unlink()

    # Delete from database (cascades to analyses)
    db.delete(video)
    db.commit()
    return {"message": "Video deleted"}
```

### Tasks

- [ ] Update upload route to save to database
- [ ] Create analysis listing endpoint
- [ ] Create analysis retrieval endpoint
- [ ] Create analysis deletion endpoint
- [ ] Create video management endpoints
- [ ] Test CRUD operations
- [ ] Update frontend to use new endpoints
- [ ] Add pagination support

---

## Phase 4: Progress Tracking (Week 4-6)

### Goals

- Track metrics over time
- Enable comparison between analyses
- Visualize progress
- Set and track goals

### Additional Models

```python
# Add to models.py

class ProgressMetric(Base):
    __tablename__ = "progress_metrics"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    metric_name = Column(String(100))  # avg_velocity, lock_off_count, etc.
    value = Column(Float)
    recorded_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User")
    analysis = relationship("Analysis")

class Goal(Base):
    __tablename__ = "goals"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    metric_name = Column(String(100))
    target_value = Column(Float)
    current_value = Column(Float)
    deadline = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    achieved = Column(Boolean, default=False)
    achieved_at = Column(DateTime)

    user = relationship("User")
```

### Progress Tracking Routes

```python
# app/api/progress.py

@router.get("/progress/{metric_name}")
async def get_metric_progress(
    metric_name: str,
    days: int = 30,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get progress for a specific metric over time."""
    cutoff = datetime.utcnow() - timedelta(days=days)

    metrics = db.query(ProgressMetric).filter(
        ProgressMetric.user_id == current_user.id,
        ProgressMetric.metric_name == metric_name,
        ProgressMetric.recorded_at >= cutoff
    ).order_by(ProgressMetric.recorded_at).all()

    return {
        "metric": metric_name,
        "data": [{
            "date": m.recorded_at,
            "value": m.value,
            "analysis_id": m.analysis_id
        } for m in metrics]
    }

@router.post("/compare")
async def compare_analyses(
    analysis_ids: list[int],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Compare multiple analyses side by side."""
    analyses = db.query(Analysis).join(Video).filter(
        Analysis.id.in_(analysis_ids),
        Video.user_id == current_user.id
    ).all()

    if len(analyses) != len(analysis_ids):
        raise HTTPException(status_code=404, detail="Some analyses not found")

    comparison = []
    for analysis in analyses:
        comparison.append({
            "id": analysis.id,
            "date": analysis.created_at,
            "metrics": analysis.summary
        })

    return {"comparisons": comparison}

@router.post("/goals")
async def create_goal(
    metric_name: str,
    target_value: float,
    deadline: datetime,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new training goal."""
    goal = Goal(
        user_id=current_user.id,
        metric_name=metric_name,
        target_value=target_value,
        deadline=deadline
    )
    db.add(goal)
    db.commit()
    return goal

@router.get("/goals")
async def list_goals(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all goals for user."""
    goals = db.query(Goal).filter(Goal.user_id == current_user.id).all()
    return goals
```

### Tasks

- [ ] Create progress tracking models
- [ ] Auto-record metrics after analysis
- [ ] Create progress retrieval endpoints
- [ ] Create comparison endpoint
- [ ] Add goal setting/tracking
- [ ] Build progress visualization UI
- [ ] Add dashboard with charts

---

## Phase 5: Social & Sharing Features (Week 6-8)

### Goals

- Share analyses with others
- Public/private visibility
- Comments and feedback
- Community features

### Additional Models

```python
class AnalysisShare(Base):
    __tablename__ = "analysis_shares"

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    shared_by = Column(Integer, ForeignKey("users.id"))
    shared_with = Column(Integer, ForeignKey("users.id"), nullable=True)  # null = public
    share_token = Column(String(100), unique=True)  # for public links
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=True)

    analysis = relationship("Analysis")

class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True)
    analysis_id = Column(Integer, ForeignKey("analyses.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(String(1000))
    created_at = Column(DateTime, default=datetime.utcnow)

    analysis = relationship("Analysis")
    user = relationship("User")
```

### Tasks

- [ ] Add visibility settings to analyses
- [ ] Create sharing system
- [ ] Generate shareable links
- [ ] Add comment system
- [ ] Build social feed (optional)
- [ ] Add like/favorite features

---

## Phase 6: Advanced Features (Week 8+)

### Performance Optimizations

- [ ] Add Redis caching for frequent queries
- [ ] Implement Celery for async video processing
- [ ] Add pagination to all list endpoints
- [ ] Optimize database queries (indexes, eager loading)

### Storage Improvements

- [ ] Move videos to S3/cloud storage
- [ ] Add video streaming
- [ ] Implement CDN for output videos
- [ ] Add cleanup for old files

### Analytics

- [ ] User activity tracking
- [ ] Platform-wide statistics
- [ ] Popular metrics dashboard
- [ ] Export data to CSV/Excel

### Mobile & API

- [ ] Build React Native mobile app
- [ ] Add push notifications
- [ ] Offline support
- [ ] Native video recording

---

## Database Schema (Complete)

```sql
-- Users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT TRUE,
    is_superuser BOOLEAN DEFAULT FALSE
);

-- Videos
CREATE TABLE videos (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    uploaded_at TIMESTAMP DEFAULT NOW(),
    duration_seconds FLOAT,
    fps FLOAT,
    width INTEGER,
    height INTEGER,
    status VARCHAR(50) DEFAULT 'uploaded'
);

-- Analyses
CREATE TABLE analyses (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    created_at TIMESTAMP DEFAULT NOW(),
    summary JSONB,
    history JSONB,
    video_quality JSONB,
    tracking_quality JSONB,
    total_frames INTEGER,
    avg_velocity FLOAT,
    max_height FLOAT
);

-- Progress Metrics
CREATE TABLE progress_metrics (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    value FLOAT NOT NULL,
    recorded_at TIMESTAMP DEFAULT NOW()
);

-- Goals
CREATE TABLE goals (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    metric_name VARCHAR(100) NOT NULL,
    target_value FLOAT NOT NULL,
    current_value FLOAT,
    deadline TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    achieved BOOLEAN DEFAULT FALSE,
    achieved_at TIMESTAMP
);

-- Climb Sessions
CREATE TABLE climb_sessions (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255),
    date TIMESTAMP,
    location VARCHAR(255),
    notes VARCHAR(1000),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Analysis Shares
CREATE TABLE analysis_shares (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
    shared_by INTEGER REFERENCES users(id),
    shared_with INTEGER REFERENCES users(id),
    share_token VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP
);

-- Comments
CREATE TABLE comments (
    id SERIAL PRIMARY KEY,
    analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),
    text VARCHAR(1000),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_videos_user ON videos(user_id);
CREATE INDEX idx_analyses_video ON analyses(video_id);
CREATE INDEX idx_progress_user_metric ON progress_metrics(user_id, metric_name);
CREATE INDEX idx_goals_user ON goals(user_id);
```

---

## Migration Strategy

### Backward Compatibility

Keep existing functionality working during migration:

```python
# Support both old (file-based) and new (database) modes
@app.post("/upload")
async def upload_video(
    file: UploadFile,
    current_user: User = Depends(get_current_user_optional),  # Optional for backward compat
    db: Session = Depends(get_db_optional),
):
    if current_user and db:
        # New path: Save to database
        return upload_with_database(file, current_user, db)
    else:
        # Old path: File-based only (deprecated)
        return upload_legacy(file)
```

### Gradual Rollout

1. **Week 1-2**: Database infrastructure (no user-facing changes)
2. **Week 3**: Add auth (optional - users can still use without login)
3. **Week 4**: Soft launch database storage (logged-in users only)
4. **Week 5+**: Encourage migration, add premium features for accounts
5. **Week 8+**: Deprecate anonymous usage (optional)

---

## Environment Variables

```bash
# .env
DATABASE_URL=postgresql://user:password@localhost/climbsensei
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Optional
REDIS_URL=redis://localhost:6379
S3_BUCKET=climbsensei-videos
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

---

## Testing Strategy

### Unit Tests

- Auth utilities (password hashing, JWT)
- Database models (CRUD operations)
- Service layer (business logic)

### Integration Tests

- Full authentication flow
- Video upload → analysis → retrieval
- Progress tracking over multiple analyses

### Performance Tests

- Database query performance
- Concurrent user handling
- Video processing throughput

---

## Deployment Considerations

### Infrastructure

- PostgreSQL database (RDS on AWS, or managed PostgreSQL)
- Redis for caching/sessions
- S3 for video storage
- Docker containers for application
- Kubernetes/ECS for orchestration

### Monitoring

- Application logs (structlog)
- Database performance (pg_stat_statements)
- API metrics (Prometheus + Grafana)
- Error tracking (Sentry)

### Security

- HTTPS only
- Rate limiting
- SQL injection prevention (SQLAlchemy ORM)
- XSS protection
- CORS configuration
- Regular security audits

---

## Success Metrics

### Phase 1-2 (Weeks 1-3)

- ✅ Database infrastructure operational
- ✅ 100% test coverage for auth
- ✅ User registration/login working

### Phase 3-4 (Weeks 3-6)

- ✅ All analyses saved to database
- ✅ Users can view analysis history
- ✅ Progress tracking functional
- 🎯 Target: 50+ registered users

### Phase 5-6 (Weeks 6-8+)

- ✅ Sharing features live
- ✅ Mobile app released
- 🎯 Target: 200+ active users
- 🎯 Target: 1000+ analyses stored

---

## Next Steps

1. **Immediate (This week)**

   - Set up local PostgreSQL database
   - Install SQLAlchemy and Alembic
   - Create initial models
   - Test database connection

2. **Short term (Next 2 weeks)**

   - Complete Phase 1 (Database Foundation)
   - Start Phase 2 (Authentication)
   - Create migration plan

3. **Medium term (Month 1-2)**

   - Complete Phases 2-3
   - Soft launch to beta users
   - Gather feedback

4. **Long term (Month 3+)**
   - Roll out advanced features
   - Build mobile apps
   - Scale infrastructure

---

## Resources & Documentation

- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Alembic Tutorial](https://alembic.sqlalchemy.org/en/latest/tutorial.html)
- [FastAPI Security](https://fastapi.tiangolo.com/tutorial/security/)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)
- [PostgreSQL Performance](https://www.postgresql.org/docs/current/performance-tips.html)
