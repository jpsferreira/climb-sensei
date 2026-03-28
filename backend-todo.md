# Backend Improvements - Todo List

## Security

### S1: Enforce AUTH_DISABLED in production
- [ ] Raise exception (not just warn) if AUTH_DISABLED=1 in production environment
- [ ] Add startup validation in `app/main.py`
- **File**: `src/climb_sensei/auth/__init__.py:100-104`
- **Severity**: HIGH

### S2: Rate-limit auth endpoints
- [ ] Add rate limiting to `/api/auth/register` and `/api/auth/jwt/login`
- [ ] Prevent brute-force login and registration spam
- **File**: `src/climb_sensei/auth/` (uses fastapi-users defaults, no limits)
- **Severity**: HIGH

### S3: Make OAuth callback URL configurable
- [ ] Replace hardcoded `http://localhost:8000` callback with env variable
- [ ] Add HTTPS redirect middleware for production
- **File**: `src/climb_sensei/auth/users.py:64-65`
- **Severity**: MEDIUM

## Performance

### P1: Fix N+1 queries in session listing
- [ ] Add `joinedload(ClimbSession.attempts).joinedload(Attempt.route)` to `list_sessions()`
- [ ] Each session currently triggers separate queries for attempts and routes
- **File**: `src/climb_sensei/progress/routes.py:454-510`
- **Severity**: HIGH

### P2: Add missing database indexes
- [ ] `Analysis.created_at` — time-range queries
- [ ] `Attempt.date` — frequent sorting/filtering
- [ ] `Goal.metric_name` — frequent filtering
- [ ] Generate Alembic migration for new indexes
- **File**: `src/climb_sensei/database/models.py`
- **Severity**: HIGH

### P3: Optimize video frame buffering
- [ ] Stream frames instead of buffering entire `landmarks_history` and `pose_results_history` in memory
- [ ] Large videos (1000+ frames) consume significant memory
- **File**: `app/services/upload.py:160-184`
- **Severity**: HIGH

### P4: Reuse PoseEngine instance
- [ ] Create PoseEngine once per app lifetime, not per analysis
- [ ] Use dependency injection or module-level singleton
- **File**: `app/services/upload.py:165`
- **Severity**: MEDIUM

### P5: Optimize sparkline queries
- [ ] `_build_sparkline()` runs per route (N extra queries for N routes)
- [ ] Build all sparklines in a single query using window functions
- **File**: `src/climb_sensei/progress/route_routes.py:49-72`
- **Severity**: MEDIUM

### P6: Avoid re-reading video for annotation
- [ ] `generate_annotated_video()` loads all frames again from disk
- [ ] Accept pre-extracted `pose_results_history` instead of re-reading
- **File**: `app/services/upload.py:498-551`
- **Severity**: LOW

## Architecture & Code Quality

### A1: Extract shared upload logic
- [ ] Deduplicate upload-and-analyze code across `video_quality.py`, `tracking_quality.py`, and `api.py`
- [ ] Create shared utility for file validation + temporary file handling
- **Files**: `app/api/video_quality.py:32-182`, `app/api/tracking_quality.py:96-160`
- **Severity**: MEDIUM

### A2: Create centralized Settings class
- [ ] Use Pydantic `BaseSettings` for all configuration
- [ ] Move hardcoded values: rate limits, upload size, allowed extensions, doc URL
- [ ] Replace scattered `os.getenv()` calls
- **Files**: `app/rate_limit.py:6`, `app/services/upload.py:38-43`, `app/routers/api.py:187`
- **Severity**: MEDIUM

### A3: Use dependency injection for services
- [ ] Replace global service instances with FastAPI `Depends()`
- [ ] Improves testability — can inject mocks
- **Files**: `app/api/climbing.py:24-25`, `app/api/video_quality.py:25`, `app/api/tracking_quality.py:26`
- **Severity**: MEDIUM

### A4: Standardize API response format
- [ ] Define `ErrorResponse` Pydantic schema
- [ ] Replace manual `JSONResponse` with `response_model` on all endpoints
- [ ] Add global exception handler for consistent error format
- **Files**: `app/routers/api.py:240`, `src/climb_sensei/progress/routes.py:291`
- **Severity**: MEDIUM

### A5: Migrate all routes to /api/v1/ prefix
- [ ] Main routers (`routes`, `attempts`, `sessions`, `goals`) lack version prefix
- [ ] Climbing API already uses `/api/v1/climbing/`
- [ ] Creates inconsistency across the API surface
- **File**: `app/main.py:141-146`
- **Severity**: LOW

## Error Handling & Reliability

### E1: Store error messages on failed analyses
- [ ] Add `error_message` field to Video model
- [ ] Store exception details when background task fails
- [ ] Return error message in `/api/videos/{id}/status` endpoint
- **File**: `app/routers/api.py:171-178`
- **Severity**: HIGH

### E2: Add analysis retry mechanism
- [ ] New endpoint to retry failed analysis without re-uploading video
- [ ] Video file is still on disk if analysis failed
- **File**: `app/routers/api.py` (new endpoint needed)
- **Severity**: MEDIUM

### E3: Add background task timeout
- [ ] Analysis thread has no timeout — can hang indefinitely
- [ ] Use `ThreadPoolExecutor` with configurable timeout
- **File**: `app/routers/api.py:220-237`
- **Severity**: MEDIUM

### E4: Improve logging for security events
- [ ] Log failed auth attempts at WARNING level
- [ ] Log unauthorized access attempts
- [ ] Ensure secrets are never logged
- **File**: `src/climb_sensei/auth/__init__.py`
- **Severity**: MEDIUM

## Testing

### T1: Add integration tests for API endpoints
- [ ] Test route CRUD (`/api/routes`)
- [ ] Test goal CRUD (`/api/goals`)
- [ ] Test session listing (`/api/sessions`)
- [ ] Test video status polling (`/api/videos/{id}/status`)
- **Severity**: HIGH

### T2: Create shared test fixtures (conftest.py)
- [ ] `test_db` fixture with in-memory SQLite
- [ ] `client` fixture with TestClient
- [ ] `authenticated_client` fixture with JWT token
- [ ] Reduce test file boilerplate
- **File**: `tests/conftest.py` (needs creation or enhancement)
- **Severity**: MEDIUM

### T3: Test upload pipeline
- [ ] Test file validation (size, format)
- [ ] Test background task error handling
- [ ] Test status polling during analysis
- [ ] Mock MediaPipe for CI environments
- **Severity**: MEDIUM

### T4: Test Service-Worker-Allowed header
- [ ] Verify `/static/sw.js` returns `Service-Worker-Allowed: /` header
- [ ] Verify manifest.json is served correctly
- **File**: new test file needed
- **Severity**: LOW

## Database

### D1: Generate baseline Alembic migration
- [ ] Only 1 migration exists for routes/attempts
- [ ] Missing migrations for ProgressMetric, Goal, initial schema
- [ ] Generate migration that covers current schema state
- **File**: `alembic/versions/`
- **Severity**: MEDIUM

### D2: Fix sync/async database inconsistency
- [ ] `aiosqlite` is a dependency but most code uses sync `SessionLocal`
- [ ] Background tasks use `SessionLocal()` directly in threads
- [ ] Either commit to async with `AsyncSession` or remove `aiosqlite`
- **File**: `src/climb_sensei/database/config.py`
- **Severity**: LOW

## Dependencies

### DEP1: Consolidate JWT handling
- [ ] Both `python-jose` and `fastapi-users` handle JWT independently
- [ ] Consolidate to single JWT source to avoid confusion
- **Files**: `src/climb_sensei/auth/__init__.py`, `src/climb_sensei/auth/users.py`
- **Severity**: LOW
