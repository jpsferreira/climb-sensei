# Production Readiness Plan

Implementation plan for making climb-sensei ready for public consumer use.
Organized by priority tier — each tier should be completed before moving to the next.

---

## Tier 1: Critical Security & Infrastructure (Do Not Deploy Without)

### 1.1 Secret Key Management

- **Files**: `src/climb_sensei/auth/users.py`, `src/climb_sensei/auth/__init__.py`, `src/climb_sensei/auth/routes_new.py`
- **Problem**: Hardcoded fallback `SECRET_KEY` across three modules. If env var is unset, all JWTs and OAuth state use a publicly known default.
- **Fix**: Remove all fallback defaults. Raise on startup if `SECRET_KEY` env var is missing. Single source of truth for the key.
- **Generate**: `openssl rand -hex 32`

### 1.2 HTTPS Enforcement

- **Files**: `app/main.py`
- **Problem**: No TLS. Auth tokens transmitted in plaintext.
- **Fix**: Add `TrustedHostMiddleware`, HTTP-to-HTTPS redirect middleware. Set `Secure` flag on cookies. In production, terminate TLS at reverse proxy (nginx/Caddy) or use managed TLS (Railway, Fly.io, etc.).

### 1.3 CORS Middleware

- **Files**: `app/main.py`
- **Problem**: No CORS headers configured.
- **Fix**: Add `CORSMiddleware` with explicit `allow_origins` whitelist. No wildcards in production.

### 1.4 Rate Limiting

- **Files**: `app/main.py`, auth endpoints
- **Problem**: No rate limiting on any endpoint. Auth endpoints are brute-force vulnerable.
- **Fix**: Add `slowapi` or similar. Targets:
  - Auth endpoints: 5 req/min per IP
  - Upload endpoint: 10 req/hour per user
  - General API: 60 req/min per user

### 1.5 Output File Access Control

- **Files**: `app/main.py` (static mount at `/outputs`), `app/routers/api.py`
- **Problem**: `/outputs/{analysis_id}_output.webm` is publicly accessible. Any user can guess another user's analysis ID and download their video.
- **Fix**: Replace static mount with an authenticated endpoint that verifies the requesting user owns the analysis before serving the file. Same for `/uploads/`.

### 1.6 Upload Size Limits

- **Files**: `app/routers/api.py`
- **Problem**: No file size validation. A user can upload arbitrarily large files.
- **Fix**: Add `Content-Length` header check (reject > 500MB). Add middleware-level request body limit. Validate on the FastAPI side before writing to disk.

### 1.7 PostgreSQL Migration

- **Files**: `src/climb_sensei/database/config.py`, `alembic/env.py`
- **Problem**: SQLite has a single-writer lock. Cannot handle concurrent users.
- **Fix**:
  - Switch `DATABASE_URL` default to PostgreSQL
  - Configure connection pooling: `pool_size=20`, `pool_pre_ping=True`, `pool_recycle=3600`
  - Remove `init_db()` → `Base.metadata.create_all()` call from production path
  - Run Alembic migrations on deploy instead
  - Create a baseline Alembic migration from current schema

### 1.8 Security Headers

- **Files**: `app/main.py`
- **Fix**: Add middleware for:
  - `X-Frame-Options: DENY`
  - `X-Content-Type-Options: nosniff`
  - `Referrer-Policy: strict-origin-when-cross-origin`
  - `Content-Security-Policy` (restrict script sources)

---

## Tier 2: Reliability & Operations

### 2.1 Task Queue (Replace Daemon Threads)

- **Files**: `app/routers/api.py` (lines 36-182)
- **Problem**: Analysis runs in `threading.Thread(daemon=True)`. Tasks are lost on server restart, no retry, no timeout, no persistence.
- **Fix**: Introduce Celery + Redis (or RQ). Steps:
  - Extract `_run_analysis_pipeline` into a Celery task
  - Add Redis as broker
  - Keep polling endpoint (`/api/videos/{id}/status`) — it already reads from DB
  - Add task timeout (10 min per video)
  - Add retry logic (max 2 retries with exponential backoff)

### 2.2 Object Storage for Videos

- **Files**: `app/services/upload.py`
- **Problem**: Videos stored on local disk. Won't survive container restarts. Doesn't scale.
- **Fix**: Upload to S3/GCS/R2. Store the object key in the `Video.file_path` field. Serve via presigned URLs (solves access control too).

### 2.3 Disk Cleanup & Lifecycle

- **Files**: `app/services/upload.py`
- **Problem**: Output files accumulate indefinitely. No cleanup.
- **Fix**:
  - Delete source upload after successful analysis (already partially done, but not on error)
  - Add a scheduled job to delete output videos older than 30 days
  - Store retention policy in the Analysis model

### 2.4 Docker Deployment

- **Files**: Create `Dockerfile`, `docker-compose.yml`, `.dockerignore`
- **Fix**:
  - Multi-stage Dockerfile (build deps → runtime)
  - Non-root user
  - Health check endpoint (`/health`)
  - docker-compose with: app, PostgreSQL, Redis, nginx (TLS termination)
  - Production uvicorn config: `--workers 4 --timeout-keep-alive 30`

### 2.5 Database Backups

- **Fix**: Automated daily PostgreSQL backups (pg_dump to S3). Test restoration quarterly.

### 2.6 Graceful Shutdown

- **Files**: `app/main.py`, `app/routers/api.py`
- **Problem**: Daemon threads killed on SIGTERM. Running analyses are lost.
- **Fix**: Use non-daemon threads or Celery tasks. Add shutdown handler that waits for in-progress tasks (with timeout).

---

## Tier 3: Observability & Error Handling

### 3.1 Global Exception Handler

- **Files**: `app/main.py`
- **Fix**: Add `@app.exception_handler(Exception)` that returns consistent JSON error format and logs the traceback. Never leak internal details to clients.

### 3.2 Structured Logging

- **Files**: All modules
- **Problem**: Minimal logging, no consistent format.
- **Fix**:
  - Central logging config (JSON format for production, human-readable for dev)
  - Log levels: ERROR for failures, WARNING for degraded behavior, INFO for key events
  - Include request_id, user_id in all log entries
  - Configure log rotation

### 3.3 Health Check Endpoint

- **Files**: `app/main.py`
- **Fix**: `GET /health` that checks DB connectivity and returns `200 OK` with version info.

### 3.4 Monitoring & Alerting

- **Fix**:
  - Prometheus metrics (request count, latency, error rate, analysis duration)
  - Alert on: error rate > 5%, analysis failure rate > 10%, disk usage > 80%
  - Optional: Sentry for error tracking

### 3.5 Background Task Error Reporting

- **Files**: `app/routers/api.py`
- **Problem**: Background failures are silent. User sees "Analysis Failed" with no details.
- **Fix**: Store error message in Video/Analysis record. Surface in API response and UI.

---

## Tier 4: Scalability & Performance

### 4.1 CDN for Video Delivery

- **Fix**: Serve output videos through CloudFront/Cloudflare. Reduces server bandwidth and latency.

### 4.2 Redis Caching

- **Fix**: Cache frequently accessed data (route lists, session summaries). Invalidate on writes.

### 4.3 API Versioning

- **Fix**: Prefix all API routes with `/api/v1/`. Enables backward-compatible evolution.

### 4.4 Database Indexing Audit

- **Files**: `src/climb_sensei/database/models.py`
- **Fix**: Review query patterns. Add composite indexes for common queries (e.g., `Attempt.route_id + Attempt.date`).

### 4.5 Video Processing Optimization

- **Fix**: Consider GPU-accelerated MediaPipe, video resolution downscaling for analysis (keep original for playback), parallel frame processing.

---

## Tier 5: Testing & CI/CD

### 5.1 Test Coverage to 80%+

- **Current**: 31% (4,473/6,447 lines)
- **Priority targets**:
  - Auth flow (login, register, token refresh, OAuth)
  - Upload → analysis → attempt pipeline end-to-end
  - Background task error cases
  - File access control
  - Database migrations

### 5.2 Security Testing

- **Fix**: Add tests for:
  - Accessing another user's videos/analyses (expect 403)
  - Uploading oversized files (expect 413)
  - Rate limit enforcement
  - Invalid/expired JWT handling
  - SQL injection on search endpoints

### 5.3 Load Testing

- **Fix**: k6 or locust scripts simulating concurrent uploads and API usage. Establish baseline latency and throughput.

### 5.4 CI/CD Deployment Pipeline

- **Files**: `.github/workflows/ci.yaml`
- **Fix**: Add deployment stages:
  - `test` → `build docker image` → `push to registry` → `deploy to staging` → `smoke test` → `deploy to production`
  - Automated rollback on failed health checks

---

## Estimated Timeline

| Tier      | Scope                     | Estimate      |
| --------- | ------------------------- | ------------- |
| 1         | Security & Infrastructure | 1-2 weeks     |
| 2         | Reliability & Operations  | 1-2 weeks     |
| 3         | Observability             | 3-5 days      |
| 4         | Scalability               | 1 week        |
| 5         | Testing & CI/CD           | 1-2 weeks     |
| **Total** |                           | **5-8 weeks** |

Tiers can overlap. Tier 1 is the hard gate — nothing goes public without it.

---

## Quick Reference: Key Files

| File                                  | What Needs Fixing                                                       |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `app/main.py`                         | CORS, security headers, exception handler, health check, HTTPS redirect |
| `app/routers/api.py`                  | Upload limits, access control, replace daemon threads, error reporting  |
| `app/services/upload.py`              | S3 storage, cleanup, disk limits                                        |
| `src/climb_sensei/auth/users.py`      | Remove SECRET_KEY fallback                                              |
| `src/climb_sensei/auth/__init__.py`   | Remove SECRET_KEY fallback                                              |
| `src/climb_sensei/auth/routes_new.py` | Remove hardcoded OAuth state secret                                     |
| `src/climb_sensei/database/config.py` | PostgreSQL, pooling, remove create_all()                                |
| `alembic/env.py`                      | Auto-run migrations on deploy                                           |
