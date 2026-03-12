# Climb Sensei — UI/UX Redesign Spec

## Overview

Comprehensive redesign of the Climb Sensei web app to shift from a session/analysis-centric model to a **route-centric "Route Journal"** model. The core concept: climbers track progression on specific routes over time through multiple video-analyzed attempts.

**Target audience:** Climbers of all levels (beginner to advanced). Metrics are surfaced progressively — overview first, details on tap. No jargon without context.

**Platform priority:** Mobile-first, responsive to desktop. Climbers upload and check progress at the gym on their phones.

## Mental Model

```
Session (climbing day)
  └── Routes attempted
        └── Attempts (videos)
              └── Metrics & analysis per attempt
                    └── Progress tracked across attempts
```

Routes are the primary object. Sessions group what you climbed on a given day. Progress is measured per-route across attempts.

## Navigation

4-tab bottom navigation bar (mobile), collapsing to sidebar or top nav on desktop:

| Tab | Icon | Page | Role |
|-----|------|------|------|
| Routes | Climbing icon | Route library | **Landing page** — the default view |
| Upload | Center FAB (+) | Upload flow | Prominent action button, visually elevated |
| Sessions | Calendar icon | Session calendar | Daily climbing log |
| Profile | User icon | Profile & settings | Stats, goals overview, account |

Auth pages (login/register) are outside the tab structure — visual refresh only, no structural changes.

## Pages

### 1. Routes (Landing Page)

**Layout:** Compact list with search and type filters.

**Components:**
- **Search bar** — filter routes by name or location
- **Type filter chips** — All / Boulder / Sport / Trad (toggle)
- **Sort control** — Recent / Grade / Most attempts
- **Route list items**, each containing:
  - Grade badge (color-coded by grade range)
  - Route name
  - Location + attempt count + last attempt date
  - Mini sparkline showing trend of `avg_velocity` (default metric; if a goal is active on the route, show that goal's metric instead)
  - Green left border for sent routes
- **"+ New Route" card** — at bottom of list or empty state CTA
- **Route count** — "12 routes" subtle header

**Empty state:** Illustration + "Add your first route" CTA pointing to upload or manual route creation.

**Tapping a route** → navigates to Route Detail.

### 2. Route Detail

**Layout:** Scrollable single page with header, metric chart, and attempt timeline.

**Route Header:**
- Back navigation ("← Routes")
- Route name + grade badge
- Location + type
- Quick stats row: Attempts | First try date | Last try date | Status (Projecting/Sent)

**Metric Chart Section:**
- Selectable metric dropdown (velocity, stability, movement economy, etc.)
- Bar/line chart showing that metric across all attempts
- Metric list is flexible — shows whatever metrics the analysis produced. Not hardcoded.

**Attempt Timeline:**
- Vertical timeline, most recent first
- Active dot on most recent attempt, muted dots on older ones
- Each attempt card shows:
  - Attempt number + date
  - Key metrics inline (3-4 values)
  - "Watch video" link → opens attempt detail
- Older attempts fade slightly for visual hierarchy

**Per-Route Goals:**
- "Set goal" action on the route detail page
- Goal: target metric + target value + optional deadline
- Progress shown inline when goal is active

**Actions:**
- "Add attempt" button → navigates to Upload with route pre-filled
- Edit route (name, grade, location, type, status)
- Delete route (with confirmation)

### 3. Attempt Detail

**Reached by:** Tapping an attempt in the route detail timeline.

**Components:**
- Video player (annotated video if available, original otherwise)
- Full metrics display organized by category (same categories as current analysis)
- Comparison with the chronologically previous attempt for the same route: delta values (↑/↓/→) for each metric
- Notes field (editable)
- Navigation: previous/next attempt arrows

### 4. Upload

**Layout:** Single-page, context-aware.

**Two entry points:**
1. **From Route Detail** ("Add attempt") → route is pre-filled, shown as a confirmed card at top with "Change" option
2. **From tab bar** → route selector shown (dropdown with search, option to create new route inline)

**Components:**
- Route selector or pre-filled route card
- Video picker (tap to select from library; camera option on mobile)
- Session assignment — defaults to today's date, auto-creates session if none exists. Dropdown to pick different session.
- Analysis options — toggle chips: Metrics (on by default) | Annotated Video | Quality Check
- "Analyze Attempt" submit button
- Progress indicator during analysis (existing polling mechanism)

**After analysis completes:** Navigate to the new attempt detail within the route, or show inline results with link to full detail.

### 5. Sessions (Calendar)

**Layout:** Mini calendar + day detail view.

**Calendar:**
- Month view with navigation arrows
- Dots on days with climbing sessions (accent color)
- Today highlighted
- Tapping a day shows the day detail below

**Day Detail (below calendar):**
- Date heading
- List of routes attempted that day, each with:
  - Grade badge
  - Route name
  - Attempt number for that route
- Location label
- Summary: X routes, Y videos
- Tapping a route → navigates to that attempt within Route Detail

**Empty state for selected day:** "No climbing this day" — subtle, not prominent.

**Session metadata:** Location and notes are editable when viewing a session day.

### 6. Profile

**Layout:** Scrollable single page with sections.

**Overall Stats Section:**
- Total routes | Total sessions | Total attempts
- Grade pyramid chart grouped by type (boulder grades and sport grades shown separately, since they are incomparable systems)
- Member since date
- Climbing frequency (e.g., "2.3 sessions/week")

**Goals Overview Section:**
- Aggregated list of all active goals across routes
- Each goal shows: route name + grade, target metric, current vs target value, progress bar, deadline if set
- Tapping a goal → navigates to that route's detail
- Filter: Active / Achieved

**Settings Section:**
- Account details (name, email)
- Password change
- Google OAuth connection status
- Logout

## Data Model Changes

### New Entity: Route

```
Route:
  id: int (PK)
  user_id: int (FK → User)
  name: str
  grade: str (e.g., "V4", "6b+", "5.11a") — free-form string, stored as-is
  grade_system: str (hueco | font | yds | french) — determines sort order
  type: str (boulder | sport | trad)
  location: str (optional)
  status: str (projecting | sent)
  created_at: datetime
  updated_at: datetime
```

**Grade sorting:** Each grade system has a known ordering (e.g., V0 < V1 < ... V16; 5a < 5b < ... 9a+). A utility function maps grade strings to a numeric sort key within their system. Routes with different grade systems sort by system first, then by grade within system. This is display-layer logic, not stored in the DB.

### New Entity: Attempt

```
Attempt:
  id: int (PK)
  route_id: int (FK → Route)
  session_id: int (FK → Session, optional)
  video_id: int (FK → Video) — references existing Video model
  analysis_id: int (FK → Analysis, optional — linked after analysis completes)
  notes: str (optional)
  date: datetime
  created_at: datetime
```

**Relationship to Video:** Attempt references the existing `Video` model via `video_id` FK. The Video model continues to own file metadata (`file_path`, `filename`, `duration_seconds`, `fps`, etc.). Attempt does not duplicate video storage — it wraps a Video with route/session context.

### Updated: Session (ClimbSession)

Existing `ClimbSession` model stays. The `name` field becomes optional (nullable) with auto-generation: when a session is auto-created during upload, name defaults to the date string (e.g., "Mar 12, 2026"). Users can edit the name later. The existing `analyses` relationship remains for backward compatibility; a new `attempts` relationship is added.

```
ClimbSession (existing, updated):
  id, user_id, date, location, notes (existing fields)
  name: str (make nullable, auto-generate from date if not provided)
  total_videos: int (existing)
  avg_performance_score: float (existing)
  → has many Analyses (existing relationship, kept)
  → has many Attempts (new relationship)
```

### Updated: Goal

Goals now belong to a Route instead of being standalone. All existing fields are preserved.

```
Goal (existing, updated):
  id, user_id (existing)
  route_id: int (FK → Route, new, nullable for migration)
  metric_name: str (existing)
  target_value: float (existing)
  current_value: float (existing, nullable) — auto-updated from latest attempt's analysis
  deadline: datetime (optional, existing)
  achieved: bool (existing)
  achieved_at: datetime (existing, nullable)
  notes: str (existing, nullable)
  created_at, updated_at (existing)
```

**Goal current_value auto-update:** When a new attempt's analysis completes for a route, any active goals on that route have their `current_value` updated from the analysis's denormalized metrics (e.g., `avg_velocity`, `avg_movement_economy`). If `current_value >= target_value`, mark `achieved = True` and set `achieved_at`.

### Existing: Video (unchanged)

The Video model is unchanged. Attempts reference it via FK.

```
Video (existing, no changes):
  id, user_id, filename, file_path, uploaded_at
  duration_seconds, fps, width, height, file_size_bytes, status
  → has many Analyses
```

### Existing: Analysis (unchanged structure)

The Analysis model structure is unchanged. It is now also reachable via Attempt → Analysis in addition to the existing Video → Analysis path.

```
Analysis (existing, no structural changes):
  id, video_id, session_id, created_at
  run_metrics, run_video, run_quality, dashboard_position
  summary (JSON), history (JSON), video_quality (JSON), tracking_quality (JSON)
  denormalized metrics (avg_velocity, max_velocity, etc.)
  output_video_path, output_json_path
  → has many ProgressMetrics
```

### Existing: ProgressMetric (kept, extended)

The `ProgressMetric` model is kept for time-series queries. It continues to be created when an analysis completes. The route context is now available by traversing Attempt: `ProgressMetric → Analysis → Attempt → Route`. No schema changes needed, but the `/api/routes/{id}/progress/{metric}` endpoint joins through this chain to filter by route.

## Visual Design

### Keep (from current design)
- Dark theme with `#0f1419` base
- Outfit (display) + DM Sans (body) font pairing
- Amber/orange accent (`#f59e0b`)
- Teal secondary (`#0ea5e9`)
- Success green (`#22c55e`), error red (`#ef4444`)
- CSS custom properties system
- Smooth transitions and fadeUp animations

### Improve
- **Spacing consistency** — standardize on 8px grid
- **Touch targets** — minimum 44px for all interactive elements
- **Information hierarchy** — reduce visual noise, use whitespace more deliberately
- **Typography scale** — tighter, more intentional size steps
- **Card depth** — subtler shadows, rely more on spacing and borders for separation

### Add
- **Bottom tab navigation** — fixed, with center FAB for upload
- **Grade-colored badges** — consistent color coding by grade range throughout
- **Sparkline micro-charts** — on route list items for quick trend visibility
- **Empty states** — every list/page has a helpful empty state with CTA
- **Skeleton loading states** — instead of spinners where possible

### Remove
- Top sticky navbar (replaced by bottom tabs)
- Unused `style.css` light theme file
- Feature grid on upload page (not needed once routes provide context)

## API Changes

New endpoints needed:

```
# Routes
GET    /api/routes              — list user's routes (with filters/search)
POST   /api/routes              — create route
GET    /api/routes/{id}         — route detail with attempts summary
PATCH  /api/routes/{id}         — update route
DELETE /api/routes/{id}         — delete route

# Attempts
GET    /api/routes/{id}/attempts         — list attempts for route (with metrics)
POST   /api/routes/{id}/attempts         — create attempt (upload video + trigger analysis)
GET    /api/routes/{id}/attempts/{id}    — attempt detail with full metrics
DELETE /api/routes/{id}/attempts/{id}    — delete attempt

# Route progress
GET    /api/routes/{id}/progress/{metric} — metric trend data for route

# Sessions (updated)
GET    /api/sessions                     — list sessions (calendar data)
GET    /api/sessions/{id}                — session detail with attempts
```

### Existing Endpoints — Migration Plan

| Existing Endpoint | Action | Notes |
|---|---|---|
| `POST /upload` | Keep, extend | Add optional `route_id` and `session_id` params. Creates Attempt record linking Video → Route → Session. Without `route_id`, behaves as before. |
| `GET /api/videos/{id}/status` | Keep | No changes needed — still polls video processing status. |
| `GET /analysis/{id}` | Keep | Still returns analysis detail. |
| `GET /api/analyses` | Keep during migration | Frontend migrates to route-based queries. Deprecate after all pages updated. |
| `GET /api/sessions` | Keep, extend | Add calendar-friendly response format (list of dates with session IDs). |
| `POST/PATCH /api/sessions` | Keep, update | Make `name` optional in POST (auto-generate from date). |
| `GET /api/goals` | Keep, extend | Add optional `route_id` filter param. |
| `POST/PATCH /api/goals` | Keep, update | Accept `route_id` in POST. |
| `GET /api/progress/{metric}` | Keep | Still works globally. New route-scoped endpoint added alongside. |

## Migration Strategy

This is a significant restructure. Recommended approach:

1. **Backend first** — add Route and Attempt models with Alembic migrations. Add new API endpoints. Keep all existing endpoints working unchanged.
2. **Data migration** — create an "Uncategorized" route per user. For each existing Analysis, create an Attempt record linking: `Attempt.video_id = Analysis.video_id`, `Attempt.analysis_id = Analysis.id`, `Attempt.session_id = Analysis.session_id`, `Attempt.route_id = uncategorized_route.id`, `Attempt.date = Analysis.created_at`. This preserves all existing data.
3. **Frontend page by page** — rebuild each template against the new API. Start with Routes (landing) since it's the core.
4. **Deprecate old endpoints** — once all templates use new API, remove unused endpoints.

## Scope Boundaries

**In scope:**
- All pages described above (Routes, Route Detail, Attempt Detail, Upload, Sessions, Profile, Login, Register)
- New data model (Route, Attempt entities)
- New API endpoints
- Mobile-first responsive design
- Visual refresh of existing design system

**Out of scope:**
- Native mobile app
- Push notifications
- Social features (sharing routes/progress)
- Video recording within the app
- Offline support
- Changes to the core analysis engine (MediaPipe pipeline)
