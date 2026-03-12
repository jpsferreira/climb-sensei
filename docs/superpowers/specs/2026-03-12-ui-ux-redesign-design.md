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
  - Mini sparkline showing trend of primary tracked metric
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
- Comparison with previous attempt: delta values (↑/↓/→) for each metric
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
- Grade pyramid or grade distribution chart (visual)
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
  grade: str (e.g., "V4", "6b+", "5.11a")
  type: str (boulder | sport | trad)
  location: str (optional)
  status: str (projecting | sent)
  created_at: datetime
  updated_at: datetime
```

### New Entity: Attempt

```
Attempt:
  id: int (PK)
  route_id: int (FK → Route)
  session_id: int (FK → Session, optional)
  analysis_id: int (FK → Analysis, optional — linked after analysis completes)
  video_path: str
  notes: str (optional)
  date: datetime
  created_at: datetime
```

### Updated: Session

Existing session model stays, but gains relationship to Attempts instead of directly to Analyses:

```
Session (existing, updated):
  id, user_id, date, location, notes (existing fields)
  → has many Attempts (new relationship)
```

### Updated: Goal

Goals now belong to a Route instead of being standalone:

```
Goal (existing, updated):
  id, user_id (existing)
  route_id: int (FK → Route, new)
  metric: str (existing)
  target_value: float (existing)
  deadline: date (optional, existing)
```

### Updated: Analysis

The existing Analysis model is now linked through Attempt rather than being a top-level entity:

```
Analysis (existing):
  → accessed via Attempt.analysis_id
  (no structural changes to analysis data itself)
```

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
- **Pull-to-refresh** — on list pages (routes, sessions)
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

Existing endpoints (`/api/analyses`, `/api/goals`, `/upload`) will need migration paths or deprecation.

## Migration Strategy

This is a significant restructure. Recommended approach:

1. **Backend first** — add Route and Attempt models, new API endpoints. Keep existing endpoints working.
2. **Frontend page by page** — rebuild each template against the new API. Start with Routes (landing) since it's the core.
3. **Data migration** — existing analyses become attempts under an "Uncategorized" route, allowing users to reorganize.
4. **Deprecate old endpoints** — once all templates use new API.

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
