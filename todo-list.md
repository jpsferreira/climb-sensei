# UX/UI Improvements - Todo List

## #4 Drag & Drop Upload

- [x] Add drag-and-drop zone with visual feedback (drag-over state)
- [x] Show video preview thumbnail after file selection
- [x] Display file size and name after selection
- [x] Add format validation before upload with user-friendly error

## #6 Skeleton Loading States

- [x] Add skeleton/shimmer CSS to base.html
- [x] Add skeleton placeholders for route list cards
- [x] Add skeleton placeholders for route detail page (sessions, profile reuse same pattern)

## #8 Search & Filter Persistence

- [x] Persist search query, filter type, and sort order in URL params
- [x] Restore filter state from URL params on page load
- [x] Update URL without full page reload on filter/search changes

## #9 Empty State Improvements

- [x] Add SVG icons and contextual descriptions to all empty states
- [x] Improve routes, route detail, profile, and sessions empty states

## #7 Route Detail - Metric Insights

- [x] Add insights summary panel below chart (last attempt %, overall trend, best, average)
- [x] Color-code trends green/red with direction-aware logic (lower-is-better for sway/fatigue)
- [x] Add delta badges to timeline attempt cards showing change from previous attempt

## #13 Accessibility Improvements

- [x] Add visible :focus-visible indicators for keyboard navigation on all interactive elements
- [x] Add skip-to-content link for keyboard users
- [x] Add modal focus trapping (Tab cycles within modal, Escape closes)
- [x] Add ARIA roles/attributes: aria-current on nav, aria-live on alerts, role=dialog on modals
- [x] Use semantic `<main>` landmark instead of `<div>` for main content
- [x] Add prefers-reduced-motion media query to disable animations

## #10 Toast Notification Positioning

- [x] Reposition toasts to bottom-center on mobile (above nav bar)
- [x] Add slide-up animation on mobile instead of slide-from-right
- [x] Add swipe-to-dismiss touch gesture for toasts
- [x] Add close/dismiss button on each toast for accessibility

## #12 Micro-animations & Feedback

- [x] Add button press scale feedback (tactile 0.97 scale on :active)
- [x] Add route card hover lift with shadow and active press state
- [x] Add staggered entrance animation for route list items
- [x] Add timeline card entrance animation with stagger on route detail
- [x] Add nav tab and FAB press feedback (scale bounce)
- [x] Add filter chip press feedback
- [x] Add insight card hover lift on route detail

## #14 Profile Data Visualization

- [x] Add activity overview card with weekly sessions bar chart (last 8 weeks)
- [x] Add "this month" session count and "best week" stats
- [x] Add week streak counter badge

## #11 Dark/Light Theme Toggle

- [x] Add light theme CSS variables under [data-theme="light"]
- [x] Auto-detect system preference via prefers-color-scheme
- [x] Persist theme choice in localStorage
- [x] Apply theme before body render to prevent flash
- [x] Add toggle button in profile settings page

## #5 Mobile Touch Gestures

- [x] Add swipe-to-navigate between calendar months on sessions page
- [x] Add slide transition animation for month changes

## #1 Onboarding Flow

- [x] Add 3-step guided walkthrough (Welcome, Create Route, Upload Video)
- [x] Show onboarding only for new users with no routes (localStorage flag)
- [x] Step indicators (dots) with active/done states
- [x] Auto-advance after route creation, dismissible at any step
- [x] Guard localStorage access with try/catch

## #2 Video Player + Metrics Synchronization

- [x] Add click-to-seek on the metric plot chart (click anywhere to jump video to that timestamp)
- [x] Add drag-to-scrub on the chart (drag to scrub video in real-time, mouse + touch)
- [x] Add vertical playhead line on the chart showing current position
- [x] Add hint text below chart ("Click or drag on the chart to seek the video")
