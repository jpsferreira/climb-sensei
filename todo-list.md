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
