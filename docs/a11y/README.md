# Accessibility — EvidenceEngine v1.2.1

## Static contract tests

The in-code accessibility surface is covered by `tests/web/test_a11y_contracts.py` —
16 pytest contract tests that render every top-level template and every component
macro, then assert against the produced HTML:

- ARIA attributes on interactive components (`aria-expanded`, `aria-controls`,
  `aria-labelledby`, `role="region"`, `role="dialog"`, `aria-modal`).
- Live region wrappers on review button clusters (`aria-live="polite"`) so
  verdict changes are announced by screen readers.
- Skip-to-content link is the first focusable element of `base.html`.
- Keyboard help modal structure: close button has `aria-label`, backdrop is
  dismissable, focus targets are reachable.
- Icon-only buttons carry accessible names.
- `alt` text on every decorative/meaningful image; SVGs used as icons are
  marked `aria-hidden` when adjacent to text labels.

Run: `pytest tests/web/test_a11y_contracts.py -q` — expected result: 16 passed.

## Live axe-core report

The automated axe-core run against a booted server is **deferred to the
post-merge live environment**. The local machine cannot boot the FastAPI
server reliably right now because macOS Gatekeeper (`syspolicyd`) is doing
first-run provenance checks over a `.venv` full of heavy ML libraries; wall
clock to first byte stretches to minutes and the Playwright harness times
out before the server responds. This is an OS/environment issue, not a code
issue. In Docker or on prod Linux this problem does not exist.

The live axe-core and Lighthouse sweep has not been run yet. It needs a
running server, and Docker is the easiest way to get one.

## What we consider "shipped" for v1.2.1 a11y

- Every ARIA contract asserted by the static tests.
- `prefers-reduced-motion: reduce` disables all animations and transitions.
- Self-hosted fonts (`font-display: swap`), no external font requests.
- Keyboard-only review flow: dashboard index → run detail → J/K nav →
  A/R/F verdict → ? help → Esc close.
- Focus-visible rings on all interactive elements.

## What is explicitly pending

- Live axe-core JSON report.
- Live Lighthouse accessibility score.
- Playwright screenshot diff sweep.

These three still need a live run against a running server.
