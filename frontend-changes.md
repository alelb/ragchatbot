# Frontend changes: dark/light theme toggle

Users can switch between the existing dark theme and a new light theme with a round icon button fixed in the top-right corner. The choice is saved in `localStorage`, and dark stays the default when nothing is saved.

## `frontend/index.html`
- A small inline script in `<head>` reads the saved theme and sets `data-theme` on `<html>` before the stylesheet loads, so the page never flashes the wrong theme on reload. If storage is unavailable, the error is caught and the page uses the default theme.
- A new `<button id="themeToggle" class="theme-toggle">` placed before `.container`. It holds two inline SVG icons, a sun and a moon, both marked `aria-hidden="true"`.
- Cache-busting versions bumped: `style.css?v=14`, `script.js?v=13`.

## `frontend/style.css`
- **New variables in `:root` (dark):** `--link-color`, `--code-bg`, `--error-color`, `--success-color`, plus `color-scheme: dark`. These replace colors that were hardcoded before: the link blue `#60a5fa`, the code background `rgba(0,0,0,.2)`, and the error/success text colors.
- **`:root[data-theme="light"]`:** overrides the variables, keeping the same color roles so the visual hierarchy doesn't change:
  - Background `#f8fafc`, surfaces `#ffffff`, hover surface `#f1f5f9`, borders `#e2e8f0`.
  - Text `#0f172a` (primary) and `#475569` (secondary), both well above WCAG AA contrast on white.
  - Primary `#1d4ed8` with hover `#1e40af`, a deeper blue that reads better on light backgrounds. User bubbles stay `#2563eb` with white text.
  - Darker link, error (`#b91c1c`) and success (`#15803d`) colors, plus a lighter code background and shadow.
  - `color-scheme: light`, so native scrollbars and form controls follow the theme.
- **Smooth theme switch:** `html.theme-transition *` cross-fades background, text, border and shadow colors over 0.3s. The class is only present while the switch happens, so it doesn't slow down the existing hover transitions.
- **`.theme-toggle`:** a 40px circle using `--surface`, `--border-color` and `--shadow`. Hover, active and focus states match the other buttons: a primary-colored border and text, a 1px lift, and a `--focus-ring` shown on `:focus-visible`. The two icons sit on top of each other, and the hidden one rotates and shrinks out (0.4s). The dark theme shows the sun, which means "switch to light"; the light theme shows the moon.
- **`prefers-reduced-motion: reduce`** turns off the toggle and theme transitions.
- **Fix:** the `blockquote` border used `var(--primary)`, which doesn't exist. It now uses `var(--primary-color)`.

## `frontend/script.js`
- `themeToggle` is added to the DOM references, and a click listener calls `toggleTheme()`. Because it's a native `<button>`, Tab focuses it and Enter/Space activate it.
- `getTheme()` reads `data-theme` from `<html>` and returns `dark` when it isn't set.
- `toggleTheme()` adds `theme-transition`, flips `data-theme`, removes the class after 300ms, and saves the choice. Saving is wrapped in try/catch so it still works when storage is blocked.
- `updateThemeToggle()` keeps `aria-label` and `title` set to the action the button will perform ("Switch to light theme" / "Switch to dark theme"). It runs on load and after each toggle.
