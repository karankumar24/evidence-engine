# Design Tokens — EvidenceEngine Reviewer UI

The reviewer dashboard is migrating from CDN Tailwind to a local Tailwind v4
build with a deliberate **editorial-institutional** design language: muted
tones, serif display type, generous rhythm. Think academic journal crossed
with long-form editorial — not SaaS dashboard.

Tokens live in `frontend/css/input.css` inside a Tailwind v4 `@theme` block,
which auto-generates matching utilities (`bg-accent-700`, `text-ink-900`,
`font-serif`, etc.).

---

## Palette

### Ink (text + chrome)
| Token              | Hex       | Intent                              |
|--------------------|-----------|-------------------------------------|
| `--color-ink-900`  | `#0B0B0B` | Body copy, headings                 |
| `--color-ink-700`  | `#2B2B2B` | Secondary text, nav links           |
| `--color-ink-500`  | `#585858` | Muted meta                          |
| `--color-ink-300`  | `#9B9B9B` | Subtle rules                        |
| `--color-ink-100`  | `#E2E2E0` | Hairline borders                    |
| `--color-ink-50`   | `#F1F1EE` | Sunken surfaces                     |

### Paper (warm off-white surfaces)
| Token                | Hex       | Intent                           |
|----------------------|-----------|----------------------------------|
| `--color-paper-50`   | `#FAFAF7` | Page background                  |
| `--color-paper-100`  | `#F4F3EC` | Card surface                     |
| `--color-paper-200`  | `#EBEAE1` | Evidence highlight background    |
| `--color-paper-900`  | `#25241F` | Inverse surfaces                 |

### Accent (deep forest green — the institutional thread)
| Token                | Hex       | Intent                           |
|----------------------|-----------|----------------------------------|
| `--color-accent-700` | `#1F3B2D` | **Primary brand ink**, CTA bg    |
| `--color-accent-500` | `#3E6248` | Hover / mid-tone                 |
| `--color-accent-100` | `#D6E0D9` | Tint wash                        |

### Verdict (muted editorial tones — NOT neon)
| Token                          | Hex       | Intent                         |
|--------------------------------|-----------|--------------------------------|
| `--color-verdict-supported`    | `#3F6B4A` | Muted pine — claim supported   |
| `--color-verdict-contradicted` | `#8B2E2A` | Oxblood — claim contradicted   |
| `--color-verdict-insufficient` | `#B8752B` | Ochre — insufficient evidence  |
| `--color-verdict-review`       | `#4A5A7A` | Slate blue — needs review      |

---

## Typography

| Token          | Stack                                                       |
|----------------|-------------------------------------------------------------|
| `--font-serif` | Source Serif 4, Georgia, Times New Roman, serif             |
| `--font-sans`  | Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto  |
| `--font-mono`  | JetBrains Mono, ui-monospace, SFMono-Regular, Menlo         |

Body defaults to **sans 15px / 1.55**. Headings use serif with `-0.01em`
letter-spacing and weight 600. Evidence prose uses serif at 17px / 1.65.

### Type scale
| Token        | Size    | Line height | Use                          |
|--------------|---------|-------------|------------------------------|
| `--text-xs`  | 12px    | 1.4         | Meta, timestamps             |
| `--text-sm`  | 13px    | 1.45        | Captions, footnotes          |
| `--text-base`| 15px    | 1.55        | Body                         |
| `--text-lg`  | 17px    | 1.5         | Evidence prose               |
| `--text-xl`  | 20px    | 1.4         | Subheads                     |
| `--text-2xl` | 24px    | 1.3         | h3                           |
| `--text-3xl` | 30px    | 1.25        | h2                           |
| `--text-4xl` | 38px    | 1.15        | h1                           |
| `--text-5xl` | 48px    | 1.1         | Hero display                 |

---

## Spacing & radii

Spacing uses an editorial rhythm: **4 / 8 / 12 / 16 / 24 / 32 / 48 px**.
Radii stay flat: `sm=4px`, `md=8px`. Only status chips use `pill=999px`.

---

## Runbook

```bash
npm install              # first-time only
npm run build:css        # one-shot minified build to src/evidenceengine/static/css/app.css
npm run watch:css        # rebuild on change during development
```

Output is committed-gitignored if desired; node_modules/ is already ignored.

---

## Tailwind v4 `@theme` note

Tailwind v4 replaces `tailwind.config.js` with CSS-native `@theme { ... }`.
Any `--color-foo-500` declared inside `@theme` generates utilities like
`bg-foo-500`, `text-foo-500`, `border-foo-500`. Same pattern for
`--font-*`, `--text-*`, `--spacing-*`, `--radius-*`. No config file,
no PostCSS plugin required — the `@tailwindcss/cli` binary handles it.
