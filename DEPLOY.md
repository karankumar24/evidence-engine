# EvidenceEngine — Fly.io deployment runbook

First-time deploy: ~30 minutes. You run seven commands; everything else is
baked into `Dockerfile` / `fly.toml` / `alembic`.

---

## Prereqs (one-time)

1. **Install flyctl** (already installed at `~/.fly/bin/flyctl` this session):
   ```sh
   export PATH="$HOME/.fly/bin:$PATH"
   flyctl version     # should print fly v0.x.x
   ```
2. **Log in to Fly** (opens your browser — only you can do this step):
   ```sh
   flyctl auth login
   ```
3. **Credit card on file.** Fly's free tier requires it, even if you never
   exceed the free limits. Add at <https://fly.io/dashboard/personal/billing>.

---

## Deploy (seven commands)

Run from the repo root (`/Users/karankumar/Desktop/vibebuild`).

### 1. Create the app

```sh
flyctl apps create evidenceengine --org personal
```

If the name is taken, pick another (e.g. `evidenceengine-kk`) and update
`app = "..."` in `fly.toml` to match.

### 2. Create managed Postgres

```sh
flyctl postgres create --name evidenceengine-db --region sjc \
    --initial-cluster-size 1 --vm-size shared-cpu-1x --volume-size 1
```

Save the connection string it prints — you'll need it in step 4.

### 3. Attach the database to the app

```sh
flyctl postgres attach evidenceengine-db --app evidenceengine
```

This auto-creates the `evidenceengine` database, sets up a user, and sets the
`DATABASE_URL` secret on the app. `alembic/env.py` reads that env var and
normalizes `postgres://` → `postgresql+asyncpg://` automatically.

### 4. Create the persistent volume (uploads + BM25 indexes)

```sh
flyctl volumes create ee_data --region sjc --size 1 --app evidenceengine
```

1 GB is enough for dozens of PDFs and indexes. Scale up later if needed.

### 5. Set the LLM secret

```sh
flyctl secrets set LLM_API_KEY='sk-or-v1-…your-openrouter-key…' \
    EXTRACTION_MODEL='arcee-ai/trinity-large-preview:free' \
    CLASSIFICATION_MODEL='arcee-ai/trinity-large-preview:free' \
    CORS_ORIGINS='https://evidenceengine.fly.dev,http://127.0.0.1:8000' \
    --app evidenceengine
```

Use the OpenRouter key already in your local `.env`. Never commit it.

### 6. Deploy

```sh
flyctl deploy --app evidenceengine
```

This builds the Docker image, runs `alembic upgrade head` as a release
command, then rolls out. First build takes ~5 minutes (caches all
subsequent builds). Watch the logs:

```sh
flyctl logs --app evidenceengine
```

### 7. Smoke-test

```sh
curl -fsS https://evidenceengine.fly.dev/healthz           # {"status":"ok"}
open https://evidenceengine.fly.dev/dashboard
```

Upload a PDF via the dashboard, watch a run complete. If you see rate-limit
errors from OpenRouter, wait 60s — free tier limits reset fast.

---

## Known limits / gotchas

- **OpenRouter free tier** rate-limits hard (roughly 20 rpm on the
  `:free` Trinity model). For a live demo, have at most one upload in
  flight at a time. No retry logic yet (v1.2.3 candidate).
- **Scale-to-zero** is enabled — the first request after idle has a
  ~2s cold-start. Disable via
  `flyctl scale count 1 --app evidenceengine` if you need always-warm.
- **Single worker** by design (BM25 index is per-process). Scale
  horizontally via `fly scale count N`, not `--workers N`.
- **macOS Tahoe local dev** still hits Gatekeeper grind on first run
  after a fresh venv; this does NOT affect Fly.io (Linux containers
  have no syspolicyd).
- **Local CSS must be rebuilt manually.** `scripts/dev.sh` does not run
  Tailwind. After editing any template or `frontend/css/input.css`, run:
  ```sh
  npm run build:css     # one-shot rebuild
  # or in a second terminal:
  npm run watch:css     # auto-rebuilds on change
  ```
  Fly.io always rebuilds CSS fresh in the Dockerfile `css-builder` stage,
  so the deployed site is always up-to-date; local dev is not.
- **No auth.** Anyone with the URL can review claims. Keep the URL
  private until auth lands.

---

## Rolling back

```sh
flyctl releases --app evidenceengine         # list releases
flyctl releases rollback N --app evidenceengine
```

## Destroying everything

```sh
flyctl apps destroy evidenceengine
flyctl postgres destroy evidenceengine-db
flyctl volumes destroy ee_data
```
