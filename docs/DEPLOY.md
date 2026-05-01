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

Run from the repo root.

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
flyctl volumes create ee_data --region sjc --size 10 --app evidenceengine
```

10 GB matches the current production deployment. Holds the SciFact-fine-tuned NLI model (~500 MB), the BGE and reranker model caches, uploaded PDFs, and BM25 indexes. Scale up later if needed.

### 5. Set runtime secrets

The classification path is local NLI (no LLM call). Only set these if you
want the optional "Explain this verdict" button to work:

```sh
flyctl secrets set LLM_API_KEY='your-gemini-or-openai-compatible-key' \
    CORS_ORIGINS='https://your-app.fly.dev,http://127.0.0.1:8000' \
    --app evidenceengine
```

`LLM_API_KEY` is used only by the on-demand "Explain this verdict" endpoint. If unset, the Explain button shows a "configure LLM_API_KEY" message and the rest of the pipeline runs unaffected. Verdict classification is local NLI and does not call any external API.

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
curl -fsS https://your-app.fly.dev/healthz           # {"status":"ok"}
open https://your-app.fly.dev/dashboard
```

Upload a PDF via the dashboard, watch a run complete. The first run after a cold start takes longer because the NLI and dense models prewarm on first request.

---

## Known limits / gotchas

- **No external rate limit on Explain endpoint.** The local NLI verdict path makes zero external calls, but the on-demand Explain button calls Gemini / your configured LLM. If you expose this to untrusted users, gate it behind auth or a per-IP rate limit. Otherwise anyone can bill arbitrary tokens against your key.
- **Machine spec.** `fly.toml` is configured for `shared-cpu-2x` with 4 GB RAM. The NLI model fits comfortably in 4 GB; if you swap to a larger NLI variant (e.g. DeBERTa-v3-large) you will OOM unless you also bump RAM. Inference time on shared-cpu-2x for 25 claims is 15-60s depending on evidence span count.
- **Auto-stop / always-warm.** Current `fly.toml` defaults to `auto_start_machines = false` and `min_machines_running = 0` (the public demo was scaled to zero before public release of the repo). To keep one machine always running, set `auto_start_machines = true`, `auto_stop_machines = "suspend"`, `min_machines_running = 1` and `fly deploy`.
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
