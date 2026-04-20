/**
 * Golden-path E2E: upload a PDF, watch the pipeline run, verify the review dashboard loads.
 *
 * Run locally:  npx playwright test
 * Run vs live:  BASE_URL=https://evidenceengine.fly.dev npx playwright test
 *
 * The pipeline uses real LLM calls, so set an LLM_API_KEY in .env before running locally.
 * Expect ~30-90s for the full pipeline to complete on the first run (model cold-start).
 */

import { test, expect } from "@playwright/test";
import path from "path";

const FIXTURE_PDF = path.resolve(__dirname, "../fixtures/simple_report.pdf");

// ── Smoke: static pages ───────────────────────────────────────────────────────

test("dashboard page loads", async ({ page }) => {
  await page.goto("/dashboard");
  await expect(page).toHaveTitle(/EvidenceEngine/i);
  // Nav brand renders as generic element in a11y tree; check the page heading instead
  await expect(page.getByRole("heading", { name: /Pipeline Runs/i })).toBeVisible();
});

test("upload page renders the form", async ({ page }) => {
  await page.goto("/upload");
  await expect(page.locator("input[name='report']")).toBeAttached();
  await expect(page.locator("button[type='submit']")).toBeVisible();
});

// ── Golden path: upload → pipeline → review dashboard ────────────────────────

test("upload PDF → pipeline runs → review dashboard loads", async ({ page }) => {
  // 1. Navigate to upload form
  await page.goto("/upload");

  // 2. Attach report PDF (input may be visually hidden behind the drag-drop UI)
  await page.locator("input[name='report']").setInputFiles(FIXTURE_PDF);

  // 3. Submit — server parses the file, creates Packet + RunVersion, redirects to status page
  await Promise.all([
    page.waitForURL(/\/runs\/[^/]+\/status/),
    page.locator("button[type='submit']").click(),
  ]);

  const statusURL = page.url();
  expect(statusURL).toMatch(/\/runs\/[^/]+\/status/);

  // 4. Pipeline runs in background (extract → retrieve → classify).
  //    When done the HTMX poll returns HX-Redirect → /dashboard/{packet}/{run}.
  //    We wait up to 90 s for that navigation.
  await page.waitForURL(/\/dashboard\//, { timeout: 90_000 });

  // 5. Review dashboard — check a recognisable UI landmark is present.
  //    The page shows either claims to review or an empty-state message;
  //    both confirm the pipeline completed and the UI rendered.
  const body = page.locator("main, body");
  await expect(body).toContainText(
    /claim|verified|supported|refuted|insufficient|extracted|pipeline/i,
  );
});

// ── Health: API stays up ──────────────────────────────────────────────────────

test("/health returns 200 ok", async ({ request }) => {
  const res = await request.get("/health");
  expect(res.status()).toBe(200);
  const body = await res.json();
  expect(body.status).toBe("ok");
});

test("/healthz returns db reachable", async ({ request }) => {
  const res = await request.get("/healthz");
  expect(res.status()).toBe(200);
  const body = await res.json();
  expect(body.db).toBe("reachable");
});
