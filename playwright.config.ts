import { defineConfig, devices } from "@playwright/test";

const BASE_URL = process.env.BASE_URL ?? "http://localhost:8000";

export default defineConfig({
  testDir: "./tests/e2e",
  timeout: 120_000, // pipeline runs take up to 60s with LLM calls
  expect: { timeout: 10_000 },
  fullyParallel: false, // sequential: each test uploads a packet, no cross-contamination
  retries: 0,
  reporter: "list",

  use: {
    baseURL: BASE_URL,
    trace: "on-first-retry",
    screenshot: "only-on-failure",
  },

  projects: [
    {
      name: "chromium",
      use: { ...devices["Desktop Chrome"] },
    },
  ],

  // Start a local server when BASE_URL is localhost.
  // Skipped automatically when BASE_URL points at a remote host.
  ...(BASE_URL.includes("localhost") || BASE_URL.includes("127.0.0.1")
    ? {
        webServer: {
          command:
            "PYTHONPATH=src .venv/bin/uvicorn evidenceengine.api.app:app --host 127.0.0.1 --port 8000",
          url: BASE_URL + "/health",
          reuseExistingServer: true,
          timeout: 60_000,
        },
      }
    : {}),
});
