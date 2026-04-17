#!/usr/bin/env node
/**
 * Take before/after screenshots for the v1.2 release docs.
 * Before: the 4 direction HTML mockups (pre-redesign reference).
 * After:  live pages from http://127.0.0.1:8000
 */

const { chromium } = require("playwright");
const path = require("path");
const fs = require("fs");

const REPO = path.resolve(__dirname, "..");
const BEFORE_DIR = path.join(REPO, "docs/design/v1.2-before");
const AFTER_DIR = path.join(REPO, "docs/design/v1.2-after");

fs.mkdirSync(BEFORE_DIR, { recursive: true });
fs.mkdirSync(AFTER_DIR, { recursive: true });

const BEFORE_PAGES = [
  { name: "direction-01-editorial", file: "docs/design/direction-01-editorial.html" },
  { name: "direction-02-scientific", file: "docs/design/direction-02-scientific.html" },
  { name: "direction-03-archival", file: "docs/design/direction-03-archival.html" },
  { name: "direction-04-minimalist", file: "docs/design/direction-04-minimalist.html" },
];

const AFTER_PAGES = [
  { name: "dashboard-index", url: "http://127.0.0.1:8000/dashboard/" },
  { name: "error-404", url: "http://127.0.0.1:8000/this-page-does-not-exist" },
  { name: "design-system", url: "http://127.0.0.1:8000/design-system" },
];

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({ viewport: { width: 1440, height: 900 } });
  const page = await context.newPage();

  console.log("Taking BEFORE screenshots (direction mockups)...");
  for (const { name, file } of BEFORE_PAGES) {
    const filePath = path.join(REPO, file);
    if (!fs.existsSync(filePath)) { console.log(`  SKIP: ${file} not found`); continue; }
    await page.goto(`file://${filePath}`);
    await page.waitForTimeout(500);
    const dest = path.join(BEFORE_DIR, `${name}.png`);
    await page.screenshot({ path: dest, fullPage: false });
    console.log(`  ✓ ${dest}`);
  }

  console.log("\nTaking AFTER screenshots (live server)...");
  for (const { name, url } of AFTER_PAGES) {
    try {
      await page.goto(url, { waitUntil: "networkidle", timeout: 10000 });
      const dest = path.join(AFTER_DIR, `${name}.png`);
      await page.screenshot({ path: dest, fullPage: false });
      console.log(`  ✓ ${dest}`);
    } catch (e) {
      console.log(`  SKIP: ${url} — ${e.message.split("\n")[0]}`);
    }
  }

  await browser.close();
  console.log("\nDone.");
})();
