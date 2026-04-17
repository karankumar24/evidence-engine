"""Tests for Phase 8 deployment hardening."""
import importlib
import inspect
from pathlib import Path
import re

import pytest
from httpx import AsyncClient, ASGITransport

from evidenceengine.api.app import create_app, SecurityHeadersMiddleware, _CSP
from evidenceengine.core.config import settings

TEMPLATES_DIR = Path(__file__).parent.parent.parent / "src" / "evidenceengine" / "templates"
STATIC_DIR = Path(__file__).parent.parent.parent / "src" / "evidenceengine" / "static"


# ── CORS ─────────────────────────────────────────────────────────────────────

def test_cors_settings_are_not_wildcard():
    """Default cors_origins must not be ['*'] — that + credentials=True is dangerous."""
    assert settings.cors_origins != ["*"], "cors_origins must not be wildcard"


def test_cors_parser_handles_comma_separated():
    from evidenceengine.core.config import Settings
    s = Settings(cors_origins="https://a.example.com,https://b.example.com")  # type: ignore[call-arg]
    assert s.cors_origins == ["https://a.example.com", "https://b.example.com"]


# ── Security headers ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_security_headers_present():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"
    assert resp.headers.get("referrer-policy") == "strict-origin-when-cross-origin"
    assert "content-security-policy" in resp.headers


@pytest.mark.asyncio
async def test_csp_has_no_wildcard_sources():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    csp = resp.headers.get("content-security-policy", "")
    assert "default-src *" not in csp
    assert "script-src *" not in csp


@pytest.mark.asyncio
async def test_csp_script_src_no_third_party_cdn():
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    csp = resp.headers.get("content-security-policy", "")
    assert "jsdelivr.net" not in csp, "CSP must not allowlist jsdelivr.net in production"
    assert "unpkg.com" not in csp, "CSP must not allowlist unpkg.com"


# ── Lifespan (no deprecated on_event) ────────────────────────────────────────

def test_app_uses_lifespan_not_on_event():
    import evidenceengine.api.app as app_module
    source = inspect.getsource(app_module)
    assert "on_event" not in source, "app.py still uses deprecated @app.on_event"
    assert "lifespan" in source


# ── Debug-only route ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_design_system_404_when_debug_false(monkeypatch):
    monkeypatch.setattr(settings, "debug", False)
    app = create_app()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/design-system")
    assert resp.status_code == 404


# ── Self-hosted vendor scripts ────────────────────────────────────────────────

def test_vendor_htmx_exists():
    vendor = STATIC_DIR / "vendor" / "htmx.min.js"
    assert vendor.exists(), "htmx.min.js not vendored"
    assert vendor.stat().st_size > 10_000, "htmx.min.js looks too small"


def test_vendor_alpine_exists():
    vendor = STATIC_DIR / "vendor" / "alpine.min.js"
    assert vendor.exists(), "alpine.min.js not vendored"
    assert vendor.stat().st_size > 10_000, "alpine.min.js looks too small"


def test_base_html_uses_vendor_scripts():
    src = (TEMPLATES_DIR / "base.html").read_text()
    assert "/static/vendor/htmx.min.js" in src
    assert "/static/vendor/alpine.min.js" in src
    assert "unpkg.com" not in src, "base.html still references unpkg CDN"


# ── Cache-busting ─────────────────────────────────────────────────────────────

def test_base_html_css_has_version_query():
    src = (TEMPLATES_DIR / "base.html").read_text()
    assert "app.css?v=" in src, "app.css missing cache-busting version"


# ── .env.example completeness ─────────────────────────────────────────────────

def test_env_example_has_required_keys():
    env_example = Path(__file__).parent.parent.parent / ".env.example"
    assert env_example.exists(), ".env.example missing"
    content = env_example.read_text()
    for key in ("DATABASE_URL", "OPENAI_API_KEY", "CORS_ORIGINS", "SENTRY_DSN", "DEBUG"):
        assert key in content, f".env.example missing {key}"


# ── Dockerfile ────────────────────────────────────────────────────────────────

def test_dockerfile_exists_and_is_multistage():
    dockerfile = Path(__file__).parent.parent.parent / "Dockerfile"
    assert dockerfile.exists(), "Dockerfile missing"
    content = dockerfile.read_text()
    assert "FROM node" in content, "Dockerfile missing node CSS build stage"
    assert "FROM python" in content, "Dockerfile missing Python app stage"
    assert "USER ee" in content, "Dockerfile must run as non-root user"
    assert "--reload" not in content, "Dockerfile must not use --reload in production"


# ── requirements.txt ─────────────────────────────────────────────────────────

def test_requirements_txt_exists():
    req = Path(__file__).parent.parent.parent / "requirements.txt"
    assert req.exists(), "requirements.txt missing"
    lines = [l for l in req.read_text().splitlines() if l and not l.startswith("#")]
    assert len(lines) >= 20, f"requirements.txt looks too short ({len(lines)} lines)"
    assert any("fastapi" in l.lower() for l in lines), "fastapi not in requirements.txt"
    assert any("sqlalchemy" in l.lower() for l in lines), "sqlalchemy not in requirements.txt"
