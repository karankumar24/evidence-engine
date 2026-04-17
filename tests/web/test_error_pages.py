"""HTML-vs-JSON content negotiation for 404/500 error responses."""

from fastapi.testclient import TestClient

from evidenceengine.api.app import app


client = TestClient(app)


def test_unknown_html_route_returns_html_404() -> None:
    r = client.get("/does-not-exist", headers={"Accept": "text/html"})
    assert r.status_code == 404
    assert "text/html" in r.headers["content-type"]
    assert "404" in r.text
    assert "Back to runs" in r.text


def test_unknown_api_route_returns_json_404() -> None:
    r = client.get("/api/packets/00000000-0000-0000-0000-000000000000")
    assert r.status_code == 404
    assert "application/json" in r.headers["content-type"]
    body = r.json()
    assert body["error"]["code"].startswith("HTTP_")


def test_unknown_route_without_html_accept_returns_json() -> None:
    r = client.get("/does-not-exist", headers={"Accept": "application/json"})
    assert r.status_code == 404
    assert "application/json" in r.headers["content-type"]
