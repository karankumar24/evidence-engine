"""Test that core models are importable at the top level without climate-app dependencies."""
import json
import subprocess
import sys

import pytest


def test_core_models_importable():
    # Fresh import in subprocess to avoid cached modules from test runner boot
    result = subprocess.run(
        [sys.executable, "-c",
         "from evidenceengine import DocumentPacket, SourceDocument, Claim, "
         "CitationAnchor, EvidenceSpan, Verdict, ReviewDecision, RunVersion; print('ok')"],
        capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_core_import_no_forbidden_transitive():
    check = (
        "import sys\n"
        "from evidenceengine import DocumentPacket, SourceDocument, Claim, "
        "CitationAnchor, EvidenceSpan, Verdict, ReviewDecision, RunVersion\n"
        "forbidden = ['fastapi','uvicorn','jinja2','openai','sentence_transformers','bm25s','pymupdf']\n"
        "loaded = [m for m in sys.modules if any(f in m for f in forbidden)]\n"
        "import json; print(json.dumps(loaded))"
    )
    result = subprocess.run([sys.executable, "-c", check], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    loaded = json.loads(result.stdout.strip())
    assert loaded == [], f"Forbidden modules loaded: {loaded}"


def test_version_accessible():
    import evidenceengine
    assert evidenceengine.__version__ == "0.1.0"


def test_all_exports_present():
    import evidenceengine
    for name in ["DocumentPacket", "SourceDocument", "Claim", "CitationAnchor",
                 "EvidenceSpan", "Verdict", "ReviewDecision", "RunVersion"]:
        assert hasattr(evidenceengine, name), f"Missing export: {name}"
