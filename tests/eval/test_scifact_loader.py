"""Unit tests for eval/scifact/loader.py — mocked HF dataset calls."""
from __future__ import annotations

import pytest

from eval.scifact.loader import (
    SCIFACT_TO_EE,
    SPLIT_MAP,
    ScifactClaim,
    load_scifact,
)


# ---------------------------------------------------------------------------
# Helpers / fake dataset builders
# ---------------------------------------------------------------------------

def _make_fake_claims_dataset(
    n_unique: int,
    rows_per_claim: int = 1,
    *,
    evidence_label: str = "SUPPORT",
    cited_doc_ids: list[int] | None = None,
) -> list[dict]:
    """Return a fake HF claims dataset with optional row explosion."""
    rows = []
    for i in range(n_unique):
        for _ in range(rows_per_claim):
            rows.append({
                "id": i,
                "claim": f"Claim {i}",
                "evidence_label": evidence_label,
                "cited_doc_ids": cited_doc_ids if cited_doc_ids is not None else [1000 + i],
            })
    return rows


def _make_fake_corpus_dataset(n: int = 5) -> list[dict]:
    return [
        {"doc_id": 1000 + i, "title": f"Title {i}", "abstract": [f"Sentence {i}"]}
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test 1: SPLIT_MAP mapping
# ---------------------------------------------------------------------------

def test_split_map_dev_maps_to_validation():
    assert SPLIT_MAP["dev"] == "validation"


def test_split_map_train_maps_to_train():
    assert SPLIT_MAP["train"] == "train"


# ---------------------------------------------------------------------------
# Test 2: SCIFACT_TO_EE label mapping
# ---------------------------------------------------------------------------

def test_scifact_to_ee_labels():
    assert SCIFACT_TO_EE["SUPPORT"] == "supported"
    assert SCIFACT_TO_EE["CONTRADICT"] == "contradicted"
    assert SCIFACT_TO_EE[""] == "insufficient_support"


# ---------------------------------------------------------------------------
# Test 3: Deduplication of exploded rows (train split — no 300 assert)
# ---------------------------------------------------------------------------

def test_load_scifact_deduplicates_exploded_rows(monkeypatch):
    """450 rows (150 claims × 3 evidence rows each) → 150 unique ScifactClaims."""
    fake_claims = _make_fake_claims_dataset(150, rows_per_claim=3)
    fake_corpus = _make_fake_corpus_dataset(150)

    def fake_load_dataset(name, config, split=None, cache_dir=None, **kwargs):
        if config == "claims":
            return fake_claims
        return fake_corpus

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    claims, corpus = load_scifact("train")
    assert len(claims) == 150, f"Expected 150, got {len(claims)}"
    # All ids are unique
    claim_ids = [c.id for c in claims]
    assert len(set(claim_ids)) == len(claim_ids)


# ---------------------------------------------------------------------------
# Test 4: Dev split asserts exactly 300 unique claims
# ---------------------------------------------------------------------------

def test_load_scifact_dev_asserts_300_passes(monkeypatch):
    """300 unique ids → passes the assert."""
    fake_claims = _make_fake_claims_dataset(300, rows_per_claim=1)
    fake_corpus = _make_fake_corpus_dataset(5)

    def fake_load_dataset(name, config, split=None, cache_dir=None, **kwargs):
        if config == "claims":
            return fake_claims
        return fake_corpus

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    claims, corpus = load_scifact("dev")
    assert len(claims) == 300


def test_load_scifact_dev_asserts_300_fails_on_290(monkeypatch):
    """290 unique ids → AssertionError raised."""
    fake_claims = _make_fake_claims_dataset(290, rows_per_claim=1)
    fake_corpus = _make_fake_corpus_dataset(5)

    def fake_load_dataset(name, config, split=None, cache_dir=None, **kwargs):
        if config == "claims":
            return fake_claims
        return fake_corpus

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    with pytest.raises(AssertionError, match="Expected 300"):
        load_scifact("dev")


# ---------------------------------------------------------------------------
# Test 5: NEI claim has empty gold label
# ---------------------------------------------------------------------------

def test_nei_claim_has_empty_gold_label(monkeypatch):
    """Row with evidence_label='' → gold_label == ''."""
    fake_claims = [{"id": 0, "claim": "NEI claim", "evidence_label": "", "cited_doc_ids": []}]
    fake_corpus = _make_fake_corpus_dataset(1)

    def fake_load_dataset(name, config, split=None, cache_dir=None, **kwargs):
        if config == "claims":
            return fake_claims
        return fake_corpus

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    claims, _ = load_scifact("train")
    assert len(claims) == 1
    assert claims[0].gold_label == ""


# ---------------------------------------------------------------------------
# Test 6: SUPPORT claim label
# ---------------------------------------------------------------------------

def test_support_claim_label(monkeypatch):
    """Row with evidence_label='SUPPORT' → gold_label == 'SUPPORT'."""
    fake_claims = [
        {"id": 1, "claim": "Support claim", "evidence_label": "SUPPORT", "cited_doc_ids": [42]}
    ]
    fake_corpus = [{"doc_id": 42, "title": "Paper", "abstract": ["Sentence one."]}]

    def fake_load_dataset(name, config, split=None, cache_dir=None, **kwargs):
        if config == "claims":
            return fake_claims
        return fake_corpus

    monkeypatch.setattr("datasets.load_dataset", fake_load_dataset)

    claims, corpus = load_scifact("train")
    assert len(claims) == 1
    assert claims[0].gold_label == "SUPPORT"
    assert claims[0].cited_doc_ids == [42]
    # Corpus properly loaded
    assert 42 in corpus
    assert corpus[42]["title"] == "Paper"
    assert corpus[42]["abstract"] == ["Sentence one."]
