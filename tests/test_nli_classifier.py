"""Unit tests for evidenceengine.classification.nli_classifier.

All tests are hermetic — no real model download, no HuggingFace network call,
no MPS inference. Transformers / torch primitives are either mocked directly
or accessed only through spies on the tokenizer call.

Locked behaviors (see Plan 02-02):
- NLI_MODEL_NAME + NLI_LABELS constants
- Device = MPS when available, CPU otherwise
- Singleton loads exactly once and fails open on load error
- Threshold mapping rules (needs_review < 0.50 < insufficient < 0.80 ≤ verdict)
- Per-span MAX aggregation (with entail check taking priority on ties)
- fp32 softmax guard (logits.float() before softmax)
- Premise/hypothesis direction (premise=evidence, hypothesis=claim)
- Tokenizer outputs moved to model device
- NLIClassifier conforms to ClassifierBackend Protocol
- classify runs per-span inference in a single asyncio.to_thread call
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
import torch

from evidenceengine.classification import nli_classifier
from evidenceengine.classification.backend import ClassifierBackend
from evidenceengine.classification.nli_classifier import (
    NLI_LABELS,
    NLI_MODEL_NAME,
    NLIClassifier,
    aggregate_nli,
    nli_probs_for_pair,
    nli_probs_to_verdict,
)


@pytest.fixture(autouse=True)
def _reset_singleton_state():
    """Reset module globals before every test so singleton state is isolated."""
    nli_classifier._model = None
    nli_classifier._tokenizer = None
    nli_classifier._device = None
    nli_classifier._load_failed = False
    yield
    nli_classifier._model = None
    nli_classifier._tokenizer = None
    nli_classifier._device = None
    nli_classifier._load_failed = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_model_name_locked():
    assert NLI_MODEL_NAME == "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"


def test_labels_order_locked():
    # HF model card locks id 0=entailment, 1=neutral, 2=contradiction.
    assert NLI_LABELS == ("entailment", "neutral", "contradiction")


# ---------------------------------------------------------------------------
# Device selection + singleton behavior
# ---------------------------------------------------------------------------


def _install_fake_hf(mmod, mtok):
    """Install a fake tokenizer + model pair on the mocked HF classes."""
    fake_model = MagicMock()
    fake_model.to.return_value = fake_model  # model.to(device) returns self
    mmod.from_pretrained.return_value = fake_model
    mtok.from_pretrained.return_value = MagicMock()
    return fake_model


def test_device_mps_when_available():
    with patch("transformers.AutoModelForSequenceClassification") as mmod, patch(
        "transformers.AutoTokenizer"
    ) as mtok, patch("torch.backends.mps.is_available", return_value=True):
        _install_fake_hf(mmod, mtok)
        nli_classifier._ensure_loaded()
    assert nli_classifier._device is not None
    assert nli_classifier._device.type == "mps"


def test_device_cpu_fallback():
    with patch("transformers.AutoModelForSequenceClassification") as mmod, patch(
        "transformers.AutoTokenizer"
    ) as mtok, patch("torch.backends.mps.is_available", return_value=False):
        _install_fake_hf(mmod, mtok)
        nli_classifier._ensure_loaded()
    assert nli_classifier._device is not None
    assert nli_classifier._device.type == "cpu"


def test_singleton_loads_once():
    with patch("transformers.AutoModelForSequenceClassification") as mmod, patch(
        "transformers.AutoTokenizer"
    ) as mtok, patch("torch.backends.mps.is_available", return_value=False):
        _install_fake_hf(mmod, mtok)
        for _ in range(3):
            nli_classifier._ensure_loaded()
    assert mmod.from_pretrained.call_count == 1
    assert mtok.from_pretrained.call_count == 1


def test_load_failure_fails_open():
    with patch("transformers.AutoModelForSequenceClassification") as mmod, patch(
        "transformers.AutoTokenizer"
    ) as mtok:
        mmod.from_pretrained.side_effect = RuntimeError("kaboom")
        mtok.from_pretrained.return_value = MagicMock()

        first = nli_classifier._ensure_loaded()
        second = nli_classifier._ensure_loaded()

    assert first == (None, None, None)
    assert second == (None, None, None)
    assert nli_classifier._load_failed is True
    # Second call must not re-trigger the failing import.
    assert mmod.from_pretrained.call_count == 1


# ---------------------------------------------------------------------------
# Threshold mapping (nli_probs_to_verdict)
# ---------------------------------------------------------------------------


def test_threshold_entail_supported():
    assert nli_probs_to_verdict(0.85, 0.10, 0.05) == ("supported", 0.85)


def test_threshold_contra_contradicted():
    assert nli_probs_to_verdict(0.05, 0.10, 0.85) == ("contradicted", 0.85)


def test_low_max_needs_review():
    # max_p < 0.50 (all classes low) → needs_review. In practice this never fires
    # because the NLI model's softmax always gives at least one class > 0.50.
    # needs_review is assigned by rule-based paths (unresolvable_anchor, etc.),
    # not by this threshold function.
    assert nli_probs_to_verdict(0.40, 0.35, 0.25) == ("needs_review", 0.40)


def test_neutral_insufficient():
    # Max ≥ 0.50 but entail < 0.80 AND contra < 0.80 → insufficient_support.
    assert nli_probs_to_verdict(0.50, 0.45, 0.05) == ("insufficient_support", 0.50)


def test_mid_entail_insufficient():
    assert nli_probs_to_verdict(0.75, 0.15, 0.10) == ("insufficient_support", 0.75)


# ---------------------------------------------------------------------------
# Aggregation (aggregate_nli)
# ---------------------------------------------------------------------------


def test_aggregate_empty_returns_zero():
    # Empty input → (0.0, 0.0, 0.0): zero signal, not fake neutral confidence.
    # Verdict mapper sees max_decisive=0.0 < 0.50 → needs_review at 0.0.
    assert aggregate_nli([]) == (0.0, 0.0, 0.0)


def test_aggregate_max_across_spans():
    spans = [(0.1, 0.2, 0.8), (0.9, 0.05, 0.05), (0.2, 0.5, 0.3)]
    assert aggregate_nli(spans) == (0.9, 0.5, 0.8)


# ---------------------------------------------------------------------------
# Inference-time invariants (premise/hypothesis, fp32 softmax, device move)
# ---------------------------------------------------------------------------


def _prime_loaded_singleton(device_sentinel="DEVICE_SENTINEL"):
    """Bypass _ensure_loaded by installing mocks directly on module state.

    Returns the fake tokenizer (call-recorded) and fake model so tests can
    assert on their invocations.
    """
    captured = {"tok_args": None, "to_device_calls": []}

    class _RecordingInput:
        def __init__(self, name):
            self.name = name

        def to(self, device):
            captured["to_device_calls"].append((self.name, device))
            return self

    def fake_tokenizer(premise, hypothesis, **kwargs):
        captured["tok_args"] = (premise, hypothesis, kwargs)
        return {
            "input_ids": _RecordingInput("input_ids"),
            "attention_mask": _RecordingInput("attention_mask"),
        }

    class _Logits:
        def __init__(self):
            # Mimic a batched tensor where [0] is a fp16 row.
            # torch.zeros(3, dtype=torch.float16) gives us a real tensor whose
            # .float() returns fp32 — the production code relies on this.
            self.logits = torch.zeros((1, 3), dtype=torch.float16)

    fake_model = MagicMock(return_value=_Logits())

    nli_classifier._model = fake_model
    nli_classifier._tokenizer = fake_tokenizer
    nli_classifier._device = device_sentinel
    return fake_model, captured


def test_premise_is_evidence_hypothesis_is_claim():
    _, captured = _prime_loaded_singleton()
    with patch(
        "torch.softmax",
        return_value=torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32),
    ):
        nli_probs_for_pair(premise="THE EVIDENCE", hypothesis="THE CLAIM")
    assert captured["tok_args"] is not None
    assert captured["tok_args"][0] == "THE EVIDENCE"
    assert captured["tok_args"][1] == "THE CLAIM"


def test_inputs_moved_to_device():
    _, captured = _prime_loaded_singleton(device_sentinel="FAKE_DEVICE")
    with patch(
        "torch.softmax",
        return_value=torch.tensor([0.2, 0.5, 0.3], dtype=torch.float32),
    ):
        nli_probs_for_pair("p", "h")
    # Both tokenizer outputs must have had .to(device) called with the model's device.
    devices_used = {call[1] for call in captured["to_device_calls"]}
    names_moved = {call[0] for call in captured["to_device_calls"]}
    assert devices_used == {"FAKE_DEVICE"}
    assert names_moved == {"input_ids", "attention_mask"}


def test_softmax_is_fp32():
    _prime_loaded_singleton()
    recorded = {}

    def spy_softmax(tensor, dim=-1):
        recorded["dtype"] = tensor.dtype
        # Return a valid fp32 1-D tensor of length 3 so downstream .cpu().tolist() works.
        return torch.tensor([0.3, 0.4, 0.3], dtype=torch.float32)

    with patch("torch.softmax", side_effect=spy_softmax):
        result = nli_probs_for_pair("p", "h")

    # The production code calls torch.softmax(logits.float(), dim=-1) — the spy
    # should have received an fp32 tensor even though the model returned fp16.
    assert recorded["dtype"] == torch.float32
    assert result == (pytest.approx(0.3), pytest.approx(0.4), pytest.approx(0.3))


def test_nli_probs_for_pair_returns_none_when_load_failed():
    nli_classifier._load_failed = True
    assert nli_probs_for_pair("p", "h") is None


# ---------------------------------------------------------------------------
# NLIClassifier.classify — async flows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_empty_evidence_returns_insufficient():
    result = await NLIClassifier().classify("claim", [])
    assert result.verdict_type == "insufficient_support"
    assert result.confidence_score == 0.0
    assert "No evidence" in result.reasoning


@pytest.mark.asyncio
async def test_classify_entailment_high_returns_supported(mock_nli_probs):
    mock_nli_probs([(0.9, 0.05, 0.05)])
    result = await NLIClassifier().classify("claim", [{"span_text": "s"}])
    assert result.verdict_type == "supported"
    assert result.confidence_score == pytest.approx(0.9)
    assert result.explanation is None  # Plan 03 fills this


@pytest.mark.asyncio
async def test_classify_contradiction_high_returns_contradicted(mock_nli_probs):
    mock_nli_probs([(0.05, 0.05, 0.9)])
    result = await NLIClassifier().classify("claim", [{"span_text": "s"}])
    assert result.verdict_type == "contradicted"
    assert result.confidence_score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_classify_max_across_spans_picks_strongest_contra(mock_nli_probs):
    # 3 spans: entail-high, neutral-mid, contra-high.
    # Aggregated: (0.9, 0.8, 0.9) — both entail and contra ≥ 0.80.
    # Mapping order (entail check first) → supported with confidence 0.9.
    # This test LOCKS that ordering — future reorderings will break it.
    mock_nli_probs(
        [
            (0.9, 0.05, 0.05),
            (0.1, 0.8, 0.1),
            (0.05, 0.05, 0.9),
        ]
    )
    result = await NLIClassifier().classify(
        "claim",
        [
            {"span_text": "a"},
            {"span_text": "b"},
            {"span_text": "c"},
        ],
    )
    assert result.verdict_type == "supported"
    assert result.confidence_score == pytest.approx(0.9)


@pytest.mark.asyncio
async def test_classify_model_unavailable_returns_needs_review(monkeypatch):
    # Every inference returns None → model unavailable → needs_review.
    monkeypatch.setattr(
        nli_classifier, "nli_probs_for_pair", lambda p, h: None
    )
    result = await NLIClassifier().classify(
        "claim", [{"span_text": "s1"}, {"span_text": "s2"}]
    )
    assert result.verdict_type == "needs_review"
    assert result.confidence_score == 0.0
    assert "NLI model unavailable" in result.reasoning


def test_nli_classifier_satisfies_protocol():
    # runtime_checkable Protocol — isinstance must return True.
    assert isinstance(NLIClassifier(), ClassifierBackend)


@pytest.mark.asyncio
async def test_classify_runs_in_thread(monkeypatch):
    # Per-span inference happens inside ONE asyncio.to_thread call, not N.
    call_count = {"n": 0}
    real_to_thread = asyncio.to_thread

    async def spy_to_thread(func, *args, **kwargs):
        call_count["n"] += 1
        return await real_to_thread(func, *args, **kwargs)

    monkeypatch.setattr(
        nli_classifier, "nli_probs_for_pair", lambda p, h: (0.9, 0.05, 0.05)
    )
    monkeypatch.setattr(nli_classifier.asyncio, "to_thread", spy_to_thread)

    result = await NLIClassifier().classify(
        "c",
        [
            {"span_text": "s1"},
            {"span_text": "s2"},
            {"span_text": "s3"},
        ],
    )
    assert call_count["n"] == 1
    assert result.verdict_type == "supported"
