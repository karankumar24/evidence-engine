"""Unit tests for the NLI second-opinion decision logic.

Tests `should_force_review` in isolation (no model load) to lock the
one-way-downgrade invariant: NLI can force needs_review but MUST NEVER
upgrade or create a contradicted verdict where the LLM didn't.
"""

from evidenceengine.classification.nli_second_opinion import (
    CONTRADICTION_TRIGGER_PROB,
    ENTAILMENT_TRIGGER_PROB,
    NLIJudgment,
    should_force_review,
)


def J(label, prob):
    """Shorthand for NLIJudgment construction in assertions."""
    return NLIJudgment(label=label, probability=prob)


# --- Case 1: NLI forces supported → needs_review on high-conf contradiction ---

def test_supported_downgraded_when_nli_high_conf_contradiction():
    force, reason = should_force_review("supported", J("contradiction", 0.95))
    assert force is True
    assert "NLI second-opinion" in reason
    assert "contradiction" in reason.lower()


def test_supported_not_downgraded_if_contradiction_below_trigger():
    force, _ = should_force_review(
        "supported", J("contradiction", CONTRADICTION_TRIGGER_PROB - 0.01),
    )
    assert force is False


def test_supported_not_downgraded_on_neutral():
    force, _ = should_force_review("supported", J("neutral", 0.99))
    assert force is False


def test_supported_not_downgraded_on_entailment():
    # NLI agreeing with LLM — no action needed.
    force, _ = should_force_review("supported", J("entailment", 0.99))
    assert force is False


# --- Case 2: NLI forces contradicted → needs_review on high-conf entailment ---

def test_contradicted_downgraded_when_nli_high_conf_entailment():
    force, reason = should_force_review("contradicted", J("entailment", 0.92))
    assert force is True
    assert "entailment" in reason.lower()


def test_contradicted_not_downgraded_if_entailment_below_trigger():
    force, _ = should_force_review(
        "contradicted", J("entailment", ENTAILMENT_TRIGGER_PROB - 0.01),
    )
    assert force is False


def test_contradicted_not_downgraded_on_neutral():
    force, _ = should_force_review("contradicted", J("neutral", 0.99))
    assert force is False


# --- One-way invariant: NLI NEVER upgrades. ---

def test_nli_never_creates_contradicted_verdict():
    """No combination of LLM verdict + NLI judgment may produce contradicted."""
    for llm in ("supported", "contradicted", "insufficient_support", "needs_review"):
        for nli_label in ("entailment", "neutral", "contradiction"):
            for p in (0.60, 0.90, 0.99):
                force, reason = should_force_review(llm, J(nli_label, p))
                # force means "force needs_review", NOT "force contradicted".
                # The function can't emit anything beyond a bool + string.
                # Every caller in the pipeline converts `force=True` to
                # verdict_type="needs_review" — covered by the integration test.
                assert isinstance(force, bool)
                assert isinstance(reason, str)
                if not force:
                    assert reason == ""


def test_nli_never_upgrades_needs_review():
    """NLI disagreement with a needs_review verdict changes nothing."""
    for nli_label in ("entailment", "neutral", "contradiction"):
        force, _ = should_force_review("needs_review", J(nli_label, 0.99))
        assert force is False


def test_nli_never_upgrades_insufficient_support():
    """Same for insufficient_support — NLI can't escalate to supported."""
    for nli_label in ("entailment", "neutral", "contradiction"):
        force, _ = should_force_review("insufficient_support", J(nli_label, 0.99))
        assert force is False


def test_none_judgment_is_no_op():
    """A None judgment (model load failed or no evidence) forces nothing."""
    for llm in ("supported", "contradicted", "insufficient_support", "needs_review"):
        force, reason = should_force_review(llm, None)
        assert force is False
        assert reason == ""
