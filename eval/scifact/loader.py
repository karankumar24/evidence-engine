"""SciFact dataset loader: HuggingFace allenai/scifact → EvidenceEngine data shapes.

Critical facts (from research, 2026-04-23):
- HF split "validation" = claims_dev.jsonl (300 unique claims). NOT "dev".
- HF builder may yield >300 rows for "validation" because multi-evidence claims
  are exploded (one row per rationale). Always deduplicate on `id` field.
- Gold labels: "SUPPORT" (not "SUPPORTS"), "CONTRADICT" (not "REFUTES"), "" (NEI).
- Corpus abstract is already sentence-split (list[str]) — no NLTK needed.
- Do NOT import from evidenceengine.api / evidenceengine.models (SQLAlchemy hang).
"""
from __future__ import annotations

from dataclasses import dataclass

# Map CLI --split arg to HuggingFace split names
SPLIT_MAP: dict[str, str] = {
    "dev": "validation",   # 300 unique claims (claims_dev.jsonl)
    "train": "train",
}

# SciFact gold label → EvidenceEngine 4-way verdict (for metrics comparison)
SCIFACT_TO_EE: dict[str, str] = {
    "SUPPORT":    "supported",
    "CONTRADICT": "contradicted",
    "":           "insufficient_support",  # NEI = empty evidence dict in HF encoding
}

# EvidenceEngine verdict → SciFact label (for macro-F1 computation)
EE_TO_SCIFACT: dict[str, str] = {
    "supported":            "SUPPORT",
    "contradicted":         "CONTRADICT",
    "insufficient_support": "",
    "needs_review":         "",
}


@dataclass
class ScifactClaim:
    id: int
    claim: str
    gold_label: str          # "SUPPORT", "CONTRADICT", or "" (NEI)
    cited_doc_ids: list[int]


def load_scifact(
    split: str = "dev",
    cache_dir: str = "eval/data/scifact/",
) -> tuple[list[ScifactClaim], dict[int, dict]]:
    """Download (or load from cache) SciFact claims + corpus.

    Args:
        split: "dev" (300 claims) or "train" (809 claims). Defaults to "dev".
        cache_dir: HuggingFace dataset cache directory.

    Returns:
        (claims, corpus) where:
          claims: list of ScifactClaim (300 items for "dev" split)
          corpus: {doc_id: {"title": str, "abstract": list[str]}}
    """
    from datasets import load_dataset  # noqa: PLC0415 — lazy to avoid slow import at collection

    hf_split = SPLIT_MAP[split]

    claims_raw = load_dataset(
        "allenai/scifact",
        "claims",
        split=hf_split,
        cache_dir=cache_dir,
    )
    corpus_raw = load_dataset(
        "allenai/scifact",
        "corpus",
        split="train",  # corpus has one split only
        cache_dir=cache_dir,
    )

    # Deduplicate: HF builder explodes multi-evidence claims into multiple rows.
    # Use first occurrence of each id — cited_doc_ids covers all abstracts.
    seen: set[int] = set()
    claims: list[ScifactClaim] = []
    for row in claims_raw:
        claim_id = int(row["id"])
        if claim_id in seen:
            continue
        seen.add(claim_id)

        # Determine gold label: empty evidence dict → NEI ("").
        # HF encodes NEI as evidence_label=="" across all evidence fields.
        evidence_label: str = row.get("evidence_label", "") or ""
        # Normalize to canonical SciFact label strings
        gold_label = evidence_label if evidence_label in ("SUPPORT", "CONTRADICT") else ""

        cited_doc_ids: list[int] = []
        raw_cited = row.get("cited_doc_ids") or []
        for doc_id in raw_cited:
            if doc_id is not None:
                cited_doc_ids.append(int(doc_id))

        claims.append(ScifactClaim(
            id=claim_id,
            claim=str(row["claim"]),
            gold_label=gold_label,
            cited_doc_ids=cited_doc_ids,
        ))

    expected = 300 if split == "dev" else None
    if expected is not None:
        assert len(claims) == expected, (
            f"Expected {expected} unique claims for split='{split}', "
            f"got {len(claims)}. HF row explosion or split mismatch."
        )

    # Build corpus lookup
    corpus: dict[int, dict] = {}
    for row in corpus_raw:
        doc_id = int(row["doc_id"])
        corpus[doc_id] = {
            "title": str(row["title"]),
            "abstract": list(row["abstract"]),  # already sentence-split list[str]
        }

    return claims, corpus
