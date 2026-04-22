"""Measure recall@5 of the cross-encoder reranker on one sample climate packet.

Informational only — satisfies Phase 3 ROADMAP success criterion #5. Not a
hard gate. Run once on the OLD model (cross-encoder/ms-marco-MiniLM-L6-v2),
once on the NEW model (BAAI/bge-reranker-v2-m3 with pinned revision), log the
delta in the Phase 3 final commit message body.

Requirements:
    - torch (NOT installed by default on macOS dev venv due to linux-only
      marker in pyproject.toml). Install ad-hoc with:
          .venv/bin/uv pip install torch sentence-transformers
    - ~1.1 GB HuggingFace cache for bge-reranker-v2-m3 first run
    - ~110 MB HuggingFace cache for ms-marco-MiniLM-L6-v2 (OLD baseline)

Usage:
    # NEW model (current Plan 01 config):
    python scripts/measure_reranker_recall.py

    # OLD model (temporarily set env override; do NOT commit .env change):
    RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L6-v2 \\
        python scripts/measure_reranker_recall.py

Sample packet: examples/climate-app/sample_packets/carbon_budget_brief

This script is ephemeral reproducibility evidence — it is NOT wired into
pytest, CI, or pre-commit. Keep it standalone.
"""
from __future__ import annotations

import asyncio
import re
from pathlib import Path

from evidenceengine.retrieval.reranker import rerank

PACKET_DIR = (
    Path(__file__).resolve().parent.parent
    / "examples"
    / "climate-app"
    / "sample_packets"
    / "carbon_budget_brief"
)

SOURCE_FILES = [
    "source_01_global_carbon_project.txt",
    "source_02_ipcc_carbon_budget.txt",
]

# Hand-labeled gold spans for recall@5 evaluation. Keys = claim text drawn
# from report.txt. Values = list of known-relevant evidence substrings that
# appear in the source documents. Substring match (case-insensitive) counts
# as a hit, so post-chunking span boundaries do not need to align exactly.
#
# Selection criteria (per plan locked_decisions):
#   - 3-5 claims mapped to known-good spans in the two source documents
#   - each claim's golds all appear verbatim (substring) in at least one source
#   - drawn from distinct sections of report.txt to exercise the reranker
GOLD: dict[str, list[str]] = {
    # Claim 1 — remaining budget for 1.5C, 50% probability
    "The remaining carbon budget consistent with limiting warming to 1.5C "
    "with a 50 percent probability is approximately 500 GtCO2 from 2020.": [
        "1.5°C, 50% probability:  approximately 500 GtCO2",
        "remaining carbon budget of 500 GtCO2 (50% probability, 1.5",
    ],
    # Claim 2 — 2022 fossil emissions
    "Global CO2 emissions from fossil fuels and industry in 2022 reached "
    "36.6 GtCO2.": [
        "Global fossil CO2 emissions in 2022 are estimated at 36.6 GtCO2",
        "2022 fossil emission rates of 36.6 GtCO2 per year",
    ],
    # Claim 3 — NDC ambition gap
    "Current nationally determined contributions would result in emissions "
    "of approximately 53 GtCO2e per year by 2030, compared with 43 GtCO2e "
    "for a 2C pathway.": [
        "approximately 53 GtCO2e per year (median of submitted",
        "2°C pathway in 2030 is approximately\n43 GtCO2e per year",
    ],
    # Claim 4 — net zero timing
    "Global CO2 emissions would need to reach net zero around 2050 to give "
    "a 50 percent chance of limiting warming to 1.5C.": [
        "Global CO2 reaches net zero around 2050",
        "limit warming to 1.5°C with limited overshoot",
    ],
    # Claim 5 — land and ocean sinks
    "Land and ocean carbon sinks together absorbed approximately 5.6 GtCO2 "
    "per year on average.": [
        "Combined sink: approximately 5.6 GtCO2 per year absorbed",
        "Oceanic sink: 2.9 GtCO2 per year",
    ],
}


def _split_into_spans(text: str, target_chars: int = 400) -> list[str]:
    """Split text into ~target_chars-sized spans on paragraph / sentence
    boundaries. Mimics the char-window chunking the real retrieval pipeline
    uses, but intentionally simple — this is ephemeral measurement code.
    """
    # Normalize whitespace, then split on blank lines first (paragraph-ish).
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    spans: list[str] = []
    buf = ""
    for para in paragraphs:
        if not buf:
            buf = para
        elif len(buf) + 1 + len(para) <= target_chars:
            buf = f"{buf}\n{para}"
        else:
            spans.append(buf)
            buf = para
    if buf:
        spans.append(buf)
    return spans


def _load_candidate_spans() -> list[str]:
    """Load every span from every source file in the carbon_budget_brief
    packet. Returns a flat list so rerank() receives the full candidate pool
    (the point of recall@5 is: can the reranker find the relevant spans
    among the full pool, not among a pre-filtered BM25 shortlist).
    """
    spans: list[str] = []
    for filename in SOURCE_FILES:
        path = PACKET_DIR / filename
        text = path.read_text(encoding="utf-8")
        spans.extend(_split_into_spans(text))
    return spans


async def main() -> None:
    spans = _load_candidate_spans()
    print(f"Loaded {len(spans)} candidate spans from {len(SOURCE_FILES)} sources")
    total_gold = sum(len(v) for v in GOLD.values())
    hits = 0
    per_claim: list[tuple[str, int, int]] = []
    for claim, golds in GOLD.items():
        ranked = await rerank(claim, spans)
        top5 = [r["text"] for r in ranked[:5]]
        claim_hits = 0
        for g in golds:
            if any(g.lower() in t.lower() for t in top5):
                claim_hits += 1
                hits += 1
        per_claim.append((claim[:60], claim_hits, len(golds)))

    print("\nPer-claim recall@5:")
    for claim_snippet, h, n in per_claim:
        print(f"  [{h}/{n}] {claim_snippet}...")
    print(f"\nrecall@5: {hits}/{total_gold} = {hits / total_gold:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
