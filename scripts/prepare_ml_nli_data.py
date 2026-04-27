"""Prepare combined NLI training data: SciFact + SciNLI (CS/ML subset).

Label mapping for SciNLI 4-class → 3-class:
  ENTAILMENT   → SUPPORTS
  CONTRADICTING → CONTRADICTS
  NEUTRAL      → NOT_ENOUGH_INFO
  REASONING    → SKIP (noisy, ~30-35% of SciNLI — excluded for cleaner training)

Usage:
  python scripts/prepare_ml_nli_data.py \
      --scifact-dir data/scifact \
      --scifact-train data/scifact/claims_train.jsonl \
      --scifact-corpus data/scifact/corpus.jsonl \
      --scifact-labels data/scifact/scifact_all_labels.jsonl \
      --scifact-nli data/scifact_nli_train.jsonl \
      --scininli-dir data/scininli \
      --output data/ml_nli_combined_train.jsonl \
      --numerical-examples data/ml_nli_numerical_examples.jsonl

The output file is in sentence-pair NLI format:
  {"premise": "...", "hypothesis": "...", "label": "SUPPORTS|CONTRADICTS|NOT_ENOUGH_INFO"}
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


# SciNLI label mapping — REASONING skipped (noisy)
SCININLI_LABEL_MAP = {
    "ENTAILMENT": "SUPPORTS",
    "CONTRADICTING": "CONTRADICTS",
    "NEUTRAL": "NOT_ENOUGH_INFO",
    "REASONING": None,  # SKIP
}

# SciFact label mapping
SCIFACT_LABEL_MAP = {
    "SUPPORT": "SUPPORTS",
    "CONTRADICT": "CONTRADICTS",
    "NOT_ENOUGH_INFO": "NOT_ENOUGH_INFO",
}


def load_scifact_nli(path: Path) -> list[dict]:
    """Load SciFact training data in sentence-pair NLI format.

    Expects the scifact_nli_train.jsonl format produced by the existing
    fine-tuning script (premise=abstract_sentence, hypothesis=claim_text).
    """
    if not path.exists():
        print(f"  WARNING: SciFact NLI file not found at {path}")
        return []

    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            # Map label if needed
            label = SCIFACT_LABEL_MAP.get(row.get("label", ""), row.get("label", ""))
            if label in ("SUPPORTS", "CONTRADICTS", "NOT_ENOUGH_INFO"):
                examples.append({
                    "premise": row["premise"],
                    "hypothesis": row["hypothesis"],
                    "label": label,
                    "source": "scifact",
                })
    print(f"  Loaded {len(examples)} SciFact examples")
    return examples


def load_scininli(data_dir: Path) -> list[dict]:
    """Load SciNLI examples, skip REASONING class.

    SciNLI format: {"sentence1": "...", "sentence2": "...", "gold_label": "..."}
    Download from: https://github.com/msadat3/SciNLI
    """
    examples = []
    skipped_reasoning = 0

    # Try common file patterns
    patterns = ["train.jsonl", "train.json", "SciNLI_train.jsonl", "scinfli_train.jsonl"]
    data_file = None
    for pat in patterns:
        candidate = data_dir / pat
        if candidate.exists():
            data_file = candidate
            break

    if data_file is None:
        print(f"  WARNING: SciNLI train file not found in {data_dir}")
        print("  Expected one of:", patterns)
        print("  Download from: https://github.com/msadat3/SciNLI")
        return []

    with open(data_file) as f:
        for line in f:
            row = json.loads(line.strip())
            raw_label = row.get("gold_label", row.get("label", ""))
            mapped = SCININLI_LABEL_MAP.get(raw_label.upper())

            if mapped is None:
                skipped_reasoning += 1
                continue

            premise = row.get("sentence1", row.get("premise", "")).strip()
            hypothesis = row.get("sentence2", row.get("hypothesis", "")).strip()

            if premise and hypothesis:
                examples.append({
                    "premise": premise,
                    "hypothesis": hypothesis,
                    "label": mapped,
                    "source": "scininli",
                })

    print(f"  Loaded {len(examples)} SciNLI examples (skipped {skipped_reasoning} REASONING)")
    return examples


def load_numerical_examples(path: Path) -> list[dict]:
    """Load hand-written numerical benchmark claim examples.

    Format: {"premise": "...", "hypothesis": "...", "label": "SUPPORTS|CONTRADICTS|NOT_ENOUGH_INFO"}

    These cover: accuracy numbers, BLEU scores, parameter counts, benchmark comparisons.
    Create these manually in data/ml_nli_numerical_examples.jsonl (~50-100 examples).
    """
    if not path.exists():
        print(f"  NOTE: Numerical examples file not found at {path}")
        print("  Skipping. Create manually for best benchmark claim coverage.")
        return []

    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            row["source"] = "numerical_hand_written"
            examples.append(row)
    print(f"  Loaded {len(examples)} hand-written numerical examples")
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare ML NLI training data")
    parser.add_argument("--scifact-nli", type=Path, default=Path("data/scifact_nli_train.jsonl"))
    parser.add_argument("--scininli-dir", type=Path, default=Path("data/scininli"))
    parser.add_argument("--numerical-examples", type=Path, default=Path("data/ml_nli_numerical_examples.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/ml_nli_combined_train.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("Loading training data...")
    scifact = load_scifact_nli(args.scifact_nli)
    scininli = load_scininli(args.scininli_dir)
    numerical = load_numerical_examples(args.numerical_examples)

    all_examples = scifact + scininli + numerical
    random.shuffle(all_examples)

    # Label distribution
    from collections import Counter
    dist = Counter(e["label"] for e in all_examples)
    print(f"\nCombined dataset: {len(all_examples)} examples")
    print("Label distribution:", dict(dist))

    source_dist = Counter(e["source"] for e in all_examples)
    print("Source distribution:", dict(source_dist))

    # Write output
    with open(args.output, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nOutput written to {args.output}")
    print("Next: run scripts/fine_tune_ml_nli.py on a GPU (Colab/RunPod)")


if __name__ == "__main__":
    main()
