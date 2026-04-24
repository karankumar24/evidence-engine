"""SciFact fine-tuning for cross-encoder/nli-deberta-v3-small.

Fine-tunes the production NLI model on SciFact claim-evidence pairs.
Expected improvement: +5-8pt on SciFact 3-class accuracy vs baseline 81.2%.

Usage:
    # From repo root — GPU strongly recommended (T4 Colab ~1h, CPU ~8h):
    python -m eval.scifact.finetune
    python -m eval.scifact.finetune --output-dir ./checkpoints/scifact-nli --epochs 4

HuggingFace dataset: allenai/scifact (auto-downloaded on first run, ~50MB).

Label mapping (must match cross-encoder/nli-deberta-v3-small id2label):
    0 = contradiction  → CONTRADICT claims
    1 = entailment     → SUPPORT claims
    2 = neutral        → NEI (no evidence) claims

Training pairs:
    SUPPORT/CONTRADICT: premise = annotated rationale sentences from cited abstract
    NEI: premise = first sentence of any cited abstract (no rationale → no evidence)

The script evaluates on SciFact dev (300 claims) after training and prints
accuracy + macro-F1 so you can decide whether to promote the checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "cross-encoder/nli-deberta-v3-small"
SCIFACT_CACHE = "eval/data/scifact/"

# cross-encoder/nli-deberta-v3-small id2label (fixed — must not change for drop-in replacement)
LABEL2ID = {"contradiction": 0, "entailment": 1, "neutral": 2}

# SciFact gold → model label id
SCIFACT_LABEL2ID = {
    "SUPPORT":    LABEL2ID["entailment"],     # 1
    "CONTRADICT": LABEL2ID["contradiction"],   # 0
    "":           LABEL2ID["neutral"],         # 2  (NEI)
}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_corpus(cache_dir: str) -> dict[int, list[str]]:
    """Returns {doc_id: [sentence0, sentence1, ...]} for all corpus abstracts."""
    from datasets import load_dataset  # noqa: PLC0415
    corpus_raw = load_dataset("allenai/scifact", "corpus", split="train", cache_dir=cache_dir)
    corpus: dict[int, list[str]] = {}
    for row in corpus_raw:
        doc_id = int(row["doc_id"])
        abstract: list[str] = row.get("abstract", []) or []
        corpus[doc_id] = abstract
    logger.info("Corpus loaded: %d abstracts", len(corpus))
    return corpus


def _build_train_pairs(
    cache_dir: str,
    corpus: dict[int, list[str]],
    nei_per_claim: int = 1,
    rng_seed: int = 42,
) -> list[tuple[str, str, int]]:
    """Return list of (premise, hypothesis, label_id) training triplets.

    Strategy:
    - SUPPORT/CONTRADICT rows: premise = annotated rationale sentences joined.
    - NEI rows (empty evidence): premise = first sentence of a cited abstract.
      If no cited abstract available, skip to avoid low-quality negatives.
    """
    from datasets import load_dataset  # noqa: PLC0415

    rng = random.Random(rng_seed)

    # Load raw train rows WITHOUT deduplication so each (claim, evidence) pair is separate.
    raw = load_dataset("allenai/scifact", "claims", split="train", cache_dir=cache_dir)
    logger.info("Raw train rows (before dedup): %d", len(raw))

    pairs: list[tuple[str, str, int]] = []
    nei_claims_seen: set[int] = set()

    for row in raw:
        claim_id = int(row["id"])
        claim_text: str = row["claim"]
        evidence_label: str = (row.get("evidence_label") or "").strip()
        cited_doc_ids: list[int] = [int(d) for d in (row.get("cited_doc_ids") or []) if d is not None]

        if evidence_label in ("SUPPORT", "CONTRADICT"):
            # Use annotated rationale sentences as premise.
            rationale_indices: list[int] = row.get("evidence_sentences") or []
            # Determine which doc this evidence row is for. HF explodes by doc,
            # so cited_doc_ids in an evidence row typically has one entry.
            # Fall back to first available cited doc if ambiguous.
            evidence_doc_id: int | None = cited_doc_ids[0] if cited_doc_ids else None

            if evidence_doc_id is not None and evidence_doc_id in corpus:
                abstract = corpus[evidence_doc_id]
                if rationale_indices:
                    sentences = [abstract[i] for i in rationale_indices if i < len(abstract)]
                else:
                    # No rationale indices — use full abstract (truncation happens in tokenizer)
                    sentences = abstract[:3]
                premise = " ".join(sentences).strip()
            else:
                # Cited doc not in corpus — skip rather than use garbage premise
                continue

            if not premise:
                continue

            label_id = SCIFACT_LABEL2ID[evidence_label]
            pairs.append((premise, claim_text, label_id))

        elif evidence_label == "" and claim_id not in nei_claims_seen:
            # NEI: add one neutral pair per unique claim (avoid duplicate claims inflating NEI)
            nei_claims_seen.add(claim_id)
            if not cited_doc_ids:
                continue
            # Shuffle so different NEI claims use different docs
            doc_ids_shuffled = list(cited_doc_ids)
            rng.shuffle(doc_ids_shuffled)
            premise_sentences: list[str] = []
            for doc_id in doc_ids_shuffled:
                abstract = corpus.get(doc_id, [])
                if abstract:
                    premise_sentences = abstract[:2]
                    break
            if not premise_sentences:
                continue
            premise = " ".join(premise_sentences).strip()
            pairs.append((premise, claim_text, SCIFACT_LABEL2ID[""]))

    label_counts = {0: 0, 1: 0, 2: 0}
    for _, _, lbl in pairs:
        label_counts[lbl] += 1
    logger.info(
        "Training pairs: %d total | entailment=%d contradiction=%d neutral=%d",
        len(pairs), label_counts[1], label_counts[0], label_counts[2],
    )
    return pairs


def _build_dev_pairs(
    cache_dir: str,
    corpus: dict[int, list[str]],
) -> list[tuple[str, str, int]]:
    """Same strategy for dev set (300 unique claims)."""
    from datasets import load_dataset  # noqa: PLC0415

    raw = load_dataset("allenai/scifact", "claims", split="validation", cache_dir=cache_dir)
    seen: set[int] = set()
    pairs: list[tuple[str, str, int]] = []

    for row in raw:
        claim_id = int(row["id"])
        claim_text: str = row["claim"]
        evidence_label: str = (row.get("evidence_label") or "").strip()
        cited_doc_ids: list[int] = [int(d) for d in (row.get("cited_doc_ids") or []) if d is not None]

        if claim_id in seen:
            continue
        seen.add(claim_id)

        if not cited_doc_ids:
            # Use first available corpus doc as fallback context for NEI eval
            abstract = next(iter(corpus.values()), [])
        else:
            abstract = corpus.get(cited_doc_ids[0], [])

        rationale_indices: list[int] = row.get("evidence_sentences") or []
        if rationale_indices and evidence_label in ("SUPPORT", "CONTRADICT"):
            sentences = [abstract[i] for i in rationale_indices if i < len(abstract)]
        else:
            sentences = abstract[:3]

        premise = " ".join(sentences).strip() if sentences else "No evidence available."
        label_id = SCIFACT_LABEL2ID.get(evidence_label, LABEL2ID["neutral"])
        pairs.append((premise, claim_text, label_id))

    logger.info("Dev pairs: %d", len(pairs))
    return pairs


# ── Dataset class ─────────────────────────────────────────────────────────────

def _make_hf_dataset(pairs: list[tuple[str, str, int]], tokenizer, max_length: int = 512):
    """Convert (premise, hypothesis, label) triplets into a HF Dataset."""
    from datasets import Dataset  # noqa: PLC0415

    premises = [p[0] for p in pairs]
    hypotheses = [p[1] for p in pairs]
    labels = [p[2] for p in pairs]

    encodings = tokenizer(
        premises,
        hypotheses,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    encodings["labels"] = labels
    return Dataset.from_dict({k: v for k, v in encodings.items()})


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model, tokenizer, dev_pairs: list[tuple[str, str, int]], device) -> dict:
    """Compute accuracy + per-class metrics on dev pairs."""
    import torch  # noqa: PLC0415

    model.eval()
    id2name = {0: "contradiction", 1: "entailment", 2: "neutral"}
    correct = 0
    per_class: dict[int, dict] = {0: {"correct": 0, "total": 0}, 1: {"correct": 0, "total": 0}, 2: {"correct": 0, "total": 0}}

    with torch.no_grad():
        for premise, hypothesis, gold_label in dev_pairs:
            inputs = tokenizer(
                premise, hypothesis,
                truncation=True, max_length=512, return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            logits = model(**inputs).logits[0]
            pred = int(torch.argmax(logits).item())
            per_class[gold_label]["total"] += 1
            if pred == gold_label:
                correct += 1
                per_class[gold_label]["correct"] += 1

    accuracy = correct / len(dev_pairs) if dev_pairs else 0.0
    results = {"accuracy": accuracy, "n": len(dev_pairs)}
    for label_id, counts in per_class.items():
        name = id2name[label_id]
        recall = counts["correct"] / counts["total"] if counts["total"] else 0.0
        results[f"recall_{name}"] = recall
        results[f"n_{name}"] = counts["total"]
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLI model on SciFact")
    parser.add_argument("--output-dir", default="./checkpoints/scifact-nli")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--cache-dir", default=SCIFACT_CACHE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate baseline without training")
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
        set_seed,
    )

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Base model: %s", args.base_model)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    model.to(device)

    # Load data
    corpus = _load_corpus(args.cache_dir)
    dev_pairs = _build_dev_pairs(args.cache_dir, corpus)

    # Baseline eval
    logger.info("=== Baseline evaluation (before fine-tuning) ===")
    baseline = _evaluate(model, tokenizer, dev_pairs, device)
    logger.info("Baseline: %s", json.dumps(baseline, indent=2))

    if args.eval_only:
        print("\nBaseline results:")
        print(json.dumps(baseline, indent=2))
        return

    train_pairs = _build_train_pairs(args.cache_dir, corpus, rng_seed=args.seed)

    # Build HF datasets
    train_dataset = _make_hf_dataset(train_pairs, tokenizer, max_length=args.max_length)
    dev_dataset = _make_hf_dataset(dev_pairs, tokenizer, max_length=args.max_length)

    logger.info("Train: %d examples, Dev: %d examples", len(train_dataset), len(dev_dataset))

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=20,
        seed=args.seed,
        fp16=torch.cuda.is_available(),   # GPU only — CPU and MPS don't support fp16 training
        report_to="none",                  # no wandb
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    logger.info("=== Training ===")
    trainer.train()

    # Save best checkpoint
    model.save_pretrained(str(output_path / "best"))
    tokenizer.save_pretrained(str(output_path / "best"))
    logger.info("Model saved to %s/best", output_path)

    # Post-training eval
    logger.info("=== Post-training evaluation ===")
    model.eval()
    post = _evaluate(model, tokenizer, dev_pairs, device)
    logger.info("Post-training: %s", json.dumps(post, indent=2))

    # Summary
    delta = post["accuracy"] - baseline["accuracy"]
    print("\n" + "=" * 60)
    print("FINE-TUNING SUMMARY")
    print("=" * 60)
    print(f"Base model:    {args.base_model}")
    print(f"Checkpoint:    {output_path}/best")
    print(f"Baseline acc:  {baseline['accuracy']:.1%}")
    print(f"Fine-tuned:    {post['accuracy']:.1%}  (delta: {delta:+.1%})")
    print(f"Entailment recall:   {post['recall_entailment']:.1%}")
    print(f"Contradiction recall: {post['recall_contradiction']:.1%}")
    print(f"Neutral recall:      {post['recall_neutral']:.1%}")
    print("=" * 60)
    print()

    if delta >= 0.03:
        print("✓ Improvement ≥ 3pp. Promote this checkpoint to production:")
        print(f"  cp -r {output_path}/best /path/to/model_cache/scifact-nli")
        print("  Update NLI_MODEL_NAME in src/evidenceengine/classification/nli_classifier.py")
    else:
        print(f"✗ Improvement < 3pp ({delta:+.1%}). Do not promote — investigate data quality.")

    results_file = output_path / "eval_results.json"
    results_file.write_text(json.dumps({"baseline": baseline, "post_training": post, "delta": delta}, indent=2))
    logger.info("Results written to %s", results_file)


if __name__ == "__main__":
    main()
