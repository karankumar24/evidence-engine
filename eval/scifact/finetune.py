"""SciFact fine-tuning for cross-encoder/nli-deberta-v3-small.

Fine-tunes the production NLI model on SciFact claim-evidence pairs.
Expected improvement: +5-8pt on SciFact 3-class accuracy vs baseline 81.2%.

Usage:
    # From repo root — MPS (Apple Silicon) runs in ~10-20 min; CPU ~2-3h
    python -m eval.scifact.finetune
    python -m eval.scifact.finetune --output-dir ./checkpoints/scifact-nli --epochs 4
    python -m eval.scifact.finetune --eval-only    # baseline only, no training

Raw data: https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
  data/corpus.jsonl        — evidence corpus (~5,183 abstracts)
  data/claims_train.jsonl  — 809 training claims
  data/claims_dev.jsonl    — 300 dev claims

claims_train.jsonl schema (confirmed from raw data):
  {
    "id": int,
    "claim": str,
    "evidence": {
      "<doc_id>": [{"sentences": [int, ...], "label": "SUPPORT"|"CONTRADICT"}]
    },     # empty {} for NEI claims
    "cited_doc_ids": [int, ...]
  }

Label mapping (must match cross-encoder/nli-deberta-v3-small id2label exactly):
    0 = contradiction  → CONTRADICT claims
    1 = entailment     → SUPPORT claims
    2 = neutral        → NEI (no evidence) claims

Training pairs:
  SUPPORT/CONTRADICT: premise = annotated rationale sentences from cited abstract
  NEI:                premise = first 2 sentences of a cited abstract
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import tarfile
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_MODEL = "cross-encoder/nli-deberta-v3-small"
SCIFACT_CACHE = "eval/data/scifact/"
SCIFACT_S3_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"

# cross-encoder/nli-deberta-v3-small id2label (fixed — must not change for drop-in replacement)
LABEL2ID = {"contradiction": 0, "entailment": 1, "neutral": 2}

# SciFact gold → model label id
SCIFACT_LABEL2ID = {
    "SUPPORT":    LABEL2ID["entailment"],     # 1
    "CONTRADICT": LABEL2ID["contradiction"],   # 0
    "":           LABEL2ID["neutral"],         # 2  (NEI)
}


# ── Data download ─────────────────────────────────────────────────────────────

def _ensure_data(cache_dir: str) -> Path:
    """Download and extract SciFact tarball if not already cached. Returns data/ path."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    data_dir = cache / "data"

    if (data_dir / "corpus.jsonl").exists() and (data_dir / "claims_train.jsonl").exists():
        logger.info("SciFact data already at %s", data_dir)
        return data_dir

    tarball = cache / "data.tar.gz"
    if not tarball.exists():
        logger.info("Downloading SciFact from S3 (~3MB)...")
        urllib.request.urlretrieve(SCIFACT_S3_URL, str(tarball))
        logger.info("Downloaded (%.1f MB)", tarball.stat().st_size / 1e6)

    logger.info("Extracting...")
    with tarfile.open(str(tarball)) as tar:
        tar.extractall(str(cache))

    assert (data_dir / "corpus.jsonl").exists(), f"Expected {data_dir}/corpus.jsonl after extraction"
    return data_dir


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_corpus(data_dir: Path) -> dict[int, list[str]]:
    """Returns {doc_id: [sentence0, sentence1, ...]} for all corpus abstracts."""
    corpus: dict[int, list[str]] = {}
    with open(data_dir / "corpus.jsonl") as f:
        for line in f:
            row = json.loads(line)
            doc_id = int(row["doc_id"])
            abstract: list[str] = row.get("abstract") or []
            corpus[doc_id] = abstract
    logger.info("Corpus: %d abstracts", len(corpus))
    return corpus


def _build_train_pairs(
    data_dir: Path,
    corpus: dict[int, list[str]],
    rng_seed: int = 42,
) -> list[tuple[str, str, int]]:
    """Return (premise, hypothesis, label_id) training triplets from claims_train.jsonl.

    Evidence schema (confirmed from raw data):
      evidence[doc_id_str] = [{"sentences": [int, ...], "label": "SUPPORT"|"CONTRADICT"}]
    Each doc_id maps to a LIST of evidence objects (may have >1 per doc).
    """
    rng = random.Random(rng_seed)
    pairs: list[tuple[str, str, int]] = []
    nei_buffer: list[tuple[str, list[int]]] = []  # (claim_text, cited_doc_ids) for NEI claims

    with open(data_dir / "claims_train.jsonl") as f:
        for line in f:
            row = json.loads(line)
            claim_text: str = row["claim"]
            evidence: dict = row.get("evidence") or {}
            cited_doc_ids: list[int] = [int(d) for d in (row.get("cited_doc_ids") or [])]

            if evidence:
                # One or more evidence documents with rationale sentences
                for doc_id_str, ev_list in evidence.items():
                    doc_id = int(doc_id_str)
                    abstract = corpus.get(doc_id, [])
                    if not abstract:
                        continue
                    # ev_list is a list of evidence objects (usually just 1, but can be >1)
                    for ev in ev_list:
                        label: str = ev.get("label", "")
                        if label not in ("SUPPORT", "CONTRADICT"):
                            continue
                        sent_indices: list[int] = ev.get("sentences") or []
                        if sent_indices:
                            sentences = [abstract[i] for i in sent_indices if i < len(abstract)]
                        else:
                            sentences = abstract[:3]
                        premise = " ".join(sentences).strip()
                        if not premise:
                            continue
                        pairs.append((premise, claim_text, SCIFACT_LABEL2ID[label]))
            else:
                # NEI claim — buffer for balanced sampling
                nei_buffer.append((claim_text, cited_doc_ids))

    # Add NEI pairs (one per claim, using first available cited abstract)
    for claim_text, cited_doc_ids in nei_buffer:
        if not cited_doc_ids:
            continue
        doc_ids_shuffled = list(cited_doc_ids)
        rng.shuffle(doc_ids_shuffled)
        for doc_id in doc_ids_shuffled:
            abstract = corpus.get(doc_id, [])
            if abstract:
                premise = " ".join(abstract[:2]).strip()
                if premise:
                    pairs.append((premise, claim_text, SCIFACT_LABEL2ID[""]))
                break

    label_counts = {0: 0, 1: 0, 2: 0}
    for _, _, lbl in pairs:
        label_counts[lbl] += 1
    logger.info(
        "Train pairs: %d | entailment=%d contradiction=%d neutral=%d",
        len(pairs), label_counts[1], label_counts[0], label_counts[2],
    )
    return pairs


def _build_dev_pairs(
    data_dir: Path,
    corpus: dict[int, list[str]],
) -> list[tuple[str, str, int]]:
    """One (premise, hypothesis, label_id) pair per dev claim (300 claims)."""
    pairs: list[tuple[str, str, int]] = []

    with open(data_dir / "claims_dev.jsonl") as f:
        for line in f:
            row = json.loads(line)
            claim_text: str = row["claim"]
            evidence: dict = row.get("evidence") or {}
            cited_doc_ids: list[int] = [int(d) for d in (row.get("cited_doc_ids") or [])]

            if evidence:
                # Use first available evidence entry
                found = False
                for doc_id_str, ev_list in evidence.items():
                    doc_id = int(doc_id_str)
                    abstract = corpus.get(doc_id, [])
                    for ev in ev_list:
                        label: str = ev.get("label", "")
                        sent_indices: list[int] = ev.get("sentences") or []
                        if sent_indices and abstract:
                            sentences = [abstract[i] for i in sent_indices if i < len(abstract)]
                        else:
                            sentences = abstract[:3]
                        premise = " ".join(sentences).strip() or "No evidence."
                        label_id = SCIFACT_LABEL2ID.get(label, LABEL2ID["neutral"])
                        pairs.append((premise, claim_text, label_id))
                        found = True
                        break
                    if found:
                        break
            else:
                # NEI — use first available cited abstract
                premise = "No evidence."
                for doc_id in cited_doc_ids:
                    abstract = corpus.get(doc_id, [])
                    if abstract:
                        premise = " ".join(abstract[:3]).strip()
                        break
                pairs.append((premise, claim_text, SCIFACT_LABEL2ID[""]))

    logger.info("Dev pairs: %d", len(pairs))
    return pairs


# ── HF Dataset wrapper ────────────────────────────────────────────────────────

def _make_hf_dataset(pairs: list[tuple[str, str, int]], tokenizer, max_length: int = 512):
    """Tokenize (premise, hypothesis, label) triplets into a HF Dataset."""
    from datasets import Dataset  # noqa: PLC0415

    premises   = [p[0] for p in pairs]
    hypotheses = [p[1] for p in pairs]
    labels     = [p[2] for p in pairs]

    # Do NOT pad here — DataCollatorWithPadding pads dynamically per batch,
    # which cuts memory vs padding every sequence to max_length.
    encodings = tokenizer(
        premises, hypotheses,
        truncation=True, max_length=max_length,
    )
    encodings["labels"] = labels
    return Dataset.from_dict(dict(encodings))


# ── Evaluation ────────────────────────────────────────────────────────────────

def _evaluate(model, tokenizer, dev_pairs: list[tuple[str, str, int]], device) -> dict:
    """Accuracy + per-class recall on dev pairs."""
    import torch  # noqa: PLC0415

    id2name = {0: "contradiction", 1: "entailment", 2: "neutral"}
    model.eval()
    correct = 0
    per_class: dict[int, dict] = {k: {"correct": 0, "total": 0} for k in (0, 1, 2)}

    with torch.no_grad():
        for premise, hypothesis, gold in dev_pairs:
            inputs = tokenizer(
                premise, hypothesis,
                truncation=True, max_length=512, return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            pred = int(torch.argmax(model(**inputs).logits[0]).item())
            per_class[gold]["total"] += 1
            if pred == gold:
                correct += 1
                per_class[gold]["correct"] += 1

    accuracy = correct / len(dev_pairs) if dev_pairs else 0.0
    results: dict = {"accuracy": accuracy, "n": len(dev_pairs)}
    for label_id, counts in per_class.items():
        name = id2name[label_id]
        recall = counts["correct"] / counts["total"] if counts["total"] else 0.0
        results[f"recall_{name}"] = recall
        results[f"n_{name}"] = counts["total"]
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fine-tune NLI model on SciFact")
    parser.add_argument("--output-dir",  default="./checkpoints/scifact-nli")
    parser.add_argument("--base-model",  default=BASE_MODEL)
    parser.add_argument("--epochs",      type=int,   default=3)
    parser.add_argument("--lr",          type=float, default=2e-5)
    parser.add_argument("--batch-size",  type=int,   default=4)    # 4 × grad_accum 4 = effective 16
    parser.add_argument("--grad-accum",  type=int,   default=4)    # simulate larger batch
    parser.add_argument("--max-length",  type=int,   default=128)  # p99 token length < 200
    parser.add_argument("--cache-dir",   default=SCIFACT_CACHE)
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--eval-only",   action="store_true",
                        help="Evaluate baseline model on dev set without training")
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    from transformers import (  # noqa: PLC0415
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info("Device: %s", device)
    logger.info("Base model: %s", args.base_model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model)
    model.to(device)

    # Ensure SciFact data is available
    data_dir = _ensure_data(args.cache_dir)
    corpus = _load_corpus(data_dir)
    dev_pairs = _build_dev_pairs(data_dir, corpus)

    # Baseline evaluation
    logger.info("=== Baseline evaluation ===")
    baseline = _evaluate(model, tokenizer, dev_pairs, device)
    logger.info("Baseline accuracy: %.1f%%  (n=%d)", baseline["accuracy"] * 100, baseline["n"])
    for name in ("entailment", "contradiction", "neutral"):
        logger.info("  recall_%s: %.1f%%  (n=%d)", name,
                    baseline[f"recall_{name}"] * 100, baseline[f"n_{name}"])

    if args.eval_only:
        print("\nBaseline results:")
        print(json.dumps(baseline, indent=2))
        return

    train_pairs = _build_train_pairs(data_dir, corpus, rng_seed=args.seed)
    train_dataset = _make_hf_dataset(train_pairs, tokenizer, max_length=args.max_length)
    dev_dataset   = _make_hf_dataset(dev_pairs,   tokenizer, max_length=args.max_length)
    logger.info("Train: %d  Dev: %d", len(train_dataset), len(dev_dataset))

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=50,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=10,
        seed=args.seed,
        fp16=torch.cuda.is_available(),
        bf16=False,
        gradient_checkpointing=True,      # recompute activations during backward — trades speed for memory
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    logger.info("=== Training (device=%s, epochs=%d) ===", device, args.epochs)
    trainer.train()

    model.save_pretrained(str(output_path / "best"))
    tokenizer.save_pretrained(str(output_path / "best"))
    logger.info("Checkpoint saved to %s/best", output_path)

    logger.info("=== Post-training evaluation ===")
    model.eval()
    post = _evaluate(model, tokenizer, dev_pairs, device)

    delta = post["accuracy"] - baseline["accuracy"]
    print("\n" + "=" * 60)
    print("FINE-TUNING SUMMARY")
    print("=" * 60)
    print(f"Base model:     {args.base_model}")
    print(f"Checkpoint:     {output_path}/best")
    print(f"Baseline acc:   {baseline['accuracy']:.1%}")
    print(f"Fine-tuned acc: {post['accuracy']:.1%}  (delta: {delta:+.1%})")
    print(f"  recall_entailment:   {post['recall_entailment']:.1%}  (n={post['n_entailment']})")
    print(f"  recall_contradiction:{post['recall_contradiction']:.1%}  (n={post['n_contradiction']})")
    print(f"  recall_neutral:      {post['recall_neutral']:.1%}  (n={post['n_neutral']})")
    print("=" * 60)

    if delta >= 0.03:
        print(f"\n✓ Improvement ≥ 3pp. Promote checkpoint to production:")
        print(f"  Update NLI_MODEL_NAME in src/evidenceengine/classification/nli_classifier.py")
        print(f"  Point it to the local checkpoint path or push to HuggingFace Hub")
    else:
        print(f"\n✗ Improvement {delta:+.1%} < 3pp. Investigate data quality before promoting.")

    results_file = output_path / "eval_results.json"
    results_file.write_text(json.dumps(
        {"baseline": baseline, "post_training": post, "delta": delta}, indent=2
    ))
    logger.info("Results written to %s", results_file)


if __name__ == "__main__":
    main()
