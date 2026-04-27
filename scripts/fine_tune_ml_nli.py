"""Fine-tune DeBERTa-v3-small NLI on SciFact + SciNLI (CS/ML subset).

Training data: SciFact (biomedical, task format) + SciNLI (CS/ML, domain coverage)
+ optional hand-written numerical claim examples.

Run prepare_ml_nli_data.py first to build data/ml_nli_combined_train.jsonl.

Hardware requirements:
  - CUDA GPU with ≥8GB VRAM (e.g. Colab A100, RunPod T4/A100)
  - M1 Air OOMs at batch_size>=4 for DeBERTa training
  - If no GPU: use Colab free tier (T4, ~8GB) or RunPod ($0.20/hr T4)

Usage:
  python scripts/fine_tune_ml_nli.py \
      --train data/ml_nli_combined_train.jsonl \
      --base-model cross-encoder/nli-deberta-v3-small \
      --output checkpoints/ml-nli \
      --epochs 4

After training:
  1. Best checkpoint will be at checkpoints/ml-nli/best/
  2. Upload to Fly.io volume: fly sftp shell → put checkpoints/ml-nli/best /data/models/ml-nli
  3. Update config.py nli_model_path to /data/models/ml-nli (or set via env var)
  4. Run eval/run_nli_benchmark.py to measure accuracy
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Label maps
LABEL_TO_INT = {"SUPPORTS": 0, "CONTRADICTS": 1, "NOT_ENOUGH_INFO": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

# HuggingFace label names expected by the model
HF_LABELS = ["entailment", "contradiction", "neutral"]
OUR_TO_HF = {"SUPPORTS": "entailment", "CONTRADICTS": "contradiction", "NOT_ENOUGH_INFO": "neutral"}


def load_nli_data(path: Path) -> list[dict]:
    """Load JSONL NLI data in {premise, hypothesis, label} format."""
    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line.strip())
            label = row["label"]
            if label not in LABEL_TO_INT:
                logger.warning("Skipping unknown label: %s", label)
                continue
            examples.append({
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "label": LABEL_TO_INT[label],
            })
    logger.info("Loaded %d NLI examples from %s", len(examples), path)
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, default=Path("data/ml_nli_combined_train.jsonl"))
    parser.add_argument("--base-model", default="cross-encoder/nli-deberta-v3-small")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/ml-nli"))
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Reduce to 4 if OOM. Use grad-accum to compensate.")
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--val-split", type=float, default=0.05,
                        help="Fraction of training data to use for validation.")
    args = parser.parse_args()

    if not args.train.exists():
        logger.error("Training data not found: %s", args.train)
        logger.error("Run: python scripts/prepare_ml_nli_data.py first")
        raise SystemExit(1)

    try:
        import torch
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )
    except ImportError as e:
        logger.error("Missing dependencies: %s", e)
        logger.error("pip install transformers datasets torch accelerate")
        raise SystemExit(1)

    # Load data
    all_data = load_nli_data(args.train)

    # Train/val split
    import random
    random.shuffle(all_data)
    val_size = max(1, int(len(all_data) * args.val_split))
    val_data = all_data[:val_size]
    train_data = all_data[val_size:]
    logger.info("Train: %d, Val: %d", len(train_data), len(val_data))

    # Load tokenizer + model
    logger.info("Loading base model: %s", args.base_model)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        ignore_mismatched_sizes=True,
    )
    # Update id2label / label2id to use our label names
    model.config.id2label = {0: "SUPPORTS", 1: "CONTRADICTS", 2: "NOT_ENOUGH_INFO"}
    model.config.label2id = {v: k for k, v in model.config.id2label.items()}

    def tokenize(examples: dict) -> dict:
        return tokenizer(
            examples["premise"],
            examples["hypothesis"],
            truncation=True,
            max_length=512,
            padding=False,
        )

    train_ds = Dataset.from_list(train_data).map(tokenize, batched=True)
    val_ds = Dataset.from_list(val_data).map(tokenize, batched=True)

    # Determine device
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"
        logger.warning("MPS detected (M1). DeBERTa training may OOM. Reduce batch_size to 2.")
    else:
        device_str = "cpu"
        logger.warning("No GPU detected. Training will be very slow.")
    logger.info("Training on: %s", device_str)

    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)
    best_dir = output_dir / "best"

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=50,
        report_to="none",  # no wandb by default
        fp16=device_str == "cuda",  # fp16 only on CUDA
        dataloader_num_workers=0,
    )

    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = (preds == labels).mean()
        return {"accuracy": float(acc)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()

    logger.info("Saving best model to %s", best_dir)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    logger.info("Training complete!")
    logger.info("Next steps:")
    logger.info("  1. fly sftp shell → put %s /data/models/ml-nli", best_dir)
    logger.info("  2. Set NLI_MODEL_PATH=/data/models/ml-nli in fly.toml [env]")
    logger.info("  3. fly deploy --strategy rolling")
    logger.info("  4. python eval/run_nli_benchmark.py  # verify accuracy")


if __name__ == "__main__":
    main()
