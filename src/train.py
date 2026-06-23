from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import evaluate
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from .utils import ensure_dir, load_config, set_seed


def build_compute_metrics(tokenizer, bleu_metric, chrf_metric):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Replace -100 in labels as we can't decode them
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [p.strip() for p in decoded_preds]
        decoded_labels = [l.strip() for l in decoded_labels]

        bleu = bleu_metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        chrf = chrf_metric.compute(predictions=decoded_preds, references=decoded_labels)

        gen_lens = [np.count_nonzero(p != tokenizer.pad_token_id) for p in preds]
        return {
            "bleu": bleu["score"],
            "chrf": chrf["score"],
            "gen_len": float(np.mean(gen_lens)),
        }

    return compute_metrics


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    processed_dir = Path(cfg["data"]["processed_dir"])
    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"

    if not train_csv.exists() or not val_csv.exists():
        raise FileNotFoundError("Processed CSVs not found. Run: python -m src.data_prep --config config.yaml")

    model_name = cfg["model"]["model_name"]
    max_src_len = int(cfg["model"]["max_source_length"])
    max_tgt_len = int(cfg["model"]["max_target_length"])

    source_col = cfg["data"]["source_col"]
    target_col = cfg["data"]["target_col"]

    out_dir = ensure_dir(cfg["training"]["output_dir"])

    # Load dataset
    data_files = {"train": str(train_csv), "validation": str(val_csv)}
    raw_ds = load_dataset("csv", data_files=data_files)

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def preprocess(batch: Dict[str, Any]) -> Dict[str, Any]:
        model_inputs = tokenizer(
            batch[source_col],
            max_length=max_src_len,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch[target_col],
            max_length=max_tgt_len,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = raw_ds.map(
        preprocess,
        batched=True,
        remove_columns=raw_ds["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    # Metrics
    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    # Training args
    tcfg = cfg["training"]
    use_fp16 = bool(tcfg.get("fp16", False)) and torch.cuda.is_available()

    args = Seq2SeqTrainingArguments(
        output_dir=str(out_dir),
        seed=seed,
        data_seed=seed,

        num_train_epochs=float(tcfg["num_train_epochs"]),
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),

        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),

        eval_strategy=str(tcfg["evaluation_strategy"]),
        eval_steps=int(tcfg["eval_steps"]),
        save_steps=int(tcfg["save_steps"]),
        logging_steps=int(tcfg["logging_steps"]),

        predict_with_generate=True,
        generation_max_length=int(tcfg["generation_max_length"]),
        generation_num_beams=int(tcfg["generation_num_beams"]),

        fp16=use_fp16,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="chrf",
        greater_is_better=True,
        report_to="none",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(tokenizer, bleu_metric, chrf_metric),
    )

    print("🚀 Starting training...")
    trainer.train()

    print("💾 Saving final model + tokenizer...")
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"✅ Done. Model saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
