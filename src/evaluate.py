from __future__ import annotations

import argparse
from pathlib import Path

import evaluate
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .utils import load_config, set_seed


@torch.inference_mode()
def generate_batch(model, tokenizer, texts, max_length: int, num_beams: int, device: str):
    enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    gen = model.generate(
        **enc,
        max_length=max_length,
        num_beams=num_beams,
    )
    return tokenizer.batch_decode(gen, skip_special_tokens=True)


def main(config_path: str, checkpoint: str) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    processed_dir = Path(cfg["data"]["processed_dir"])
    test_csv = processed_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError("test.csv not found. Run data_prep first.")

    source_col = cfg["data"]["source_col"]
    target_col = cfg["data"]["target_col"]

    gen_max_len = int(cfg["training"]["generation_max_length"])
    num_beams = int(cfg["training"]["generation_num_beams"])

    ckpt = Path(checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint folder not found: {ckpt.resolve()}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(str(ckpt))
    model = AutoModelForSeq2SeqLM.from_pretrained(str(ckpt)).to(device)
    model.eval()

    ds = load_dataset("csv", data_files={"test": str(test_csv)})["test"]

    bleu_metric = evaluate.load("sacrebleu")
    chrf_metric = evaluate.load("chrf")

    sources = ds[source_col]
    refs = ds[target_col]

    preds = []
    batch_size = 16 if device == "cuda" else 8

    for i in tqdm(range(0, len(sources), batch_size), desc="Generating"):
        batch_src = sources[i : i + batch_size]
        batch_pred = generate_batch(model, tokenizer, batch_src, gen_max_len, num_beams, device)
        preds.extend([p.strip() for p in batch_pred])

    refs_clean = [str(r).strip() for r in refs]

    bleu = bleu_metric.compute(predictions=preds, references=[[r] for r in refs_clean])["score"]
    chrf = chrf_metric.compute(predictions=preds, references=refs_clean)["score"]

    out_path = ckpt / "test_predictions.csv"
    pd.DataFrame(
        {
            "source": sources,
            "reference": refs_clean,
            "prediction": preds,
        }
    ).to_csv(out_path, index=False, encoding="utf-8")

    print("\n=== Test Metrics ===")
    print(f"BLEU : {bleu:.2f}")
    print(f"chrF : {chrf:.2f}")
    print(f"Saved predictions: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--checkpoint", required=True, help="Model folder, e.g. outputs/en-ar")
    args = parser.parse_args()
    main(args.config, args.checkpoint)
