from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

from .utils import load_config


def main(config_path: str) -> None:
    cfg = load_config(config_path)

    model_name = cfg["model"]["model_name"]
    processed_dir = Path(cfg["data"]["processed_dir"])
    train_csv = processed_dir / "train.csv"

    source_col = cfg["data"]["source_col"]
    target_col = cfg["data"]["target_col"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # "Vocab size" evidence
    vocab_size = tokenizer.vocab_size if tokenizer.vocab_size is not None else len(tokenizer)
    print("=== Tokenizer Inspection ===")
    print(f"Model/tokenizer: {model_name}")
    print(f"Vocabulary size: {vocab_size}")

    if not train_csv.exists():
        print(f"\nNOTE: {train_csv} not found yet. Run data_prep first.")
        return

    df = pd.read_csv(train_csv).dropna()
    df = df[[source_col, target_col]].head(3)

    print("\n=== Example tokenization (first 3 training pairs) ===")
    for i, row in df.iterrows():
        src = str(row[source_col])
        tgt = str(row[target_col])

        src_enc = tokenizer(src, add_special_tokens=True)
        tgt_enc = tokenizer(tgt, add_special_tokens=True)

        print(f"\n--- Example {i+1} ---")
        print(f"SRC ({source_col}): {src}")
        print("SRC tokens:", tokenizer.convert_ids_to_tokens(src_enc["input_ids"][:50]))
        print(f"TGT ({target_col}): {tgt}")
        print("TGT tokens:", tokenizer.convert_ids_to_tokens(tgt_enc["input_ids"][:50]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
