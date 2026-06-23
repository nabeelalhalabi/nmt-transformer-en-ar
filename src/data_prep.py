from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .utils import ensure_dir, is_blank, load_config, set_seed


def word_count(s: str) -> int:
    # Simple length rule based on whitespace tokens (good enough for filtering extremes)
    return len(str(s).strip().split())


def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    raw_path = Path(cfg["data"]["raw_path"])
    processed_dir = ensure_dir(cfg["data"]["processed_dir"])

    source_col = cfg["data"]["source_col"]
    target_col = cfg["data"]["target_col"]

    max_src_words = int(cfg["data"]["max_source_words"])
    max_tgt_words = int(cfg["data"]["max_target_words"])

    if not raw_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {raw_path.resolve()}")

    df = pd.read_csv(raw_path, encoding="utf-8-sig")

    if source_col not in df.columns or target_col not in df.columns:
        raise ValueError(
            f"CSV must contain columns: {source_col!r}, {target_col!r}. "
            f"Found columns: {list(df.columns)}"
        )

    # Keep only needed columns
    df = df[[source_col, target_col]].copy()

    # Clean: strip whitespace, remove empty rows
    df[source_col] = df[source_col].astype(str).map(lambda x: x.strip())
    df[target_col] = df[target_col].astype(str).map(lambda x: x.strip())

    df = df[~df[source_col].map(is_blank)]
    df = df[~df[target_col].map(is_blank)]

    before_dupes = len(df)
    df = df.drop_duplicates(subset=[source_col, target_col], keep="first")
    after_dupes = len(df)

    # Remove very long pairs (rule: drop examples above max word counts)
    df["src_words"] = df[source_col].map(word_count)
    df["tgt_words"] = df[target_col].map(word_count)

    before_len = len(df)
    df = df[(df["src_words"] <= max_src_words) & (df["tgt_words"] <= max_tgt_words)]
    after_len = len(df)

    # Split: train 80%, val 10%, test 10% with fixed seed
    train_df, temp_df = train_test_split(df, test_size=0.2, random_state=seed, shuffle=True)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed, shuffle=True)

    # Drop helper columns before saving
    keep_cols = [source_col, target_col]
    train_df = train_df[keep_cols].reset_index(drop=True)
    val_df = val_df[keep_cols].reset_index(drop=True)
    test_df = test_df[keep_cols].reset_index(drop=True)

    train_path = processed_dir / "train.csv"
    val_path = processed_dir / "val.csv"
    test_path = processed_dir / "test.csv"

    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")

    stats = {
        "raw_rows": int(pd.read_csv(raw_path).shape[0]),
        "after_drop_empty_and_strip": int(before_dupes),
        "removed_exact_duplicates": int(before_dupes - after_dupes),
        "before_length_filter": int(before_len),
        "removed_by_length_filter": int(before_len - after_len),
        "final_rows": int(after_len),
        "splits": {
            "train": int(len(train_df)),
            "val": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "length_rule": {
            "max_source_words": max_src_words,
            "max_target_words": max_tgt_words,
            "note": "Filtered by whitespace word count to remove extreme long pairs.",
        },
    }

    with open(processed_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("✅ Data preparation complete.")
    print(f"Saved: {train_path}, {val_path}, {test_path}")
    print(f"Stats: {processed_dir / 'stats.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()
    main(args.config)
