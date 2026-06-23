from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import CHRF
from .utils import load_config

chrf_metric = CHRF(word_order=2)  # common setting; stable for sentence-level scoring


def sentence_chrf(reference: str, prediction: str) -> float:
    # sacrebleu CHRF returns an object with .score
    return chrf_metric.sentence_score(prediction, [reference]).score


def main(config_path: str, predictions_path: str) -> None:
    _ = load_config(config_path)  # kept for future extensions
    pred_path = Path(predictions_path)
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {pred_path.resolve()}")

    df = pd.read_csv(pred_path).dropna()
    for col in ["source", "reference", "prediction"]:
        if col not in df.columns:
            raise ValueError(f"Expected columns [source, reference, prediction]. Found: {list(df.columns)}")

    df["chrf_sent"] = [
        sentence_chrf(r, p) for r, p in zip(df["reference"].astype(str), df["prediction"].astype(str))
    ]
    df["src_len"] = df["source"].astype(str).map(lambda s: len(s.split()))
    df["ref_len"] = df["reference"].astype(str).map(lambda s: len(s.split()))
    df["pred_len"] = df["prediction"].astype(str).map(lambda s: len(s.split()))
    df["len_ratio"] = (df["pred_len"] + 1e-6) / (df["ref_len"] + 1e-6)

    worst = df.sort_values("chrf_sent", ascending=True).head(10)
    best = df.sort_values("chrf_sent", ascending=False).head(10)

    out_md = pred_path.parent / "error_analysis_report.md"

    def format_examples(title: str, subdf: pd.DataFrame) -> str:
        lines = [f"## {title}\n"]
        for _, row in subdf.iterrows():
            lines.append(f"- **chrF**: {row['chrf_sent']:.2f} | **len_ratio**: {row['len_ratio']:.2f}")
            lines.append(f"  - SRC: {row['source']}")
            lines.append(f"  - REF: {row['reference']}")
            lines.append(f"  - PRED: {row['prediction']}\n")
        return "\n".join(lines)

    # Quick heuristics for common issues
    too_short = df[df["len_ratio"] < 0.6].sort_values("chrf_sent").head(10)
    too_long = df[df["len_ratio"] > 1.6].sort_values("chrf_sent").head(10)

    report = []
    report.append("# Error Analysis Report\n")
    report.append("This report samples translations and highlights likely error patterns.\n")
    report.append("## Common qualitative error buckets (use these during write-up)\n")
    report.append("- **Under-translation**: missing content (often shorter outputs)\n")
    report.append("- **Over-translation / repetition**: output longer than reference, repeated phrases\n")
    report.append("- **Word order**: Arabic reordering vs. English source\n")
    report.append("- **Named entities / numbers**: transliteration inconsistencies, digit formatting\n")
    report.append("- **Morphology / agreement**: gender/number agreement, definiteness, verb forms\n")
    report.append("- **Punctuation**: Arabic/English punctuation and spacing differences\n")

    report.append(format_examples("Worst 10 (lowest sentence chrF)", worst))
    report.append(format_examples("Best 10 (highest sentence chrF)", best))
    report.append(format_examples("Likely under-translation (len_ratio < 0.6)", too_short))
    report.append(format_examples("Likely over-translation (len_ratio > 1.6)", too_long))

    out_md.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote: {out_md}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--predictions", required=True, help="CSV from evaluate.py (test_predictions.csv)")
    args = parser.parse_args()
    main(args.config, args.predictions)
