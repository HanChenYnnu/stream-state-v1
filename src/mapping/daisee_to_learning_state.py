from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# DAiSEE class levels assumed 0..3 (low -> high)


def map_daisee_to_state(boredom: int, confusion: int, engagement: int, frustration: int) -> str:
    if frustration >= 2 and confusion >= 2:
        return "S4"  # suspected cognitive blockage
    if frustration >= 2:
        return "S3"  # frustration
    if engagement <= 1 and boredom >= 2:
        return "S2"  # high-risk mind wandering
    if engagement <= 2 or boredom >= 1:
        return "S1"  # mild distraction
    return "S0"  # normal progress


def map_row_to_learning_state(row: dict) -> str:
    return map_daisee_to_state(
        int(row["pred_boredom"]),
        int(row["pred_confusion"]),
        int(row["pred_engagement"]),
        int(row["pred_frustration"]),
    )


def apply_mapping(pred_csv: str, out_csv: str):
    df = pd.read_csv(pred_csv)
    df["learning_state"] = df.apply(lambda r: map_daisee_to_state(r["pred_boredom"], r["pred_confusion"], r["pred_engagement"], r["pred_frustration"]), axis=1)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_csv", required=True)
    parser.add_argument("--out_csv", default="results/daisee/mapped_states.csv")
    args = parser.parse_args()
    df = apply_mapping(args.pred_csv, args.out_csv)
    print(f"mapped rows={len(df)} -> {args.out_csv}")


if __name__ == "__main__":
    main()
