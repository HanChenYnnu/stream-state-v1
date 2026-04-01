from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

MANIFEST_COLUMNS = [
    "video_path",
    "clip_id",
    "subject_id",
    "split",
    "boredom",
    "confusion",
    "engagement",
    "frustration",
]


def _extract_subject_id(video_rel: str, clip_id: str) -> str:
    parts = Path(video_rel).parts
    for p in parts:
        low = p.lower()
        if "user" in low or "subject" in low or low.startswith("u"):
            return p
    return clip_id.split("_")[0]


def _collect_videos(dataset_root: Path) -> list[Path]:
    return sorted([p for p in dataset_root.rglob("*.mp4") if p.is_file()])


def _read_official_split_annotations(dataset_root: Path) -> Optional[pd.DataFrame]:
    rows = []
    for split in ["train", "val", "test"]:
        csv_path = dataset_root / f"{split}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        required = {"clip_id", "boredom", "confusion", "engagement", "frustration"}
        if not required.issubset(df.columns):
            continue
        df = df.copy()
        df["split"] = split
        rows.append(df)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    return out


def _subject_safe_split(df: pd.DataFrame, val_ratio: float = 0.15, test_ratio: float = 0.15) -> pd.DataFrame:
    subjects = sorted(df["subject_id"].unique())
    n = len(subjects)
    n_test = max(1, int(round(n * test_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    test_sub = set(subjects[:n_test])
    val_sub = set(subjects[n_test : n_test + n_val])

    def f(sub):
        if sub in test_sub:
            return "test"
        if sub in val_sub:
            return "val"
        return "train"

    df["split"] = df["subject_id"].map(f)
    return df


def build_manifest(dataset_root: str, out_csv: str, strict_check: bool = True) -> pd.DataFrame:
    root = Path(dataset_root)
    videos = _collect_videos(root)
    records = []
    for vp in videos:
        rel = vp.relative_to(root).as_posix()
        clip_id = vp.stem
        subject_id = _extract_subject_id(rel, clip_id)
        records.append(
            {
                "video_path": rel,
                "clip_id": clip_id,
                "subject_id": subject_id,
                "split": "unknown",
                "boredom": -1,
                "confusion": -1,
                "engagement": -1,
                "frustration": -1,
            }
        )
    df = pd.DataFrame(records, columns=MANIFEST_COLUMNS)

    anno = _read_official_split_annotations(root)
    if anno is not None:
        label_cols = ["boredom", "confusion", "engagement", "frustration", "split"]
        merged = df.merge(anno[["clip_id", *label_cols]], on="clip_id", how="left", suffixes=("", "_a"))
        for c in label_cols:
            merged[c] = merged[f"{c}_a"].fillna(merged[c])
            merged.drop(columns=[f"{c}_a"], inplace=True)
        df = merged
    else:
        df = _subject_safe_split(df)

    if strict_check:
        valid_rows = []
        for _, row in df.iterrows():
            p = root / row["video_path"]
            if p.exists() and p.stat().st_size > 0:
                valid_rows.append(True)
            else:
                valid_rows.append(False)
        df = df[valid_rows].reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_csv", default="data/interim/daisee_manifest.csv")
    parser.add_argument("--no_strict_check", action="store_true")
    args = parser.parse_args()
    df = build_manifest(args.dataset_root, args.out_csv, strict_check=not args.no_strict_check)
    print(f"manifest rows={len(df)} saved={args.out_csv}")


if __name__ == "__main__":
    main()
