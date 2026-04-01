from pathlib import Path

import cv2
import numpy as np

from src.datasets.daisee_manifest import MANIFEST_COLUMNS, build_manifest


def _make_dummy_video(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (64, 64))
    for _ in range(5):
        w.write(np.zeros((64, 64, 3), dtype=np.uint8))
    w.release()


def test_manifest_schema(tmp_path):
    root = tmp_path / "daisee"
    _make_dummy_video(root / "User01" / "clip_a.mp4")
    out_csv = tmp_path / "manifest.csv"
    df = build_manifest(str(root), str(out_csv), strict_check=True)
    assert list(df.columns) == MANIFEST_COLUMNS
    assert len(df) == 1
