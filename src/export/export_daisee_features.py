from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.utils import ensure_parent
from src.pipelines.infer_image import ImageInferencePipeline

FEATURE_KEYS = [
    "face_presence",
    "gaze_forward",
    "posture_engaged",
    "brow_tension",
    "mouth_downturn",
    "eye_squint",
    "attention_score",
    "affect_score",
    "blocked_score",
]


def _sample_frame_indices(video_path: str, num_frames: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if total <= 0:
        return np.zeros((num_frames,), dtype=np.int64)
    return np.linspace(0, max(0, total - 1), num=num_frames, dtype=np.int64)


def _extract_clip_features(video_path: str, num_frames: int, image_size: tuple[int, int], pipe: ImageInferencePipeline) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    targets = set(_sample_frame_indices(video_path, num_frames).tolist())
    seq = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in targets:
            frame = cv2.resize(frame, image_size)
            _, _, scores, feats = pipe.run_frame(frame)
            vec = np.array([
                feats["face_presence"],
                feats["gaze_forward"],
                feats["posture_engaged"],
                feats["brow_tension"],
                feats["mouth_downturn"],
                feats["eye_squint"],
                scores["attention_score"],
                scores["affect_score"],
                scores["blocked_score"],
            ], dtype=np.float32)
            seq.append(vec)
        i += 1
    cap.release()
    if not seq:
        seq = [np.zeros((len(FEATURE_KEYS),), dtype=np.float32) for _ in range(num_frames)]
    while len(seq) < num_frames:
        seq.append(seq[-1])
    return np.stack(seq[:num_frames], axis=0)


def export_features(manifest_csv: str, dataset_root: str, out_dir: str, out_manifest_csv: str, num_frames: int = 32, image_size=(224, 224)):
    df = pd.read_csv(manifest_csv)
    root = Path(dataset_root)
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    pipe = ImageInferencePipeline()

    paths = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="export_daisee_features"):
        video_abs = root / row["video_path"]
        feat_path = out_root / f"{row['clip_id']}.npz"
        feat = _extract_clip_features(str(video_abs), num_frames=num_frames, image_size=image_size, pipe=pipe)
        np.savez_compressed(feat_path, features=feat, feature_keys=np.array(FEATURE_KEYS, dtype=object))
        paths.append(str(feat_path.as_posix()))

    df = df.copy()
    df["feature_path"] = paths
    ensure_parent(out_manifest_csv)
    df.to_csv(out_manifest_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out_dir", default="data/interim/daisee_proxy_features")
    parser.add_argument("--out_manifest_csv", default="data/interim/daisee_manifest_with_features.csv")
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    export_features(
        manifest_csv=args.manifest_csv,
        dataset_root=args.dataset_root,
        out_dir=args.out_dir,
        out_manifest_csv=args.out_manifest_csv,
        num_frames=args.num_frames,
        image_size=(args.image_size, args.image_size),
    )
    print(f"saved manifest with feature paths: {args.out_manifest_csv}")


if __name__ == "__main__":
    main()
