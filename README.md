# k12_stream_state_v1

Phase-1+ prototype for K-12 **streaming learning-state perception** with two parallel tracks:
1) demo/live educational-state inference (S0-S5), and  
2) DAiSEE benchmark workflow (native 4-head prediction + mapped S0-S4).

> This repository provides an engineering baseline. It does **not** claim validated scientific/clinical performance.

## Phase-1 Demo Scope (Preserved)

Current demo path still supports:
- offline video inference (`src/pipelines/infer_video.py`)
- visual proxy features + sliding-window rule fusion
- FastAPI frame inference endpoint
- webcam HTTP client + Gradio webcam demo
- Slurm templates for GPU_L20

Educational states in demo path:
- S0 正常推进
- S1 轻度分心
- S2 高风险走神
- S3 情绪受挫
- S4 疑似认知受阻
- S5 暂时离开 (demo-only absent/away state)

## Environment Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start: Demo/Offline Inference

```bash
python -m src.pipelines.infer_video \
  --input data/raw/demo_videos/sample.mp4 \
  --out_video data/outputs/videos/demo_overlay.mp4 \
  --out_csv data/outputs/csv/demo_states.csv \
  --out_figure data/outputs/figures/demo_timeline.png
```

API service:
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Webcam client:
```bash
python -m src.client.camera_client --api http://127.0.0.1:8000/predict --camera_id 0
```

Optional Gradio webcam demo:
```bash
python -m src.client.webcam_gradio
```

---

## DAiSEE Benchmark Workflow

### 1) Expected dataset layout

```text
/public/home/USERNAME/datasets/DAiSEE/
├─ train.csv            # optional official annotations/splits
├─ val.csv              # optional
├─ test.csv             # optional
├─ .../*.mp4            # DAiSEE videos (any nested layout)
```

Manifest schema produced/used in this repo:
`video_path, clip_id, subject_id, split, boredom, confusion, engagement, frustration`

If official split CSVs exist, they are used; otherwise subject-safe split fallback is applied.

### 2) Generate DAiSEE manifest

```bash
python -m src.datasets.daisee_manifest \
  --dataset_root /public/home/USERNAME/datasets/DAiSEE \
  --out_csv data/interim/daisee_manifest.csv
```

### 3) Export proxy sequence features (optional but recommended)

```bash
python -m src.export.export_daisee_features \
  --manifest_csv data/interim/daisee_manifest.csv \
  --dataset_root /public/home/USERNAME/datasets/DAiSEE \
  --out_dir data/interim/daisee_proxy_features \
  --out_manifest_csv data/interim/daisee_manifest_with_features.csv \
  --num_frames 32 --image_size 160
```

Then set `dataset.manifest_csv` in `configs/daisee.yaml` to the feature manifest for proxy-based models.

### 4) Train

```bash
python -m src.training.train_daisee --config configs/daisee.yaml
```

### 5) Evaluate

```bash
python -m src.training.eval_daisee \
  --config configs/daisee.yaml \
  --checkpoint results/daisee/train_run/best.pt
```

### 6) Mapped educational-state evaluation (S0-S4)

```bash
python -m src.mapping.daisee_to_learning_state \
  --pred_csv results/daisee/eval_run/predictions.csv \
  --out_csv results/daisee/eval_run/mapped_states.csv
```

### 7) Artifacts generated

- `results/daisee/train_run/best.pt`
- `results/daisee/train_run/train_log.csv`
- `results/daisee/train_run/best_val_metrics.json`
- `results/daisee/eval_run/predictions.csv`
- `results/daisee/eval_run/classification_report.json`
- `results/daisee/eval_run/figures/cm_*.png`
- `results/daisee/eval_run/figures/classwise_f1.png`
- `results/daisee/eval_run/figures/mapped_state_timeline.png`

### Note on S5 and DAiSEE

S5 (暂时离开/absent) is **demo-only** and not part of DAiSEE benchmark mapping/evaluation.
DAiSEE mapping targets S0-S4.

## Cluster / Slurm usage (GPU_L20)

Existing demo scripts (preserved):
```bash
sbatch scripts/run_video_l20.slurm
sbatch scripts/run_api_l20.slurm
sbatch scripts/run_dual_l20_vision.slurm
sbatch scripts/run_dual_l20_fusion.slurm
```

New DAiSEE scripts:
```bash
sbatch scripts/run_daisee_manifest_l20.slurm
sbatch scripts/run_daisee_feature_export_l20.slurm
sbatch scripts/run_daisee_train_l20.slurm
sbatch scripts/run_daisee_eval_l20.slurm
```

## Tests

```bash
pytest -q
```

Includes baseline tests for rules/API/pipeline and new DAiSEE manifest/mapping/smoke tests.

## User actions required

1. Accept DAiSEE license terms and download/place dataset locally.  
2. Edit `USERNAME` / `ENV_NAME` placeholders in Slurm scripts and paths/config values.
