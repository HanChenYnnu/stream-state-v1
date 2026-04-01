# k12_stream_state_v1

Phase-1 baseline prototype for **camera/video-based streaming learning-state perception** in K-12 settings.

> This repository is intentionally lightweight: it focuses on practical engineering and interpretable rules, not benchmark claims.

## 1) Project Goal

Given webcam/video frames, extract low-cost visual signals and output a learning-state label:

- **S0: 正常推进**
- **S1: 轻度分心**
- **S2: 高风险走神**
- **S3: 情绪受挫**
- **S4: 疑似认知受阻**
- **S5: 暂时离开**

## 2) Scope of Phase-1

Included:
- Offline video inference (primary workflow)
- Frame-level visual proxy features (face, rough gaze/head proxy, affect proxy, posture proxy)
- Sliding-window temporal aggregation
- Interpretable rule-based state fusion
- Overlay video + CSV + timeline figure outputs
- FastAPI endpoint for frame inference
- Webcam HTTP client + Gradio demo skeleton
- Slurm templates for GPU_L20 cluster usage

Not included yet:
- Audio modality
- Personalized long-term memory
- End-to-end deep temporal model training
- Scientifically validated performance claims

## 3) Repository Structure

See `src/` for modular implementation:
- `models/`: visual feature extractors (mediapipe-based proxies)
- `rules/`: explicit mapping logic from scores to labels
- `pipelines/`: image/video/stream inference
- `api/`: FastAPI service
- `client/`: webcam clients
- `viz/`: overlay and timeline plotting

## 4) Environment Setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 5) Quick Start

### A. Offline video inference (recommended first)

Put a demo video at: `data/raw/demo_videos/sample.mp4`

```bash
python -m src.pipelines.infer_video \
  --input data/raw/demo_videos/sample.mp4 \
  --out_video data/outputs/videos/demo_overlay.mp4 \
  --out_csv data/outputs/csv/demo_states.csv \
  --out_figure data/outputs/figures/demo_timeline.png
```

Outputs:
- overlay video: `data/outputs/videos/`
- per-time-step csv: `data/outputs/csv/`
- timeline figure: `data/outputs/figures/`

### B. Run API service

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### C. Webcam HTTP client

```bash
python -m src.client.camera_client --api http://127.0.0.1:8000/predict --camera_id 0
```

Press `q` to quit.

### D. Gradio webcam demo (optional)

```bash
python -m src.client.webcam_gradio
```

## 6) Rule Baseline (Interpretability)

Per-frame proxies:
- `face_presence`
- `gaze_forward` (nose-eye geometry proxy)
- `posture_engaged` (torso geometry proxy)
- `brow_tension`, `mouth_downturn`, `eye_squint` → `affect_score`

Fused scores:
- `attention_score`
- `affect_score`
- `blocked_score`

Then explicit rules map scores to S0-S5. This is meant as an extensible baseline and should not be interpreted as validated clinical/pedagogical diagnosis.

## 7) Slurm Usage (GPU_L20 templates)

Edit placeholders (`USERNAME`, conda env, paths), then:

```bash
sbatch scripts/run_video_l20.slurm
sbatch scripts/run_api_l20.slurm
sbatch scripts/run_dual_l20_vision.slurm
sbatch scripts/run_dual_l20_fusion.slurm
```

## 8) Tests

```bash
pytest -q
```

## 9) Limitations

- Uses heuristic visual proxies (not robust in all lighting/camera conditions)
- No calibration for individual learners
- No validated causal claim about cognition/emotion
- Mediapipe estimates can degrade under occlusion/extreme pose

## 10) Next Steps

- Add temporal model (e.g., GRU/TCN) on top of current features
- Add audio channel and multimodal fusion
- Add session memory and event summarization
- Add stronger deployment workflow (container, queue, monitoring)
- Integrate with future LangGraph-based tutoring orchestration
