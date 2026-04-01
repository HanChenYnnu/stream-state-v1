from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

from src.api.schemas import PredictResponse
from src.pipelines.infer_image import ImageInferencePipeline
from src.rules.state_mapper import STATE_TO_ZH

app = FastAPI(title="k12_stream_state_v1 API", version="0.1.0")
pipe = ImageInferencePipeline()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict_image(file: UploadFile = File(...)):
    raw = await file.read()
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        return PredictResponse(
            state="S5",
            state_name_zh=STATE_TO_ZH["S5"],
            confidence=0.1,
            scores={"attention_score": 0.0, "affect_score": 0.0, "blocked_score": 0.0},
            features={"face_presence": 0.0},
        )
    state, conf, scores, features = pipe.run_frame(frame)
    return PredictResponse(state=state, state_name_zh=STATE_TO_ZH[state], confidence=conf, scores=scores, features=features)
