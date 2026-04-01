import numpy as np

from src.pipelines.infer_image import ImageInferencePipeline


def test_image_pipeline_smoke():
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    pipe = ImageInferencePipeline()
    state, conf, scores, feats = pipe.run_frame(frame)
    assert state.startswith('S')
    assert 0.0 <= conf <= 1.0
    assert "attention_score" in scores
    assert "face_presence" in feats
