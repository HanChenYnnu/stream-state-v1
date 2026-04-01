import numpy as np
import cv2
from fastapi.testclient import TestClient

from src.api.app import app


def test_predict_schema_keys():
    client = TestClient(app)
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    ok, buf = cv2.imencode('.jpg', img)
    assert ok
    resp = client.post('/predict', files={'file': ('x.jpg', buf.tobytes(), 'image/jpeg')})
    assert resp.status_code == 200
    body = resp.json()
    for k in ["state", "state_name_zh", "confidence", "scores", "features"]:
        assert k in body
