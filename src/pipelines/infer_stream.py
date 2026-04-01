import time

import cv2

from src.pipelines.infer_image import ImageInferencePipeline


def run_webcam_loop(camera_id: int = 0):
    cap = cv2.VideoCapture(camera_id)
    pipe = ImageInferencePipeline()
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        state, conf, scores, feats = pipe.run_frame(frame)
        cv2.putText(frame, f"{state} {conf:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("stream_state_demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        time.sleep(0.02)
    cap.release()
    cv2.destroyAllWindows()
