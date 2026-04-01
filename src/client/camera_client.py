import argparse

import cv2
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--camera_id", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera_id)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        resp = requests.post(args.api, files={"file": ("frame.jpg", buf.tobytes(), "image/jpeg")}, timeout=5)
        data = resp.json()
        text = f"{data['state']} {data['state_name_zh']} conf={data['confidence']:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("camera_client", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
