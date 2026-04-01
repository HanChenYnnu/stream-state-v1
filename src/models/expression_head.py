from src.models.face_landmarks import FaceProxyFeatures


def negative_affect_proxy(face: FaceProxyFeatures) -> float:
    return max(0.0, min(1.0, 0.45 * face.brow_tension + 0.3 * face.mouth_downturn + 0.25 * face.eye_squint))
