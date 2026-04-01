from src.models.expression_head import negative_affect_proxy
from src.models.face_detector import FaceDetector
from src.models.face_landmarks import FaceLandmarksEstimator
from src.models.pose_estimator import PoseEstimator
from src.models.state_fusion import fuse_scores
from src.rules.state_mapper import map_state


class ImageInferencePipeline:
    def __init__(self):
        self.face_det = FaceDetector()
        self.face_lm = FaceLandmarksEstimator()
        self.pose = PoseEstimator()

    def run_frame(self, frame):
        fd = self.face_det.predict(frame)
        fl = self.face_lm.estimate(frame)
        po = self.pose.estimate(frame)
        aff = negative_affect_proxy(fl)
        scores = fuse_scores(float(fd.present), fl.gaze_forward, po.posture_engaged, aff)
        state, conf = map_state(scores.attention_score, scores.affect_score, scores.blocked_score, float(fd.present))
        features = {
            "face_presence": float(fd.present),
            "gaze_forward": fl.gaze_forward,
            "posture_engaged": po.posture_engaged,
            "brow_tension": fl.brow_tension,
            "mouth_downturn": fl.mouth_downturn,
            "eye_squint": fl.eye_squint,
        }
        return state, conf, scores.__dict__, features
