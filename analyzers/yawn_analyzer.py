"""
Yawn Detection Module.
Calculates Mouth Aspect Ratio (MAR) to detect yawning and early signs of drowsiness.
"""

from scipy.spatial import distance as dist
from config import (
    MAR_THRESHOLD,
    YAWN_CONSEC_FRAMES,
    DLIB_LEFT_MOUTH,
    DLIB_RIGHT_MOUTH,
    DLIB_TOP_LIP,
    DLIB_BOTTOM_LIP,
    DLIB_TOP_INNER_LIP,
    DLIB_BOTTOM_INNER_LIP
)

class YawnAnalyzer:
    def __init__(self, mar_threshold=MAR_THRESHOLD, consec_frames=YAWN_CONSEC_FRAMES):
        self.mar_threshold = mar_threshold
        self.consec_frames = consec_frames
        self.frame_counter = 0
        self.total_yawns = 0
        self.is_currently_yawning = False

    def calculate_mar(self, landmarks):
        """
        Calculates Mouth Aspect Ratio:
        MAR = (||top_lip - bottom_lip|| + ||top_inner - bottom_inner||) / (2 * ||left - right||)
        """
        try:
            top_lip = landmarks[DLIB_TOP_LIP]
            bottom_lip = landmarks[DLIB_BOTTOM_LIP]
            top_inner = landmarks[DLIB_TOP_INNER_LIP]
            bottom_inner = landmarks[DLIB_BOTTOM_INNER_LIP]
            left_mouth = landmarks[DLIB_LEFT_MOUTH]
            right_mouth = landmarks[DLIB_RIGHT_MOUTH]

            v1 = dist.euclidean(top_lip, bottom_lip)
            v2 = dist.euclidean(top_inner, bottom_inner)
            horizontal = dist.euclidean(left_mouth, right_mouth)

            if horizontal == 0:
                return 0.0

            mar = (v1 + v2) / (2.0 * horizontal)
            return float(mar)
        except Exception:
            return 0.0

    def analyze(self, landmarks):
        """
        Analyzes mouth state.
        Returns:
            dict containing MAR, is_yawning status, yawn count, and yawn score.
        """
        if landmarks is None or len(landmarks) < 68:
            return {
                "mar": 0.0,
                "is_yawning": False,
                "yawn_count": self.total_yawns,
                "yawn_score": 1.0
            }

        mar = self.calculate_mar(landmarks)

        if mar > self.mar_threshold:
            self.frame_counter += 1
            if self.frame_counter >= self.consec_frames:
                if not self.is_currently_yawning:
                    self.total_yawns += 1
                    self.is_currently_yawning = True
        else:
            self.frame_counter = 0
            self.is_currently_yawning = False

        yawn_score = 0.25 if self.is_currently_yawning else 1.0

        return {
            "mar": round(mar, 3),
            "is_yawning": self.is_currently_yawning,
            "yawn_count": self.total_yawns,
            "yawn_score": yawn_score
        }
