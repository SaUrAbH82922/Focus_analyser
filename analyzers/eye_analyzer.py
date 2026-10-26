"""
Eye Analysis Module.
Calculates Eye Aspect Ratio (EAR), detects blinks, and scores eye openness.
"""

import numpy as np
from scipy.spatial import distance as dist
from config import (
    EAR_THRESHOLD,
    BLINK_CONSEC_FRAMES,
    EAR_OPEN_SCORE,
    EAR_CLOSED_SCORE,
    DLIB_LEFT_EYE,
    DLIB_RIGHT_EYE
)

class EyeAnalyzer:
    def __init__(self, ear_threshold=EAR_THRESHOLD):
        self.ear_threshold = ear_threshold
        self.blink_counter = 0
        self.total_blinks = 0
        self.eyes_closed_frames = 0
        self.current_state = "OPEN"

    @staticmethod
    def calculate_ear(eye_points):
        """
        Eye Aspect Ratio:
        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        eye_points: 6 (x, y) coordinates
        """
        try:
            # Vertical distances
            A = dist.euclidean(eye_points[1], eye_points[5])
            B = dist.euclidean(eye_points[2], eye_points[4])
            # Horizontal distance
            C = dist.euclidean(eye_points[0], eye_points[3])
            if C == 0:
                return 0.0
            return float((A + B) / (2.0 * C))
        except Exception:
            return 0.0

    def analyze(self, landmarks):
        """
        Analyzes eye state given 68 facial landmarks.
        Returns:
            dict containing EAR metrics, blink count, eyes open status, eye score.
        """
        if landmarks is None or len(landmarks) < 48:
            return {
                "left_ear": 0.0,
                "right_ear": 0.0,
                "avg_ear": 0.0,
                "eyes_open": False,
                "blink_count": self.total_blinks,
                "eye_score": EAR_CLOSED_SCORE,
                "state": "NO_FACE"
            }

        left_eye_pts = [landmarks[i] for i in DLIB_LEFT_EYE]
        right_eye_pts = [landmarks[i] for i in DLIB_RIGHT_EYE]

        left_ear = self.calculate_ear(left_eye_pts)
        right_ear = self.calculate_ear(right_eye_pts)
        avg_ear = (left_ear + right_ear) / 2.0

        is_open = avg_ear >= self.ear_threshold

        if not is_open:
            self.blink_counter += 1
            self.eyes_closed_frames += 1
            self.current_state = "CLOSED"
        else:
            if self.blink_counter >= BLINK_CONSEC_FRAMES:
                self.total_blinks += 1
            self.blink_counter = 0
            self.eyes_closed_frames = 0
            self.current_state = "OPEN"

        # Continuous eye openness score (normalized between 0 and 1)
        if avg_ear >= self.ear_threshold:
            # Scale from threshold to healthy open (~0.35)
            eye_score = min(1.0, 0.6 + 0.4 * ((avg_ear - self.ear_threshold) / 0.15))
        else:
            eye_score = max(0.0, avg_ear / self.ear_threshold * 0.5)

        return {
            "left_ear": round(left_ear, 3),
            "right_ear": round(right_ear, 3),
            "avg_ear": round(avg_ear, 3),
            "eyes_open": is_open,
            "blink_count": self.total_blinks,
            "eye_score": round(eye_score, 2),
            "state": self.current_state,
            "left_points": left_eye_pts,
            "right_points": right_eye_pts
        }
