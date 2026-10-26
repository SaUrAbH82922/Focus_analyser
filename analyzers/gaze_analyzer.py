"""
Gaze Analysis Module.
Estimates horizontal and vertical gaze direction (Left, Center, Right)
using eye landmark regions and pupil segmentation.
"""

import cv2
import numpy as np
from config import (
    GAZE_CENTER_MIN,
    GAZE_CENTER_MAX,
    GAZE_CENTER_SCORE,
    GAZE_SIDE_SCORE,
    DLIB_LEFT_EYE,
    DLIB_RIGHT_EYE
)

class GazeAnalyzer:
    def __init__(self, center_min=GAZE_CENTER_MIN, center_max=GAZE_CENTER_MAX):
        self.center_min = center_min
        self.center_max = center_max

    def _get_eye_ratio(self, eye_pts, gray_frame):
        """
        Estimates the horizontal ratio of the pupil inside the eye socket.
        Returns a float between 0.0 (far left) and 1.0 (far right).
        """
        try:
            pts = np.array(eye_pts, dtype=np.int32)
            x, y, w, h = cv2.boundingRect(pts)
            
            if w <= 4 or h <= 4:
                return 0.5
            
            # Crop eye region with slight padding
            y1 = max(0, y - 2)
            y2 = min(gray_frame.shape[0], y + h + 2)
            x1 = max(0, x - 2)
            x2 = min(gray_frame.shape[1], x + w + 2)
            
            eye_roi = gray_frame[y1:y2, x1:x2]
            if eye_roi.size == 0:
                return 0.5

            # Apply Gaussian Blur to reduce noise
            blurred = cv2.GaussianBlur(eye_roi, (7, 7), 0)
            
            # Find the darkest pixel / region (pupil)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(blurred)
            pupil_x_in_roi = min_loc[0]
            
            # Ratio of pupil position relative to width
            ratio = pupil_x_in_roi / float(eye_roi.shape[1])
            return float(np.clip(ratio, 0.0, 1.0))
        except Exception:
            return 0.5

    def analyze(self, landmarks, frame):
        """
        Analyzes gaze direction.
        Returns:
            dict containing gaze direction, ratio, and gaze score.
        """
        if landmarks is None or len(landmarks) < 48 or frame is None:
            return {
                "direction": "CENTER",
                "left_ratio": 0.5,
                "right_ratio": 0.5,
                "avg_ratio": 0.5,
                "gaze_score": GAZE_CENTER_SCORE
            }

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        
        left_eye_pts = [landmarks[i] for i in DLIB_LEFT_EYE]
        right_eye_pts = [landmarks[i] for i in DLIB_RIGHT_EYE]

        left_ratio = self._get_eye_ratio(left_eye_pts, gray)
        right_ratio = self._get_eye_ratio(right_eye_pts, gray)
        avg_ratio = (left_ratio + right_ratio) / 2.0

        if avg_ratio < self.center_min:
            direction = "LOOKING_LEFT"
            score = GAZE_SIDE_SCORE
        elif avg_ratio > self.center_max:
            direction = "LOOKING_RIGHT"
            score = GAZE_SIDE_SCORE
        else:
            direction = "CENTER"
            score = GAZE_CENTER_SCORE

        return {
            "direction": direction,
            "left_ratio": round(left_ratio, 2),
            "right_ratio": round(right_ratio, 2),
            "avg_ratio": round(avg_ratio, 2),
            "gaze_score": score
        }
