"""
Posture Analysis Module.
Analyzes upper body and head posture (Upright, Slouching, Leaning Sideways)
using body landmarks or facial geometry roll and tilt.
"""

import numpy as np
from config import (
    POSTURE_GOOD_ANGLE_MAX,
    POSTURE_GOOD_SCORE,
    POSTURE_BAD_SCORE
)

class PostureAnalyzer:
    def __init__(self, max_tilt_deg=POSTURE_GOOD_ANGLE_MAX):
        self.max_tilt_deg = max_tilt_deg
        self.baseline_y = None

    def analyze(self, pose_landmarks=None, head_pose_dict=None, face_bbox=None, frame_shape=(480, 640)):
        """
        Analyzes user posture.
        Accepts optional pose_landmarks (MediaPipe) and/or head_pose_dict (yaw, pitch, roll) and face_bbox.
        Returns:
            dict containing posture_label, tilt_angle, and posture_score.
        """
        tilt_angle = 0.0
        posture_label = "Good"
        score = POSTURE_GOOD_SCORE

        # Mode A: Full Body Pose if available
        if pose_landmarks is not None and len(pose_landmarks) > 12:
            try:
                # Shoulders: 11 (left), 12 (right)
                ls = pose_landmarks[11]
                rs = pose_landmarks[12]
                dx = rs[0] - ls[0]
                dy = rs[1] - ls[1]
                tilt_angle = abs(np.degrees(np.arctan2(dy, dx)))
                if tilt_angle > 90:
                    tilt_angle = abs(180 - tilt_angle)

                if tilt_angle > self.max_tilt_deg:
                    posture_label = "Leaning Sideways"
                    score = POSTURE_BAD_SCORE
                else:
                    posture_label = "Good"
                    score = POSTURE_GOOD_SCORE

                return {
                    "shoulder_angle": round(tilt_angle, 1),
                    "posture_label": posture_label,
                    "posture_score": round(score, 2)
                }
            except Exception:
                pass

        # Mode B: Facial geometry fallback (Roll + Face position)
        if head_pose_dict is not None:
            roll = abs(head_pose_dict.get("roll", 0.0))
            pitch = head_pose_dict.get("pitch", 0.0)
            tilt_angle = roll

            if roll > self.max_tilt_deg * 1.2:
                posture_label = "Head Tilted"
                score = POSTURE_BAD_SCORE + 0.15
            elif pitch > 18.0:
                posture_label = "Slouching Forward"
                score = POSTURE_BAD_SCORE + 0.1
            else:
                posture_label = "Upright"
                score = POSTURE_GOOD_SCORE

            return {
                "shoulder_angle": round(tilt_angle, 1),
                "posture_label": posture_label,
                "posture_score": round(score, 2)
            }

        return {
            "shoulder_angle": 0.0,
            "posture_label": "Normal",
            "posture_score": POSTURE_GOOD_SCORE
        }
