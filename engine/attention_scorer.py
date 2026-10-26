"""
Attention Scorer Module.
Implements the multi-modal mathematical attention function:
A = w1*G + w2*E + w3*H + w4*P + w5*R
"""

import numpy as np
from config import (
    WEIGHT_GAZE,
    WEIGHT_EYE,
    WEIGHT_HEAD_POSE,
    WEIGHT_POSTURE,
    WEIGHT_ALERTNESS,
    FOCUS_THRESHOLD,
    DISTRACTED_THRESHOLD
)

class AttentionScorer:
    def __init__(self,
                 w_gaze=WEIGHT_GAZE,
                 w_eye=WEIGHT_EYE,
                 w_head=WEIGHT_HEAD_POSE,
                 w_posture=WEIGHT_POSTURE,
                 w_alert=WEIGHT_ALERTNESS):
        self.w_gaze = w_gaze
        self.w_eye = w_eye
        self.w_head = w_head
        self.w_posture = w_posture
        self.w_alert = w_alert

    def calculate(self,
                  gaze_score: float,
                  eye_score: float,
                  head_score: float,
                  posture_score: float,
                  alertness_score: float,
                  distraction_score: float = 1.0,
                  is_drowsy: bool = False,
                  face_present: bool = True):
        """
        Calculates instantaneous attention score and discrete focus state.
        Returns:
            dict containing attention_score (0.0 - 1.0), state ("FOCUSED", "MODERATE", "DISTRACTED", "DROWSY", "ABSENT"),
            and raw component scores.
        """
        if not face_present:
            return {
                "attention_score": 0.0,
                "attention_pct": 0,
                "state": "ABSENT",
                "components": {
                    "gaze": 0.0,
                    "eye": 0.0,
                    "head": 0.0,
                    "posture": 0.0,
                    "alertness": 0.0
                }
            }

        # Weighted linear combination
        raw_score = (
            self.w_gaze * gaze_score +
            self.w_eye * eye_score +
            self.w_head * head_score +
            self.w_posture * posture_score +
            self.w_alert * alertness_score
        )

        # Apply distraction penalty
        composite_score = raw_score * distraction_score
        composite_score = float(np.clip(composite_score, 0.0, 1.0))

        # State classification
        if is_drowsy:
            state = "DROWSY"
        elif composite_score >= FOCUS_THRESHOLD:
            state = "FOCUSED"
        elif composite_score >= DISTRACTED_THRESHOLD:
            state = "MODERATE"
        else:
            state = "DISTRACTED"

        return {
            "attention_score": round(composite_score, 3),
            "attention_pct": int(round(composite_score * 100)),
            "state": state,
            "components": {
                "gaze": round(gaze_score, 2),
                "eye": round(eye_score, 2),
                "head": round(head_score, 2),
                "posture": round(posture_score, 2),
                "alertness": round(alertness_score, 2)
            }
        }
