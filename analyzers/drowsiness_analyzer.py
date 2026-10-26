"""
Drowsiness and Fatigue Analysis Module.
Calculates PERCLOS (Percentage of Eye Closure), blink frequency, and composite drowsiness.
"""

from collections import deque
import time
from config import (
    PERCLOS_WINDOW,
    PERCLOS_THRESHOLD,
    DROWSY_BLINK_RATE,
    ALERT_SCORE,
    DROWSY_SCORE
)

class DrowsinessAnalyzer:
    def __init__(self, window_size=PERCLOS_WINDOW):
        self.window_size = window_size
        self.eye_state_history = deque(maxlen=window_size)
        self.blink_timestamps = deque()

    def analyze(self, eyes_open, is_yawning, blink_count, elapsed_time_sec):
        """
        Calculates drowsiness level and alertness score.
        eyes_open: bool
        is_yawning: bool
        blink_count: int
        elapsed_time_sec: float
        """
        # Record eye state (1 for closed, 0 for open)
        closed_flag = 0 if eyes_open else 1
        self.eye_state_history.append(closed_flag)

        # 1. PERCLOS calculation
        if len(self.eye_state_history) > 0:
            perclos = sum(self.eye_state_history) / float(len(self.eye_state_history))
        else:
            perclos = 0.0

        # 2. Blink rate per minute
        if elapsed_time_sec > 10.0:
            blink_rate = (blink_count / (elapsed_time_sec / 60.0))
        else:
            blink_rate = 15.0  # Normal default prior to sufficient data

        # 3. Drowsiness condition assessment
        is_drowsy = False
        drowsiness_level = perclos  # baseline

        if perclos >= PERCLOS_THRESHOLD:
            is_drowsy = True
            drowsiness_level = min(1.0, drowsiness_level + 0.3)

        if is_yawning:
            is_drowsy = True
            drowsiness_level = min(1.0, drowsiness_level + 0.25)

        if blink_rate > DROWSY_BLINK_RATE and perclos > 0.20:
            is_drowsy = True
            drowsiness_level = min(1.0, drowsiness_level + 0.2)

        # 4. Alertness Score (1.0 = fully alert, 0.0 = completely drowsy)
        alertness_score = max(DROWSY_SCORE, 1.0 - drowsiness_level)

        return {
            "perclos": round(perclos, 3),
            "blink_rate": round(blink_rate, 1),
            "drowsiness_level": round(drowsiness_level, 2),
            "is_drowsy": is_drowsy,
            "alertness_score": round(alertness_score, 2)
        }
