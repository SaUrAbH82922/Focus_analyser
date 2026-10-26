"""
Personalized Baseline Calibrator Module.
Performs an initial calibration phase (e.g. 10–30s) to calculate customized
individual thresholds for EAR, gaze centering, and resting head pose.
"""

import numpy as np
from config import (
    CALIBRATION_DURATION,
    EAR_THRESHOLD,
    GAZE_CENTER_MIN,
    GAZE_CENTER_MAX,
    YAW_THRESHOLD,
    PITCH_THRESHOLD
)

class BaselineCalibrator:
    def __init__(self, duration_sec=CALIBRATION_DURATION):
        self.duration_sec = duration_sec
        self.is_calibrating = True
        self.start_time = None
        self.is_completed = False

        self.ear_samples = []
        self.gaze_samples = []
        self.yaw_samples = []
        self.pitch_samples = []

        # Personalized thresholds (fallback to defaults initially)
        self.thresholds = {
            "ear_threshold": EAR_THRESHOLD,
            "gaze_min": GAZE_CENTER_MIN,
            "gaze_max": GAZE_CENTER_MAX,
            "yaw_threshold": YAW_THRESHOLD,
            "pitch_threshold": PITCH_THRESHOLD
        }

    def update(self, current_time: float, avg_ear: float, gaze_ratio: float, yaw: float, pitch: float):
        """
        Records a sample during the calibration window.
        Returns:
            dict containing calibration progress (0-100%) and status message.
        """
        if self.is_completed:
            return {"progress": 100, "active": False, "msg": "Calibration Complete"}

        if self.start_time is None:
            self.start_time = current_time

        elapsed = current_time - self.start_time
        progress = min(100, int((elapsed / self.duration_sec) * 100))

        if avg_ear > 0.1:
            self.ear_samples.append(avg_ear)
        if 0.1 <= gaze_ratio <= 0.9:
            self.gaze_samples.append(gaze_ratio)
        self.yaw_samples.append(yaw)
        self.pitch_samples.append(pitch)

        if elapsed >= self.duration_sec:
            self._finalize()
            return {"progress": 100, "active": False, "msg": "Calibration Finished"}

        return {
            "progress": progress,
            "active": True,
            "msg": f"Calibrating personal baseline ({int(self.duration_sec - elapsed)}s left)..."
        }

    def _finalize(self):
        """Computes customized thresholds from the accumulated samples."""
        self.is_calibrating = False
        self.is_completed = True

        if len(self.ear_samples) >= 15:
            mean_ear = float(np.mean(self.ear_samples))
            std_ear = float(np.std(self.ear_samples))
            # Individual EAR threshold: typically 75% of resting open eye
            self.thresholds["ear_threshold"] = round(max(0.18, min(0.30, mean_ear * 0.72)), 3)

        if len(self.gaze_samples) >= 15:
            mean_gaze = float(np.mean(self.gaze_samples))
            self.thresholds["gaze_min"] = round(max(0.25, mean_gaze - 0.18), 2)
            self.thresholds["gaze_max"] = round(min(0.75, mean_gaze + 0.18), 2)

        if len(self.yaw_samples) >= 15:
            mean_yaw = float(np.mean(self.yaw_samples))
            # Adjust center yaw offset
            self.thresholds["yaw_center_offset"] = round(mean_yaw, 1)

    def get_thresholds(self):
        """Returns the personalized or default thresholds."""
        return self.thresholds
