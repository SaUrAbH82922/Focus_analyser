"""
Live Overlay HUD Module.
Renders real-time computer vision telemetry, attention gauges, streak counters,
and biometric indicator status directly onto the webcam video feed.
"""

import cv2
import numpy as np
from config import (
    COLOR_FOCUSED,
    COLOR_MODERATE,
    COLOR_DISTRACTED,
    COLOR_DROWSY,
    COLOR_TEXT,
    COLOR_BG,
    COLOR_ACCENT
)

class LiveOverlay:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Formats seconds into MM:SS string."""
        mins = int(seconds) // 60
        secs = int(seconds) % 60
        return f"{mins:02d}:{secs:02d}"

    def draw(self, frame, data: dict):
        """
        Draws the comprehensive telemetry dashboard over the video frame.
        data dict requires:
            attention_score, state, attention_pct,
            current_streak, longest_streak,
            focused_time, distracted_time, drowsy_time, distraction_count,
            gaze_direction, eye_state, avg_ear,
            head_direction, yaw, pitch,
            posture_label,
            is_calibrating, calibration_progress,
            landmarks (optional)
        """
        if frame is None:
            return frame

        h, w = frame.shape[:2]
        state = data.get("state", "FOCUSED")

        # Pick accent color based on state
        if state == "FOCUSED":
            state_color = COLOR_FOCUSED
        elif state == "MODERATE":
            state_color = COLOR_MODERATE
        elif state == "DROWSY":
            state_color = COLOR_DROWSY
        else:
            state_color = COLOR_DISTRACTED

        # 1. Subtle colored perimeter border
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), state_color, 3)

        # 2. Draw eye landmarks if provided
        landmarks = data.get("landmarks")
        if landmarks:
            for (x, y) in landmarks[36:48]:  # Eyes
                cv2.circle(frame, (int(x), int(y)), 2, state_color, -1)
            # Nose tip
            cv2.circle(frame, (int(landmarks[30][0]), int(landmarks[30][1])), 3, COLOR_ACCENT, -1)

        # 3. Top-Left: Attention Gauge & State Badge
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (250, 85), COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (10, 10), (250, 85), state_color, 1)

        score_pct = data.get("attention_pct", 0)
        cv2.putText(frame, f"ATTENTION: {score_pct}%", (20, 36), self.font, 0.65, COLOR_TEXT, 2)
        cv2.putText(frame, f"STATE: {state}", (20, 58), self.font, 0.55, state_color, 2)

        # Attention Progress Bar
        bar_x, bar_y, bar_w, bar_h = 20, 66, 210, 10
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), -1)
        fill_w = int((score_pct / 100.0) * bar_w)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), state_color, -1)

        # 4. Top-Right: Focus Streak & Best Streak
        overlay_r = frame.copy()
        cv2.rectangle(overlay_r, (w - 240, 10), (w - 10, 85), COLOR_BG, -1)
        cv2.addWeighted(overlay_r, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (w - 240, 10), (w - 10, 85), COLOR_ACCENT, 1)

        streak_str = self._format_time(data.get("current_streak", 0.0))
        best_str = self._format_time(data.get("longest_streak", 0.0))

        cv2.putText(frame, f"FOCUS STREAK", (w - 225, 33), self.font, 0.55, COLOR_ACCENT, 2)
        cv2.putText(frame, f"{streak_str}", (w - 225, 58), self.font, 0.75, COLOR_TEXT, 2)
        cv2.putText(frame, f"Best: {best_str}", (w - 225, 78), self.font, 0.42, (180, 180, 180), 1)

        # 5. Bottom Telemetry Bar (Multi-row stats)
        bar_height = 80
        overlay_b = frame.copy()
        cv2.rectangle(overlay_b, (0, h - bar_height), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_b, 0.82, frame, 0.18, 0, frame)
        cv2.line(frame, (0, h - bar_height), (w, h - bar_height), (80, 80, 80), 1)

        # Row 1: Session Durations
        f_time = self._format_time(data.get("focused_time", 0.0))
        d_time = self._format_time(data.get("distracted_time", 0.0))
        dr_time = self._format_time(data.get("drowsy_time", 0.0))
        dist_count = data.get("distraction_count", 0)

        row1_text = f"Focus: {f_time}  |  Distracted: {d_time}  |  Drowsy: {dr_time}  |  Distractions: {dist_count}"
        cv2.putText(frame, row1_text, (15, h - 50), self.font, 0.48, (220, 220, 220), 1)

        # Row 2: Biometric Indicators
        gaze = data.get("gaze_direction", "CENTER")
        eye_st = data.get("eye_state", "OPEN")
        ear_val = data.get("avg_ear", 0.0)
        head = data.get("head_direction", "FORWARD")
        posture = data.get("posture_label", "Good")

        row2_text = f"Gaze: {gaze}  |  Eyes: {eye_st} ({ear_val:.2f})  |  Head: {head}  |  Posture: {posture}"
        cv2.putText(frame, row2_text, (15, h - 22), self.font, 0.44, (160, 210, 255), 1)

        # 6. Calibration Banner if active
        if data.get("is_calibrating", False):
            prog = data.get("calibration_progress", 0)
            cal_overlay = frame.copy()
            cv2.rectangle(cal_overlay, (int(w * 0.2), int(h * 0.4)), (int(w * 0.8), int(h * 0.6)), (30, 30, 30), -1)
            cv2.addWeighted(cal_overlay, 0.85, frame, 0.15, 0, frame)
            cv2.rectangle(frame, (int(w * 0.2), int(h * 0.4)), (int(w * 0.8), int(h * 0.6)), COLOR_ACCENT, 2)
            cv2.putText(frame, f"CALIBRATING BASELINE: {prog}%", (int(w * 0.25), int(h * 0.48)), self.font, 0.65, COLOR_TEXT, 2)
            cv2.putText(frame, "Please look naturally at the center of the screen", (int(w * 0.22), int(h * 0.54)), self.font, 0.45, (200, 200, 200), 1)

        return frame
