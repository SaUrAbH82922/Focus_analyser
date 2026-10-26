"""
Temporal Analyzer Module.
Maintains continuous time-series of attention scores, generates segment-based heatmaps,
and provides moving-average smoothed curves for post-session visualization.
"""

from collections import deque
import numpy as np
from config import SEGMENT_DURATION, TIMELINE_SMOOTHING

class TemporalAnalyzer:
    def __init__(self, segment_duration=SEGMENT_DURATION, smoothing_window=TIMELINE_SMOOTHING):
        self.segment_duration = segment_duration
        self.smoothing_window = smoothing_window

        self.timestamps = []
        self.scores = []
        self.states = []

        # Current ongoing segment accumulator
        self.current_segment_scores = []
        self.current_segment_start = 0.0
        self.heatmap_segments = []

    def update(self, timestamp: float, score: float, state: str):
        """Records a new frame's temporal observation."""
        self.timestamps.append(timestamp)
        self.scores.append(score)
        self.states.append(state)

        self.current_segment_scores.append(score)

        # Check if a segment has completed
        if timestamp - self.current_segment_start >= self.segment_duration:
            avg_seg_score = float(np.mean(self.current_segment_scores))
            
            if avg_seg_score >= 0.70:
                color_label = "HIGH"
            elif avg_seg_score >= 0.40:
                color_label = "MODERATE"
            else:
                color_label = "LOW"

            self.heatmap_segments.append({
                "start": round(self.current_segment_start, 1),
                "end": round(timestamp, 1),
                "avg_score": round(avg_seg_score, 2),
                "label": color_label
            })

            self.current_segment_start = timestamp
            self.current_segment_scores = []

    def get_timeline(self):
        """Returns raw timestamps and attention scores."""
        return list(self.timestamps), list(self.scores)

    def get_smoothed_timeline(self):
        """Computes a moving-average smoothed attention curve."""
        if len(self.scores) < self.smoothing_window:
            return list(self.timestamps), list(self.scores)

        weights = np.ones(self.smoothing_window) / self.smoothing_window
        smoothed = np.convolve(self.scores, weights, mode='valid')
        trimmed_timestamps = self.timestamps[self.smoothing_window - 1:]

        return list(trimmed_timestamps), [round(float(s), 3) for s in smoothed]

    def get_heatmap(self):
        """Returns the list of discretized heatmap segments."""
        segments = list(self.heatmap_segments)
        # Include partial ongoing segment if session is ending
        if self.current_segment_scores:
            avg = float(np.mean(self.current_segment_scores))
            segments.append({
                "start": round(self.current_segment_start, 1),
                "end": round(self.timestamps[-1], 1) if self.timestamps else self.current_segment_start,
                "avg_score": round(avg, 2),
                "label": "HIGH" if avg >= 0.7 else ("MODERATE" if avg >= 0.4 else "LOW")
            })
        return segments
