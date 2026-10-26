"""
Focus Tracker Module.
Tracks continuous attention spans, live focus streaks, distraction counts,
and cumulative focused/distracted/drowsy time.
"""

from config import STREAK_RESET_SECONDS

class FocusTracker:
    def __init__(self, streak_reset_seconds=STREAK_RESET_SECONDS):
        self.streak_reset_seconds = streak_reset_seconds
        
        self.total_focused_time = 0.0
        self.total_distracted_time = 0.0
        self.total_drowsy_time = 0.0
        
        self.current_streak = 0.0
        self.longest_streak = 0.0
        
        self.current_span_duration = 0.0
        self.completed_spans = []
        
        self.distraction_count = 0
        self.low_focus_timer = 0.0
        
        self.last_state = "DISTRACTED"
        self.last_timestamp = None

    def update(self, state: str, attention_score: float, current_time: float):
        """
        Updates tracking metrics with delta time since last call.
        Returns:
            dict containing comprehensive session tracking metrics.
        """
        if self.last_timestamp is None:
            self.last_timestamp = current_time
            return self.get_stats()

        dt = current_time - self.last_timestamp
        self.last_timestamp = current_time

        if dt <= 0:
            return self.get_stats()

        # Cumulative time accumulation
        if state == "FOCUSED":
            self.total_focused_time += dt
            self.current_streak += dt
            self.current_span_duration += dt
            self.low_focus_timer = 0.0

            if self.current_streak > self.longest_streak:
                self.longest_streak = self.current_streak

        elif state == "DROWSY":
            self.total_drowsy_time += dt
            self.low_focus_timer += dt
        else:  # "DISTRACTED" or "ABSENT" or "MODERATE" (if sub-focus)
            self.total_distracted_time += dt
            self.low_focus_timer += dt

        # Handle distraction transition & streak reset
        if state in ["DISTRACTED", "ABSENT", "DROWSY"]:
            if self.last_state == "FOCUSED":
                self.distraction_count += 1
                if self.current_span_duration > 1.0:
                    self.completed_spans.append(round(self.current_span_duration, 1))
                self.current_span_duration = 0.0

            if self.low_focus_timer >= self.streak_reset_seconds:
                self.current_streak = 0.0

        self.last_state = state
        return self.get_stats()

    def get_stats(self):
        """Returns the current summary statistics."""
        all_spans = list(self.completed_spans)
        if self.current_span_duration > 1.0:
            all_spans.append(round(self.current_span_duration, 1))

        avg_span = (sum(all_spans) / len(all_spans)) if all_spans else 0.0

        return {
            "current_streak": round(self.current_streak, 1),
            "longest_streak": round(self.longest_streak, 1),
            "total_focused": round(self.total_focused_time, 1),
            "total_distracted": round(self.total_distracted_time, 1),
            "total_drowsy": round(self.total_drowsy_time, 1),
            "distraction_count": self.distraction_count,
            "avg_focus_span": round(avg_span, 1),
            "completed_spans": all_spans
        }
