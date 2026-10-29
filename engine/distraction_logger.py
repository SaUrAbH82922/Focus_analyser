"""
Distraction Logger Module ("Why Did I Lose Focus?").
Maintains an explainable diagnostic log of specific behavioral triggers
(gaze shift, head turned, prolonged eye closure, yawning, slouching, face absent).
"""

from collections import Counter

class DistractionLogger:
    def __init__(self):
        self.events = []
        self.last_logged_time = -10.0
        self.cooldown_sec = 1.5  # Log every 1.5s when an active distraction occurs

    def log_event(self, timestamp: float, indicators: dict):
        """
        Records a distraction event.
        indicators: {
            "gaze_away": bool,
            "head_turned": bool,
            "eyes_closed": bool,
            "yawning": bool,
            "slouching": bool,
            "face_absent": bool
        }
        """
        if timestamp - self.last_logged_time < self.cooldown_sec:
            return

        active_reasons = []
        for reason, active in indicators.items():
            if active:
                clean = reason.replace("_", " ").title()
                active_reasons.append(clean)

        if not active_reasons:
            return

        self.events.append({
            "timestamp": round(timestamp, 1),
            "reasons": active_reasons
        })
        self.last_logged_time = timestamp

    def get_events(self):
        """Returns all recorded distraction events."""
        return list(self.events)

    def get_summary(self):
        """
        Generates aggregate breakdown of the most frequent behavioral distraction causes.
        """
        all_reasons = []
        for ev in self.events:
            all_reasons.extend(ev["reasons"])

        counts = Counter(all_reasons)
        total = len(self.events)
        most_common = counts.most_common(1)
        top_reason = most_common[0][0] if most_common else "None"

        return {
            "total_events": total,
            "reason_counts": dict(counts),
            "primary_distractor": top_reason
        }
