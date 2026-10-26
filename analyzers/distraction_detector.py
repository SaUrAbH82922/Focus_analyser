"""
Distraction Object Detection Module.
Detects potential distraction items (e.g. mobile phone, bottle, book)
using lightweight YOLOv8 when available, or provides a graceful no-op fallback.
"""

from config import (
    DISTRACTION_OBJECTS,
    DISTRACTION_CONFIDENCE,
    DISTRACTION_PENALTY
)

class DistractionDetector:
    def __init__(self, confidence=DISTRACTION_CONFIDENCE):
        self.confidence = confidence
        self.enabled = False
        self.model = None

        try:
            from ultralytics import YOLO
            # Try to load lightweight nano model
            self.model = YOLO("yolov8n.pt")
            self.enabled = True
        except Exception:
            self.model = None
            self.enabled = False

    def analyze(self, frame):
        """
        Runs object detection on frame to find target distraction objects.
        Returns:
            dict containing detected_objects, is_distracted flag, distraction_score.
        """
        if not self.enabled or self.model is None or frame is None:
            return {
                "detected_objects": [],
                "is_distracted": False,
                "distraction_score": 1.0,
                "enabled": False
            }

        try:
            results = self.model(frame, verbose=False, conf=self.confidence)
            detected = []

            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id].lower()
                    conf = float(box.conf[0])
                    
                    for target in DISTRACTION_OBJECTS:
                        if target in label:
                            detected.append((label, round(conf, 2)))

            is_distracted = len(detected) > 0
            score = DISTRACTION_PENALTY if is_distracted else 1.0

            return {
                "detected_objects": detected,
                "is_distracted": is_distracted,
                "distraction_score": score,
                "enabled": True
            }
        except Exception:
            return {
                "detected_objects": [],
                "is_distracted": False,
                "distraction_score": 1.0,
                "enabled": False
            }
