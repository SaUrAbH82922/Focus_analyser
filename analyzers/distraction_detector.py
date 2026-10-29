"""
Distraction Object Detection Module.
Detects potential distraction items (e.g. mobile phone, bottle, book)
using lightweight YOLOv8. Returns bounding boxes and detection confidence.
"""

import sys
from config import (
    DISTRACTION_OBJECTS,
    DISTRACTION_CONFIDENCE,
    DISTRACTION_PENALTY
)

class DistractionDetector:
    def __init__(self, confidence=0.35):
        self.confidence = confidence
        self.enabled = False
        self.model = None
        self.status_msg = ""

        try:
            from ultralytics import YOLO
            # Load lightweight nano model (downloads yolov8n.pt ~6MB on first run)
            self.model = YOLO("yolov8n.pt")
            self.enabled = True
            self.status_msg = "YOLOv8 nano model loaded successfully."
            print("[✓] DistractionDetector: YOLOv8 enabled (Phone & Object tracking active)")
        except ImportError:
            self.model = None
            self.enabled = False
            self.status_msg = "Package 'ultralytics' not installed. Run: pip install ultralytics"
            print("[!] DistractionDetector: 'ultralytics' not installed. Phone detection is DISABLED.")
            print("    To enable phone detection, run in your terminal: pip install ultralytics")
        except Exception as e:
            self.model = None
            self.enabled = False
            self.status_msg = f"Error loading YOLO: {e}"
            print(f"[!] DistractionDetector initialization error: {e}")

    def analyze(self, frame):
        """
        Runs object detection on frame to find target distraction objects.
        Returns:
            dict containing:
                detected_objects: list of dicts [{"label": str, "conf": float, "box": (x1, y1, x2, y2)}]
                is_distracted: bool
                distraction_score: float (0.3 if distraction found, 1.0 otherwise)
                enabled: bool
        """
        if not self.enabled or self.model is None or frame is None:
            return {
                "detected_objects": [],
                "is_distracted": False,
                "distraction_score": 1.0,
                "enabled": self.enabled,
                "status_msg": self.status_msg
            }

        try:
            # Run inference on frame
            results = self.model(frame, verbose=False, conf=self.confidence)
            detected = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names[cls_id].lower()
                    conf = float(box.conf[0])
                    
                    # Check if detected object is in distraction targets
                    is_match = False
                    for target in DISTRACTION_OBJECTS:
                        if target in label or label in target:
                            is_match = True
                            break

                    if is_match:
                        # Extract bounding box [x1, y1, x2, y2]
                        coords = box.xyxy[0].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                        detected.append({
                            "label": label,
                            "conf": round(conf, 2),
                            "box": (x1, y1, x2, y2)
                        })

            is_distracted = len(detected) > 0
            score = DISTRACTION_PENALTY if is_distracted else 1.0

            return {
                "detected_objects": detected,
                "is_distracted": is_distracted,
                "distraction_score": score,
                "enabled": True,
                "status_msg": "Active"
            }
        except Exception as e:
            return {
                "detected_objects": [],
                "is_distracted": False,
                "distraction_score": 1.0,
                "enabled": self.enabled,
                "status_msg": f"Inference error: {e}"
            }
