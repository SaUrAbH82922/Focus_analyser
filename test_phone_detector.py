"""
Standalone Phone & Distraction Detection Diagnostic Script.
Use this script to verify webcam object and phone detection.
"""

import sys
import cv2

print("=" * 70)
print("  FocusAnalyser — Phone & Object Detection Diagnostics")
print("=" * 70)

try:
    from ultralytics import YOLO
    print("[✓] Package 'ultralytics' is installed.")
except ImportError:
    print("[!] ERROR: 'ultralytics' is NOT installed in your Python environment!")
    print("\nTo fix this and enable phone detection, open your terminal and run:")
    print("      pip install ultralytics\n")
    print("=" * 70)
    sys.exit(1)

print("[*] Loading YOLOv8 nano model (yolov8n.pt)...")
model = YOLO("yolov8n.pt")
print("[✓] Model loaded successfully.")
print("[*] Opening webcam... Hold your phone, bottle, cup, or book in front of the camera.")
print("[*] Press 'q' to quit.")
print("=" * 70)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[!] Could not open webcam.")
    sys.exit(1)

TARGETS = ["cell phone", "phone", "bottle", "cup", "book"]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference at 0.35 confidence
    results = model(frame, verbose=False, conf=0.35)

    phone_found = False

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id].lower()
            conf = float(box.conf[0])

            # Check if target
            is_target = any(t in name for t in TARGETS)
            color = (0, 0, 255) if "phone" in name else (0, 255, 255) if is_target else (128, 128, 128)

            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label_text = f"{name.upper()} {int(conf * 100)}%"
            cv2.putText(frame, label_text, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if "phone" in name:
                phone_found = True

    if phone_found:
        cv2.rectangle(frame, (10, 10), (320, 50), (0, 0, 200), -1)
        cv2.putText(frame, "PHONE DETECTED!", (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Phone Detection Diagnostic Test (Press 'q' to exit)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("[*] Diagnostic test complete.")

