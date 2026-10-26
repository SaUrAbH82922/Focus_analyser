"""
=============================================================================
FocusAnalyser: AI-Based Attention Span & Productivity Monitoring System
=============================================================================
Main application entry point. Orchestrates computer-vision modules,
temporal tracking engine, real-time HUD, SQLite persistence, and post-session reporting.
"""

import sys
import os
import time
import argparse
from datetime import datetime
import cv2

# Project Modules
from config import (
    CAMERA_INDEX,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    CALIBRATION_DURATION
)
from analyzers import (
    FaceDetector,
    EyeAnalyzer,
    GazeAnalyzer,
    HeadPoseAnalyzer,
    YawnAnalyzer,
    DrowsinessAnalyzer,
    PostureAnalyzer,
    DistractionDetector
)
from engine import (
    AttentionScorer,
    FocusTracker,
    TemporalAnalyzer,
    BaselineCalibrator,
    DistractionLogger
)
from database import SessionDB
from dashboard import LiveOverlay, ReportGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="FocusAnalyser — Real-time Attention & Productivity Monitoring")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX, help="Webcam device index (default: 0)")
    parser.add_argument("--calibrate", action="store_true", default=True, help="Enable initial personal baseline calibration")
    parser.add_argument("--no-calibrate", dest="calibrate", action="store_false", help="Skip baseline calibration")
    parser.add_argument("--label", type=str, default="Standard", help="Experiment or session label (e.g. 'Study-Normal', 'Phone-Nearby')")
    parser.add_argument("--no-gui-chart", action="store_true", help="Save report chart without popping up matplotlib window")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("  FocusAnalyser: AI-Based Attention & Productivity Monitoring System")
    print("=" * 70)
    print(f"[*] Starting session with label: '{args.label}'")
    print("[*] Initializing computer vision models...")

    # 1. Initialize Analyzers
    try:
        face_detector = FaceDetector()
    except Exception as e:
        print(f"[!] Error initializing FaceDetector: {e}")
        sys.exit(1)

    eye_analyzer = EyeAnalyzer()
    gaze_analyzer = GazeAnalyzer()
    head_pose_analyzer = HeadPoseAnalyzer()
    yawn_analyzer = YawnAnalyzer()
    drowsiness_analyzer = DrowsinessAnalyzer()
    posture_analyzer = PostureAnalyzer()
    distraction_detector = DistractionDetector()

    # 2. Initialize Engines & Storage
    attention_scorer = AttentionScorer()
    focus_tracker = FocusTracker()
    temporal_analyzer = TemporalAnalyzer()
    baseline_calibrator = BaselineCalibrator(duration_sec=CALIBRATION_DURATION if args.calibrate else 0)
    distraction_logger = DistractionLogger()

    db = SessionDB()
    overlay = LiveOverlay()
    report_gen = ReportGenerator()

    # 3. Open Camera Feed
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
    if not cap.isOpened():
        # Fallback without CAP_DSHOW
        cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        print(f"[!] Error: Could not open camera {args.camera}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    session_start_time = time.time()
    session_start_str = datetime.now().strftime("%H:%M:%S")

    print("[*] Camera started successfully.")
    print("=" * 70)
    print("  CONTROLS:")
    print("    [q] - Finish session & generate analytical report")
    print("    [c] - Trigger baseline recalibration")
    print("=" * 70)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[!] Failed to capture webcam frame.")
                break

            current_timestamp = time.time() - session_start_time

            # -------------------------------------------------------------
            # Stage A: Facial & Body Detection
            # -------------------------------------------------------------
            landmarks, face_bbox = face_detector.detect(frame)
            pose_landmarks = face_detector.detect_pose(frame)
            face_present = (landmarks is not None)

            # -------------------------------------------------------------
            # Stage B: Biometric Analyzers
            # -------------------------------------------------------------
            eye_res = eye_analyzer.analyze(landmarks)
            gaze_res = gaze_analyzer.analyze(landmarks, frame)
            head_res = head_pose_analyzer.analyze(landmarks, frame.shape)
            yawn_res = yawn_analyzer.analyze(landmarks)
            
            drowsy_res = drowsiness_analyzer.analyze(
                eyes_open=eye_res["eyes_open"],
                is_yawning=yawn_res["is_yawning"],
                blink_count=eye_res["blink_count"],
                elapsed_time_sec=current_timestamp
            )

            posture_res = posture_analyzer.analyze(
                pose_landmarks=pose_landmarks,
                head_pose_dict=head_res,
                face_bbox=face_bbox,
                frame_shape=frame.shape
            )

            distract_res = distraction_detector.analyze(frame)

            # -------------------------------------------------------------
            # Stage C: Baseline Calibration
            # -------------------------------------------------------------
            cal_info = {"progress": 100, "active": False}
            if args.calibrate and not baseline_calibrator.is_completed:
                cal_info = baseline_calibrator.update(
                    current_time=current_timestamp,
                    avg_ear=eye_res["avg_ear"],
                    gaze_ratio=gaze_res["avg_ratio"],
                    yaw=head_res["yaw"],
                    pitch=head_res["pitch"]
                )

            # -------------------------------------------------------------
            # Stage D: Attention Scoring & State Classification
            # -------------------------------------------------------------
            score_data = attention_scorer.calculate(
                gaze_score=gaze_res["gaze_score"],
                eye_score=eye_res["eye_score"],
                head_score=head_res["head_score"],
                posture_score=posture_res["posture_score"],
                alertness_score=drowsy_res["alertness_score"],
                distraction_score=distract_res["distraction_score"],
                is_drowsy=drowsy_res["is_drowsy"],
                face_present=face_present
            )

            # -------------------------------------------------------------
            # Stage E: Temporal Tracking & Diagnostics
            # -------------------------------------------------------------
            tracker_stats = focus_tracker.update(
                state=score_data["state"],
                attention_score=score_data["attention_score"],
                current_time=current_timestamp
            )

            temporal_analyzer.update(
                timestamp=current_timestamp,
                score=score_data["attention_score"],
                state=score_data["state"]
            )

            # If user drops focus, log contributing diagnostic reasons
            if score_data["state"] in ["DISTRACTED", "DROWSY", "ABSENT"]:
                distraction_logger.log_event(
                    timestamp=current_timestamp,
                    indicators={
                        "gaze_away": gaze_res["direction"] != "CENTER",
                        "head_turned": head_res["direction"] != "FORWARD",
                        "eyes_closed": not eye_res["eyes_open"],
                        "yawning": yawn_res["is_yawning"],
                        "slouching": "Slouching" in posture_res["posture_label"],
                        "object_detected": distract_res["is_distracted"],
                        "face_absent": not face_present
                    }
                )

            # -------------------------------------------------------------
            # Stage F: HUD Rendering
            # -------------------------------------------------------------
            hud_payload = {
                "attention_score": score_data["attention_score"],
                "attention_pct": score_data["attention_pct"],
                "state": score_data["state"],
                "current_streak": tracker_stats["current_streak"],
                "longest_streak": tracker_stats["longest_streak"],
                "focused_time": tracker_stats["total_focused"],
                "distracted_time": tracker_stats["total_distracted"],
                "drowsy_time": tracker_stats["total_drowsy"],
                "distraction_count": tracker_stats["distraction_count"],
                "gaze_direction": gaze_res["direction"],
                "eye_state": eye_res["state"],
                "avg_ear": eye_res["avg_ear"],
                "head_direction": head_res["direction"],
                "yaw": head_res["yaw"],
                "pitch": head_res["pitch"],
                "posture_label": posture_res["posture_label"],
                "is_calibrating": cal_info["active"],
                "calibration_progress": cal_info["progress"],
                "landmarks": landmarks
            }

            frame = overlay.draw(frame, hud_payload)
            cv2.imshow("FocusAnalyser — Real-Time Monitor", frame)

            # -------------------------------------------------------------
            # Stage G: Keyboard Listeners
            # -------------------------------------------------------------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                print("[*] Resetting baseline calibration...")
                baseline_calibrator = BaselineCalibrator(duration_sec=CALIBRATION_DURATION)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    total_session_time = max(0.1, time.time() - session_start_time)
    final_stats = focus_tracker.get_stats()
    distraction_summary = distraction_logger.get_summary()

    # Calculate overall session attention score (%)
    timeline_timestamps, timeline_scores = temporal_analyzer.get_timeline()
    overall_attention_pct = int(round(float(sum(timeline_scores) / max(1, len(timeline_scores))) * 100)) if timeline_scores else 0

    session_record = {
        "start_time": session_start_str,
        "duration": round(total_session_time, 1),
        "attention_score": overall_attention_pct,
        "focused_time": final_stats["total_focused"],
        "distracted_time": final_stats["total_distracted"],
        "drowsy_time": final_stats["total_drowsy"],
        "avg_focus_span": final_stats["avg_focus_span"],
        "longest_focus_span": final_stats["longest_streak"],
        "distraction_count": final_stats["distraction_count"],
        "blink_count": eye_analyzer.total_blinks,
        "yawn_count": yawn_analyzer.total_yawns,
        "primary_distractor": distraction_summary["primary_distractor"],
        "experiment_label": args.label
    }

    # 4. Save to Database
    db_id = db.save_session(session_record)
    print("\n" + "=" * 70)
    print(f"  SESSION COMPLETE  [DB ID: #{db_id}]")
    print("=" * 70)
    print(f"  • Overall Attention Score: {overall_attention_pct}%")
    print(f"  • Total Duration:          {total_session_time:.1f} s")
    print(f"  • Total Focused Time:      {final_stats['total_focused']:.1f} s")
    print(f"  • Total Distracted Time:   {final_stats['total_distracted']:.1f} s")
    print(f"  • Average Focus Span:      {final_stats['avg_focus_span']:.1f} s")
    print(f"  • Longest Focus Streak:    {final_stats['longest_streak']:.1f} s")
    print(f"  • Distraction Incidents:   {final_stats['distraction_count']}")
    print(f"  • Primary Distractor:      {distraction_summary['primary_distractor']}")
    print("=" * 70)

    # 5. Generate and Display Analytical Report Chart
    print("[*] Generating post-session analytical visualization...")
    heatmap_data = temporal_analyzer.get_heatmap()
    report_gen.generate(
        timeline_data=(timeline_timestamps, timeline_scores),
        heatmap_data=heatmap_data,
        stats=session_record,
        distraction_summary=distraction_summary,
        show_window=(not args.no_gui_chart)
    )


if __name__ == "__main__":
    main()
