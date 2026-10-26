"""
Head Pose Estimation Module.
Estimates Yaw, Pitch, and Roll angles using OpenCV's solvePnP and 3D facial geometry.
"""

import cv2
import numpy as np
from config import (
    YAW_THRESHOLD,
    PITCH_THRESHOLD,
    HEAD_FORWARD_SCORE,
    HEAD_TURNED_SCORE,
    DLIB_NOSE_TIP,
    DLIB_CHIN,
    DLIB_LEFT_EYE_CORNER,
    DLIB_RIGHT_EYE_CORNER,
    DLIB_LEFT_MOUTH,
    DLIB_RIGHT_MOUTH
)

class HeadPoseAnalyzer:
    def __init__(self, yaw_thresh=YAW_THRESHOLD, pitch_thresh=PITCH_THRESHOLD):
        self.yaw_thresh = yaw_thresh
        self.pitch_thresh = pitch_thresh

        # Standard 3D anthropometric face model coordinates
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye corner
            (225.0, 170.0, -135.0),      # Right eye corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
        ], dtype=np.float64)

    def analyze(self, landmarks, frame_shape):
        """
        Estimates head pose Euler angles.
        Returns:
            dict containing yaw, pitch, roll, direction classification, and head score.
        """
        if landmarks is None or len(landmarks) < 68 or frame_shape is None:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "direction": "FORWARD",
                "head_score": HEAD_FORWARD_SCORE
            }

        h, w = frame_shape[:2]

        # 2D landmark coordinates on image plane
        image_points = np.array([
            landmarks[DLIB_NOSE_TIP],
            landmarks[DLIB_CHIN],
            landmarks[DLIB_LEFT_EYE_CORNER],
            landmarks[DLIB_RIGHT_EYE_CORNER],
            landmarks[DLIB_LEFT_MOUTH],
            landmarks[DLIB_RIGHT_MOUTH]
        ], dtype=np.float64)

        # Camera calibration matrix approximation
        focal_length = w
        center = (w / 2.0, h / 2.0)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        # Solve Perspective-n-Point
        success, rot_vec, trans_vec = cv2.solvePnP(
            self.model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            return {
                "yaw": 0.0,
                "pitch": 0.0,
                "roll": 0.0,
                "direction": "FORWARD",
                "head_score": HEAD_FORWARD_SCORE
            }

        # Convert rotation vector to rotation matrix
        rot_mat, _ = cv2.Rodrigues(rot_vec)

        # Extract Euler angles (pitch, yaw, roll)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        roll = float(angles[2])

        # Classify direction
        direction = "FORWARD"
        if yaw < -self.yaw_thresh:
            direction = "LOOKING_LEFT"
        elif yaw > self.yaw_thresh:
            direction = "LOOKING_RIGHT"
        elif pitch < -self.pitch_thresh:
            direction = "LOOKING_UP"
        elif pitch > self.pitch_thresh:
            direction = "LOOKING_DOWN"

        # Continuous head score computation
        # Distance from center orientation
        angle_dev = np.sqrt((yaw / self.yaw_thresh)**2 + (pitch / self.pitch_thresh)**2)
        if angle_dev <= 1.0:
            score = HEAD_FORWARD_SCORE - (0.3 * angle_dev)
        else:
            score = max(HEAD_TURNED_SCORE, HEAD_FORWARD_SCORE - (0.3 + 0.4 * (angle_dev - 1.0)))

        return {
            "yaw": round(yaw, 1),
            "pitch": round(pitch, 1),
            "roll": round(roll, 1),
            "direction": direction,
            "head_score": round(float(np.clip(score, 0.0, 1.0)), 2)
        }
