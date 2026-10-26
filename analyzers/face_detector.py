"""
Face and Facial Landmark Detector Module.
Uses dlib with 68 facial landmarks model (with optional MediaPipe fallback).
"""

import os
import cv2
import numpy as np
import dlib
from config import LANDMARK_MODEL_PATH, DLIB_LEFT_EYE, DLIB_RIGHT_EYE

class FaceDetector:
    def __init__(self, model_path=None):
        self.model_path = model_path or LANDMARK_MODEL_PATH
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Landmark model not found at '{self.model_path}'. "
                "Ensure shape_predictor_68_face_landmarks.dat exists."
            )
        
        # Initialize dlib frontal face detector & predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.model_path)
        
        # Optional MediaPipe Pose for full body posture if available
        self.pose_detector = None
        try:
            import mediapipe as mp
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            self.pose_detector = None

    def detect(self, frame):
        """
        Detects the primary face and returns (landmarks_list, bounding_box).
        landmarks_list: [(x, y), ...] for 68 points.
        bounding_box: (x, y, w, h)
        """
        if frame is None:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)
        
        if len(faces) == 0:
            return None, None

        # Pick the largest face (closest to camera)
        primary_face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = self.predictor(gray, primary_face)
        
        landmarks = [(shape.part(i).x, shape.part(i).y) for i in range(68)]
        x = primary_face.left()
        y = primary_face.top()
        w = primary_face.width()
        h = primary_face.height()
        
        return landmarks, (x, y, w, h)

    def detect_pose(self, frame):
        """
        Detects body pose landmarks using MediaPipe if available.
        Returns list of (x, y) in pixel coordinates or None.
        """
        if self.pose_detector is None or frame is None:
            return None
        
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose_detector.process(rgb)
            if not results.pose_landmarks:
                return None
            
            h, w = frame.shape[:2]
            pose_points = []
            for lm in results.pose_landmarks.landmark:
                pose_points.append((int(lm.x * w), int(lm.y * h)))
            return pose_points
        except Exception:
            return None
