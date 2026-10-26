# ============================================================
# FocusAnalyser — Configuration
# All thresholds, weights, and settings in one place.
# ============================================================

import os

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard", "saved_reports")
DB_PATH = os.path.join(DATA_DIR, "sessions.db")

# Locate landmark model in models/ or root
LANDMARK_MODEL_PATH = os.path.join(MODELS_DIR, "shape_predictor_68_face_landmarks.dat")
if not os.path.exists(LANDMARK_MODEL_PATH):
    LANDMARK_MODEL_PATH = os.path.join(BASE_DIR, "shape_predictor_68_face_landmarks.dat")

# --- Camera ---
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# --- Eye Analysis ---
EAR_THRESHOLD = 0.22            # Below this → eyes closed
BLINK_CONSEC_FRAMES = 2         # Min consecutive frames for a blink
EAR_OPEN_SCORE = 1.0            # Score when eyes are open
EAR_CLOSED_SCORE = 0.0          # Score when eyes are closed

# --- Gaze Analysis ---
GAZE_CENTER_MIN = 0.35          # Gaze ratio range for "center"
GAZE_CENTER_MAX = 0.65
GAZE_CENTER_SCORE = 1.0         # Score for looking at screen
GAZE_SIDE_SCORE = 0.3           # Score for looking away

# --- Head Pose ---
YAW_THRESHOLD = 22.0            # Degrees beyond which head is "turned"
PITCH_THRESHOLD = 18.0          # Degrees beyond which head is "tilted"
HEAD_FORWARD_SCORE = 1.0
HEAD_TURNED_SCORE = 0.2

# --- Drowsiness ---
PERCLOS_WINDOW = 90             # Frames to calculate PERCLOS over (~3 sec at 30fps)
PERCLOS_THRESHOLD = 0.35        # >35% eyes closed in window → drowsy
DROWSY_BLINK_RATE = 28          # Blinks/min above this → possible drowsiness
ALERT_SCORE = 1.0
DROWSY_SCORE = 0.1

# --- Yawn Detection ---
MAR_THRESHOLD = 0.55            # Mouth Aspect Ratio above this → yawning
YAWN_CONSEC_FRAMES = 12         # Frames mouth must be open to count as yawn

# --- Posture ---
POSTURE_GOOD_ANGLE_MAX = 14.0   # Degrees of shoulder/head tilt for "good" posture
POSTURE_GOOD_SCORE = 1.0
POSTURE_BAD_SCORE = 0.35

# --- Distraction Detection ---
DISTRACTION_OBJECTS = ["cell phone", "phone", "book", "bottle", "cup"]
DISTRACTION_CONFIDENCE = 0.45   # Min detection confidence
DISTRACTION_PENALTY = 0.3       # Score multiplier when distraction detected

# --- Attention Scoring Weights ---
# A = w1*Gaze + w2*Eye + w3*HeadPose + w4*Posture + w5*Alertness
WEIGHT_GAZE = 0.25
WEIGHT_EYE = 0.20
WEIGHT_HEAD_POSE = 0.25
WEIGHT_POSTURE = 0.15
WEIGHT_ALERTNESS = 0.15

# --- Focus Tracking ---
FOCUS_THRESHOLD = 0.60          # Attention score above this → "Focused"
DISTRACTED_THRESHOLD = 0.38     # Below this → "Distracted" (between = "Moderate")
STREAK_RESET_SECONDS = 2.5      # Seconds of low focus before streak resets

# --- Calibration ---
CALIBRATION_DURATION = 10       # Seconds for quick baseline calibration (default 10s)

# --- Temporal Analysis ---
SEGMENT_DURATION = 5.0          # Seconds per heatmap segment
TIMELINE_SMOOTHING = 5          # Rolling average window (frames)

# --- Visualization Colors (BGR) ---
COLOR_FOCUSED = (60, 220, 60)      # Bright Green
COLOR_MODERATE = (40, 200, 255)   # Amber/Yellow
COLOR_DISTRACTED = (50, 50, 240)  # Bright Red
COLOR_DROWSY = (220, 50, 220)     # Magenta/Purple
COLOR_TEXT = (255, 255, 255)      # White
COLOR_BG = (30, 30, 30)           # Dark gray
COLOR_ACCENT = (255, 170, 0)      # Vivid Blue

# --- Dlib 68-Landmark Indices ---
DLIB_LEFT_EYE = [36, 37, 38, 39, 40, 41]
DLIB_RIGHT_EYE = [42, 43, 44, 45, 46, 47]
DLIB_NOSE_TIP = 30
DLIB_CHIN = 8
DLIB_LEFT_EYE_CORNER = 36
DLIB_RIGHT_EYE_CORNER = 45
DLIB_LEFT_MOUTH = 48
DLIB_RIGHT_MOUTH = 54
DLIB_TOP_LIP = 51
DLIB_BOTTOM_LIP = 57
DLIB_TOP_INNER_LIP = 62
DLIB_BOTTOM_INNER_LIP = 66
