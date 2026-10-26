import cv2
import dlib
import time
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist



# Eye Aspect Ratio (EAR) calculation
def calculate_ear(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.25  # Threshold for eye  detection

def main():
    cap = cv2.VideoCapture(0)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    focus_start_time = None
    total_focus_time = 0
    focused = False

    timestamps = []
    attentiveness = []
    session_start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        current_time = time.time() - session_start_time
        timestamps.append(current_time)

        is_focused = False

        for face in faces:
            landmarks = predictor(gray, face)
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear > EAR_THRESHOLD: 
                is_focused = True

       
            for (x, y) in left_eye + right_eye:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        
        if is_focused:
            if not focused:
                focus_start_time = time.time()
                focused = True
            attentiveness.append(1)
        else:
            if focused:
                total_focus_time += time.time() - focus_start_time
                focused = False
            attentiveness.append(0)

        focus_display = total_focus_time
        if focused and focus_start_time is not None:
            focus_display += time.time() - focus_start_time
        cv2.putText(frame, f'Focus Time: {focus_display:.2f}s', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Focus Analyzer', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if focused and focus_start_time is not None:
        total_focus_time += time.time() - focus_start_time

    cap.release()
    cv2.destroyAllWindows()

    print(f'Total Focus Time: {total_focus_time:.2f} seconds')

    plt.figure(figsize=(12, 6), dpi=100)
    plt.plot(timestamps, attentiveness, color='dodgerblue', linewidth=2, linestyle='-', marker='o', markersize=4)
    plt.fill_between(timestamps, attentiveness, color='lightgreen', alpha=0.2)
    plt.xlabel('Time (s)', fontsize=14, fontweight='bold')
    plt.ylabel('Attentiveness', fontsize=14, fontweight='bold')
    plt.title('Focus Analysis Over Time', fontsize=16, fontweight='bold')
    plt.yticks([0, 1], ['Distracted', 'Focused'], fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    filename = f'dashboard/focus_report_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.png'
    plt.savefig(filename, dpi=150)
    print(f'Graph saved to: {filename}')
    plt.show()

if __name__ == "__main__":
    main()
