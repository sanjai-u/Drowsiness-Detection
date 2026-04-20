import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from scipy.spatial import distance as dist
import threading
import os
import time

# PARAMETERS
EAR_THRESHOLD = 0.23
CONSEC_FRAMES = 20
SMOOTHING_FRAMES = 5

counter = 0
alarm_on = False
ear_history = []

# ALARM FUNCTION (Mac)
def play_alarm():
    os.system("say Wake up Bro")

# EAR CALCULATION
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# LOAD FACE LANDMARKER MODEL
base_options = python.BaseOptions(model_asset_path="face_landmarker.task")
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)

# Eye landmark indexes (MediaPipe standard)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# START CAMERA

cap = cv2.VideoCapture(0)
prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = detector.detect(mp_image)

    if result.face_landmarks:
        face_landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        left_eye = []
        right_eye = []

        for idx in LEFT_EYE:
            x = int(face_landmarks[idx].x * w)
            y = int(face_landmarks[idx].y * h)
            left_eye.append((x, y))

        for idx in RIGHT_EYE:
            x = int(face_landmarks[idx].x * w)
            y = int(face_landmarks[idx].y * h)
            right_eye.append((x, y))

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0

        # SMOOTHING
        ear_history.append(ear)
        if len(ear_history) > SMOOTHING_FRAMES:
            ear_history.pop(0)

        smoothed_ear = sum(ear_history) / len(ear_history)

        # Draw eye points
        for point in left_eye + right_eye:
            cv2.circle(frame, point, 2, (0, 255, 0), -1)

        # Drowsiness logic
        if smoothed_ear < EAR_THRESHOLD:
            counter += 1
            if counter >= CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS ALERT!",
                            (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.5,
                            (0, 0, 255),
                            3)

                if not alarm_on:
                    alarm_on = True
                    t = threading.Thread(target=play_alarm)
                    t.daemon = True
                    t.start()
        else:
            counter = 0
            alarm_on = False

        cv2.putText(frame, f"EAR: {smoothed_ear:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2)

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    cv2.putText(frame, f"FPS: {int(fps)}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2)

    cv2.imshow("AI Drowsiness Detection - Sanjai", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()