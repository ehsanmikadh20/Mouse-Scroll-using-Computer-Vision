import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

cap = cv2.VideoCapture(0)

neutral = None
prev_signal = None
last_action_time = 0
gesture_start = None

SCROLL_AMOUNT = 30
COOLDOWN = 0.35

UP_THRESHOLD = 0.015     # harder
DOWN_THRESHOLD = -0.008  # easier (wrinkles are subtle)

HOLD_TIME = 0.10         # seconds
SMOOTH = 0.6

def avg_y(lm, idx):
    return np.mean([lm[i].y for i in idx])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if result.multi_face_landmarks:
        lm = result.multi_face_landmarks[0].landmark

        eyebrow_y = avg_y(lm, [65, 66, 67])
        eye_y = lm[159].y

        signal = eye_y - eyebrow_y  # normalized already (0–1)

        # Smoothing
        if prev_signal is None:
            prev_signal = signal
        signal = SMOOTH * prev_signal + (1 - SMOOTH) * signal
        prev_signal = signal

        # Neutral calibration
        if neutral is None:
            neutral = signal
            cv2.putText(frame, "Calibrating neutral...",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (255, 255, 0), 2)
        else:
            delta = signal - neutral
            now = time.time()

            # Track gesture hold
            if delta > UP_THRESHOLD or delta < DOWN_THRESHOLD:
                if gesture_start is None:
                    gesture_start = now
            else:
                gesture_start = None

            if gesture_start and now - gesture_start > HOLD_TIME:
                if now - last_action_time > COOLDOWN:

                    if delta > UP_THRESHOLD:
                        pyautogui.scroll(SCROLL_AMOUNT)
                        last_action_time = now
                        gesture_start = None
                        cv2.putText(frame, "UP SCROLL",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 255, 0), 2)

                    elif delta < DOWN_THRESHOLD:
                        pyautogui.scroll(-SCROLL_AMOUNT)
                        last_action_time = now
                        gesture_start = None
                        cv2.putText(frame, "DOWN SCROLL",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)

    cv2.imshow("Eyebrow Scroll Control (DOWN FIXED)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
