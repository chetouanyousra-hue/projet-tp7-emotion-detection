import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import time
import os
from gaze_tracker_utils import compute_actual_eye_gaze
from gaze_logger import init_csv, append_gaze_data
from gaze_visualizer import draw_gaze_bar

# Load model and labels
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', padding='valid', input_shape=(48,48,1)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu', padding='valid'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu', padding='valid'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])
    return model

model = build_model()
model.load_weights('fer2013_emotion_model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
IMG_SIZE = 48

# Gaze thresholds
FIXATION_THRESHOLD = 0.05
LEFT_THRESHOLD     = -0.2
RIGHT_THRESHOLD    =  0.2

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
mp_holistic  = mp.solutions.holistic
mp_drawing   = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                   max_num_faces=2,
                                   refine_landmarks=True,
                                   min_detection_confidence=0.6,
                                   min_tracking_confidence=0.6)

holistic     = mp_holistic.Holistic(min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

# Prepare logging
csv_file = 'gaze_data.csv'
init_csv(csv_file)

# Start video
cap          = cv2.VideoCapture(0)
prev_time    = 0
frame_count  = 0
last_side    = None
switch_count = 0

# Gaze direction counters
gaze_counts = {'Left': 0, 'Center': 0, 'Right': 0}

# Display toggles
show_face       = True
show_eyes_mouth = False
show_pose       = True
show_hands      = True
show_rectangle  = True
show_gaze       = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame            = cv2.flip(frame, 1)
    rgb_frame        = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    holistic_results = holistic.process(rgb_frame)
    face_results     = face_mesh.process(rgb_frame)
    h, w, _          = frame.shape

    if show_pose and holistic_results.pose_landmarks:
        mp_drawing.draw_landmarks(frame,
                                  holistic_results.pose_landmarks,
                                  mp_holistic.POSE_CONNECTIONS)

    if show_hands and holistic_results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame,
                                  holistic_results.left_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)
    if show_hands and holistic_results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame,
                                  holistic_results.right_hand_landmarks,
                                  mp_holistic.HAND_CONNECTIONS)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            if show_face:
                mp_drawing.draw_landmarks(frame,
                                          face_landmarks,
                                          mp_face_mesh.FACEMESH_TESSELATION,
                                          drawing_spec,
                                          drawing_spec)
            elif show_eyes_mouth:
                for idx in list(range(33, 133)) + list(range(263, 294)) + list(range(78, 88)):
                    x = int(face_landmarks.landmark[idx].x * w)
                    y = int(face_landmarks.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 255), -1)

            # Emotion prediction
            x_coords = [lm.x for lm in face_landmarks.landmark]
            y_coords = [lm.y for lm in face_landmarks.landmark]
            x_min    = int(min(x_coords) * w)
            y_min    = int(min(y_coords) * h)
            x_max    = int(max(x_coords) * w)
            y_max    = int(max(y_coords) * h)
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            face_roi = frame[y_min:y_max, x_min:x_max]
            if face_roi.size > 0:
                gray_face       = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                resized_face    = cv2.resize(gray_face, (IMG_SIZE, IMG_SIZE),
                                             interpolation=cv2.INTER_AREA)
                normalized_face = resized_face / 255.0
                face_input      = normalized_face.reshape(1, IMG_SIZE, IMG_SIZE, 1)
                preds           = model.predict(face_input, verbose=0)
                emotion_idx     = np.argmax(preds)
                emotion_text    = emotion_labels[emotion_idx]
                if show_rectangle:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                  (255, 0, 0), 2)
                    cv2.putText(frame, emotion_text, (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Gaze estimation and logging
            if show_gaze:
                eye_center, gaze_vec = compute_actual_eye_gaze(
                    face_landmarks.landmark, w, h)
                eye_tip = tuple(np.int32(eye_center + gaze_vec * 80))
                cv2.arrowedLine(frame,
                                tuple(np.int32(eye_center)),
                                eye_tip,
                                (0, 255, 0), 2)

                gaze_x   = gaze_vec[0]
                fixation = abs(gaze_x) < FIXATION_THRESHOLD
                side     = 'Center'
                if gaze_x < LEFT_THRESHOLD:
                    side = 'Left'
                elif gaze_x > RIGHT_THRESHOLD:
                    side = 'Right'

                gaze_counts[side] += 1

                if last_side and side != last_side:
                    switch_count += 1
                last_side = side

                append_gaze_data(csv_file, frame_count, gaze_vec, fixation, side)

    # Overlay gaze stats
    draw_gaze_bar(frame, gaze_counts)
    cv2.putText(frame, f'Switches: {switch_count}', (20, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

    # FPS overlay
    curr_time   = time.time()
    fps         = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time   = curr_time
    frame_count += 1
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Autism-related Emotion and Behavior Detection', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        show_face       = True
        show_eyes_mouth = False
    elif key == ord('m'):
        show_face       = False
        show_eyes_mouth = True
    elif key == ord('p'):
        show_pose = True
    elif key == ord('h'):
        show_hands = True
    elif key == ord('c'):
        show_face       = False
        show_eyes_mouth = False
        show_pose       = False
        show_hands      = False
        show_rectangle  = False
        show_gaze       = False
    elif key == ord('r'):
        show_rectangle = True
    elif key == ord('g'):
        show_gaze = True

cap.release()
cv2.destroyAllWindows()