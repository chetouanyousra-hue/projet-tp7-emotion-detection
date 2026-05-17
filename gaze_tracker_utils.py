# gaze_tracker_utils.py
import numpy as np

def compute_actual_eye_gaze(landmarks, frame_width, frame_height):

    # 1. Position centrale de l'iris gauche (MediaPipe landmark 468)
    left_iris_center = np.array([landmarks[468].x * frame_width,
                                  landmarks[468].y * frame_height])

    # 2. Points clés de l'oeil gauche (coin externe et interne)
    left_eye_outer = np.array([landmarks[33].x * frame_width,
                                landmarks[33].y * frame_height])
    left_eye_inner = np.array([landmarks[133].x * frame_width,
                                landmarks[133].y * frame_height])

    # 3. Axe horizontal de l'oeil (vecteur entre coin interne et externe)
    eye_axis = left_eye_inner - left_eye_outer

    # 4. Centre géométrique de l'oeil
    eye_center = (left_eye_inner + left_eye_outer) / 2

    # 5. Décalage entre l'iris et le centre de l'oeil
    gaze_offset = left_iris_center - eye_center

    # 6. Normalisation par la longueur de l'oeil
    norm = np.linalg.norm(eye_axis)
    gaze_vec = gaze_offset / norm if norm > 1e-6 else np.zeros_like(gaze_offset)

    return eye_center, gaze_vec