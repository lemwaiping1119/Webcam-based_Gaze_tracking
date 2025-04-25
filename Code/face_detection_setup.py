import mediapipe as mp

# -------------------------
# MediaPipe Face Detection Setup
# -------------------------
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.75)
