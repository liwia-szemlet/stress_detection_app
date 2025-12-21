import time
from collections import deque

import cv2
import numpy as np
import mediapipe as mp


class FaceTracker:
    def __init__(self, user_id="user_1"):
        self.user_id = user_id
        self.last_features = None
        self.callbacks = []

        # Okna czasowe (~1 s przy ~30 FPS)
        self.history_brow = deque(maxlen=30)
        self.history_mouth = deque(maxlen=30)
        self.history_eye = deque(maxlen=30)

        # Face Mesh (468 punktów)
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # face oval (obrys twarzy) [web:277][web:299]
        self.face_oval_indices = [
            10, 338, 297, 332, 284, 251, 389, 356,
            454, 323, 361, 288, 397, 365, 379, 378,
            400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21,
            54, 103, 67, 109,
        ]

    def close(self):
        if self.face_mesh is not None:
            self.face_mesh.close()

    def register_callback(self, fn):
        self.callbacks.append(fn)

    def _notify_callbacks(self, features):
        for fn in self.callbacks:
            fn(features)

    def process(self, frame_bgr):
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            self._process_landmarks(frame_bgr, face_landmarks)

            if self.last_features is not None:
                self._draw_emotion_label(
                    frame_bgr,
                    face_landmarks,
                    self.last_features.get("emotion_label", "neutral"),
                )

        return frame_bgr

    def _process_landmarks(self, frame, face_landmarks):
        h, w, _ = frame.shape
        for lm in face_landmarks:
            x = int(lm.x * w)
            y = int(lm.y * h)
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

        features = self._compute_features(face_landmarks, frame.shape)
        self.last_features = features
        self._notify_callbacks(features)

    def _compute_features(self, face_landmarks, frame_shape):
        h, w, _ = frame_shape
        points = []
        for lm in face_landmarks:
            x = lm.x * w
            y = lm.y * h
            points.append([x, y])
        points = np.array(points)

        # Kluczowe indeksy (Face Mesh 468) [web:188][web:231]
        left_eye_outer = points[33]
        right_eye_outer = points[263]
        left_eye_top = points[159]
        left_eye_bottom = points[145]
        right_eye_top = points[386]
        right_eye_bottom = points[374]

        mouth_top = points[13]
        mouth_bottom = points[14]
        mouth_left = points[61]
        mouth_right = points[291]
        chin = points[152]
        nose_base = points[1]

        left_brow = points[70]
        right_brow = points[300]

        face_width = np.linalg.norm(right_eye_outer - left_eye_outer)

        def norm_dist(p1, p2):
            return float(np.linalg.norm(p1 - p2) / (face_width + 1e-6))

        # Cechy geometryczne
        brow_furrow = norm_dist(left_brow, left_eye_outer) + norm_dist(right_brow, right_eye_outer)
        mouth_open = norm_dist(mouth_top, mouth_bottom)
        mouth_width = norm_dist(mouth_left, mouth_right)
        jaw_clench = norm_dist(chin, nose_base)
        left_eye_open = norm_dist(left_eye_top, left_eye_bottom)
        right_eye_open = norm_dist(right_eye_top, right_eye_bottom)
        eye_open_avg = 0.5 * (left_eye_open + right_eye_open)

        eye_line = right_eye_outer - left_eye_outer
        head_tilt = float(np.arctan2(eye_line[1], eye_line[0]))

        # Historia czasowa
        self.history_brow.append(brow_furrow)
        self.history_mouth.append(mouth_open)
        self.history_eye.append(eye_open_avg)

        brow_mean = float(np.mean(self.history_brow)) if self.history_brow else brow_furrow
        mouth_mean = float(np.mean(self.history_mouth)) if self.history_mouth else mouth_open
        eye_mean = float(np.mean(self.history_eye)) if self.history_eye else eye_open_avg

        # ---------------- STRESS INDEX (0–100) ----------------

        def clamp01(x):
            return float(np.clip(x, 0.0, 1.0))

        # Heurystyczne zakresy:
        # brow_furrow: 0.30 .. 0.70
        # eye_open_avg: 0.05 .. 0.25
        # jaw_clench: 0.10 .. 0.28
        # mouth_open: 0.00 .. 0.25
        brow_norm = clamp01((0.70 - brow_mean) / (0.70 - 0.30))
        eye_norm = clamp01((0.25 - eye_mean) / (0.25 - 0.05))
        jaw_norm = clamp01((0.28 - jaw_clench) / (0.28 - 0.10))
        mouth_norm = clamp01((mouth_open - 0.00) / (0.25 - 0.00))

        # prosty stres: brwi + oczy + szczęka, minus usta
        stress_score = (
            0.4 * brow_norm +
            0.3 * eye_norm +
            0.2 * jaw_norm -
            0.4 * mouth_norm
        )

        stress_score = float(np.clip(stress_score, 0.0, 1.0))
        stress_index_0100 = float(stress_score * 100.0)

        # ---------------- EMOCJE (neutral / sad / happy / angry) ----------------

        emotion = "neutral"

        # HAPPY: uśmiech – łagodniejsze progi.
        # mouth_norm > 0.18 (niewielkie otwarcie ust),
        # eye_norm < 0.9 – oczy nie muszą być bardzo szeroko otwarte.
        if mouth_norm > 0.18 and eye_norm < 0.9:
            emotion = "happy"

        # ANGRY: tylko gdy naprawdę mocne napięcie brwi i oczu, bez dużego uśmiechu.
        if emotion != "happy":
            if brow_norm > 0.6 and eye_norm > 0.6 and mouth_norm < 0.25:
                emotion = "angry"

        # SAD: lekkie napięcie brwi, małe usta, oczy raczej nie zmrużone.
        if emotion == "neutral":
            if brow_norm > 0.4 and mouth_norm < 0.15 and eye_norm < 0.5:
                emotion = "sad"

        # jeśli stres bardzo niski – zostaw neutral lub happy, wyczyść sad/angry
        if stress_index_0100 < 10.0 and emotion not in ("happy", "neutral"):
            emotion = "neutral"

        features = {
            "user_id": self.user_id,
            "timestamp": time.time(),
            "brow_furrow": brow_furrow,
            "mouth_open": mouth_open,
            "mouth_width": mouth_width,
            "jaw_clench": jaw_clench,
            "eye_open_avg": eye_open_avg,
            "head_tilt": head_tilt,
            "brow_mean": brow_mean,
            "mouth_mean": mouth_mean,
            "eye_mean": eye_mean,
            "stress_index_0100": stress_index_0100,
            "emotion_label": emotion,
        }

        return features

    def _draw_emotion_label(self, frame, face_landmarks, emotion_label: str):
        """
        Ramka na całą twarz + pasek z napisem na górze.
        """
        h, w, _ = frame.shape

        xs = []
        ys = []
        for idx in self.face_oval_indices:
            lm = face_landmarks[idx]
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))

        if not xs or not ys:
            return

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        pad_x = int(0.10 * (max_x - min_x + 1))
        pad_y = int(0.10 * (max_y - min_y + 1))

        box_x1 = max(min_x - pad_x, 0)
        box_x2 = min(max_x + pad_x, w)
        box_y1 = max(min_y - pad_y, 0)
        box_y2 = min(max_y + pad_y, h)

        cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (255, 255, 255), thickness=2)

        label_height = 32
        label_y1 = box_y1
        label_y2 = min(box_y1 + label_height, box_y2)

        color_map = {
            "neutral": (200, 200, 200),
            "sad": (200, 160, 0),
            "angry": (0, 0, 255),
            "happy": (0, 200, 0),
        }
        color = color_map.get(emotion_label, (200, 200, 200))

        cv2.rectangle(frame, (box_x1, label_y1), (box_x2, label_y2), color, thickness=-1)

        text = emotion_label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = box_x1 + (box_x2 - box_x1 - text_size[0]) // 2
        text_y = label_y1 + (label_y2 - label_y1 + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def get_features(self):
        return self.last_features
